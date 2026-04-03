import os
import requests
import smtplib
import socket
import time
import subprocess
from datetime import datetime, timedelta, date
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple, Any
from email.mime.text import MIMEText

from arxiv import Client, Search, SortCriterion, SortOrder
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import openai
import markdown2
from loguru import logger
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from config import AI_CONFIG, EMAIL_SERVER_CONFIG, GENERAL_CONFIG, USERS_CONFIG, DEFAULT_PROMPT_TEMPLATE
from database import get_db


# ============================================================================
# 邮件发送
# ============================================================================

def send_email(subject: str, content: str, receiver_email: str) -> bool:
    """发送邮件通知

    Args:
        subject: 邮件主题
        content: Markdown 格式的邮件内容
        receiver_email: 接收者邮箱，多个邮箱用逗号分隔

    Returns:
        bool: 发送成功返回 True，失败返回 False
    """
    html_content = markdown2.markdown(content, extras=["tables", "latex", "fenced-code-blocks"])
    msg = MIMEText(html_content, "html", "utf-8")
    msg["Subject"] = subject
    msg["From"] = "arxiv-bot@noaddress.com"
    msg["To"] = receiver_email

    server = None
    try:
        logger.info("正在连接SMTP服务器，发送邮件...")
        server = smtplib.SMTP(
            EMAIL_SERVER_CONFIG["smtp_server"],
            EMAIL_SERVER_CONFIG["smtp_port"],
            timeout=10
        )
        server.starttls()
        server.login(EMAIL_SERVER_CONFIG["sender"], EMAIL_SERVER_CONFIG["password"])

        # 处理多个接收者
        receivers = receiver_email.split(",") if "," in receiver_email else [receiver_email]
        server.sendmail(EMAIL_SERVER_CONFIG["sender"], receivers, msg.as_string())

        logger.success("邮件发送成功")
        return True

    except socket.timeout:
        logger.warning("连接SMTP服务器超时")
        return False
    except smtplib.SMTPException as e:
        logger.error(f"SMTP错误: {e.smtp_error.decode() if hasattr(e, 'smtp_error') else str(e)}")
        return False
    except Exception as e:
        logger.error(f"邮件发送失败: {type(e).__name__} - {str(e)}")
        return False
    finally:
        if server:
            try:
                server.quit()
            except Exception:
                pass


# ============================================================================
# 论文获取（ArXiv API）
# ============================================================================

def fetch_papers(arxiv_categories: List[str]) -> List[Dict[str, Any]]:
    """获取指定分类的论文

    Args:
        arxiv_categories: ArXiv 分类列表，如 ["physics.atom-ph", "physics.optics"]

    Returns:
        论文列表，每篇论文包含标题、URL、摘要、作者等信息
    """
    search_query = " OR ".join([f"cat:{cat}" for cat in arxiv_categories])
    client = Client(
        page_size=50,
        delay_seconds=3,  # 避免被限流
        num_retries=5
    )
    search = Search(
        query=search_query,
        sort_by=SortCriterion.SubmittedDate,
        sort_order=SortOrder.Descending,
        max_results=100
    )

    papers = []
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    target_date = today - timedelta(days=GENERAL_CONFIG["days_lookback"])

    # 如果目标日期是周末，回退到上个工作日（周五）
    weekday = target_date.weekday()
    if weekday >= 5:
        target_date -= timedelta(days=weekday - 4)

    logger.info(f"获取论文，目标日期: {target_date.strftime('%Y-%m-%d')}")

    for result in client.results(search):
        published_dt = result.published.replace(tzinfo=None)
        if target_date <= published_dt:
            papers.append({
                "title": result.title,
                "url": result.entry_id,
                "pdf_url": result.pdf_url,
                "abstract": result.summary,
                "authors": [a.name for a in result.authors],
                "published": result.published,
                "categories": [c for c in result.categories],
                "primary_category": result.primary_category if result.primary_category else None
            })

    logger.success(f"找到 {len(papers)} 篇论文（发表于 {target_date.strftime('%Y-%m-%d')} 之后）")
    return papers


# ============================================================================
# 文本提取（PDF/HTML）
# ============================================================================

def download_pdf(url: str, filename: str, max_retries: int = 3) -> bool:
    """下载 PDF 文件

    Args:
        url: PDF 文件 URL
        filename: 保存的文件名
        max_retries: 最大重试次数

    Returns:
        下载成功返回 True，失败返回 False
    """
    # 确保 URL 是正确的 PDF 链接
    if 'arxiv.org' in url and not url.endswith('.pdf'):
        paper_id = url.split('/')[-1]
        url = f"https://arxiv.org/pdf/{paper_id}.pdf"

    logger.info(f"下载 PDF: {url}")

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '')
                if 'pdf' not in content_type.lower() and len(response.content) < 10000:
                    logger.warning(f"响应可能不是 PDF 文件 (Content-Type: {content_type})")

                with open(filename, 'wb') as f:
                    f.write(response.content)

                # 验证文件大小
                file_size = os.path.getsize(filename)
                if file_size < 1000:
                    logger.warning(f"下载的文件过小 ({file_size} 字节)")
                    continue

                return True
            else:
                logger.error(f"下载失败: HTTP {response.status_code}")

        except Exception as e:
            logger.warning(f"尝试 {attempt+1}/{max_retries} 失败: {str(e)}")

        # 指数退避
        if attempt < max_retries - 1:
            time.sleep(2 * (attempt + 1))

    return False


def extract_text_from_pdf(pdf_path: str) -> str:
    """从 PDF 提取文本

    Args:
        pdf_path: PDF 文件路径

    Returns:
        提取的文本内容
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    logger.warning(f"无法提取第 {page_num+1} 页: {str(e)}")
    except Exception as e:
        logger.error(f"PDF 解析失败: {str(e)}")

    return text


def download_pdf_and_extract_text(paper: Dict[str, Any], user_dir: str) -> str:
    """下载 PDF 并提取文本"""
    pdf_path = f"{user_dir}/{paper['title']}.pdf"
    if download_pdf(paper['pdf_url'], pdf_path):
        text = extract_text_from_pdf(pdf_path)
        if not text:
            logger.warning(f"无法从 {paper['title']} 提取文本")
        return text
    else:
        logger.error(f"无法下载 {paper['title']} 的 PDF")
        return ""


def download_html_and_extract_text(paper: Dict[str, Any], user_dir: str) -> str:
    """从 ArXiv 下载 HTML 版本并提取文本"""
    try:
        url = paper['url']
        if 'arxiv.org' in url:
            paper_id = url.split('/')[-1]
            html_url = f"https://arxiv.org/html/{paper_id}"
        else:
            html_url = url.replace('.pdf', '.html')

        logger.info(f"下载 HTML: {html_url}")
        response = requests.get(html_url, timeout=30)

        if response.status_code == 200:
            temp_html_path = f"{user_dir}/{paper['title']}_temp.html"
            with open(temp_html_path, 'wb') as f:
                f.write(response.content)

            # 尝试使用 wkhtmltopdf 将 HTML 转换为 PDF
            pdf_path = f"{user_dir}/{paper['title']}_from_html.pdf"
            try:
                subprocess.run(['wkhtmltopdf', temp_html_path, pdf_path],
                              check=True, timeout=60)
                logger.info(f"HTML 已转换为 PDF: {pdf_path}")

                pdf_text = extract_text_from_pdf(pdf_path)
                if pdf_text and len(pdf_text) > 1000:
                    return pdf_text
            except Exception as pdf_err:
                logger.error(f"HTML 转 PDF 失败: {str(pdf_err)}")

            # 如果 PDF 转换失败，直接从 HTML 提取
            soup = BeautifulSoup(response.content, 'html.parser')

            for script in soup(["script", "style"]):
                script.extract()

            text = soup.get_text(separator="\n", strip=True)
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)

            logger.info(f"从 HTML 提取了 {len(text)} 字符的文本")
            return text
        else:
            logger.error(f"HTML 下载失败: HTTP {response.status_code}")
            return ""
    except Exception as e:
        logger.error(f"HTML 处理错误: {str(e)}")
        return ""


def get_paper_text(paper: Dict[str, Any], user_dir: str) -> str:
    """尝试多种方式获取论文文本内容

    优先级：PDF -> HTML -> 摘要

    Args:
        paper: 论文信息字典
        user_dir: 用户临时目录

    Returns:
        论文文本内容
    """
    # 方式1: 尝试 PDF
    text = download_pdf_and_extract_text(paper, user_dir)

    # 方式2: 如果 PDF 失败或内容太少，尝试 HTML
    if not text or len(text) < 1000:
        logger.info("PDF 提取失败或内容太少，尝试 HTML 方式")
        text = download_html_and_extract_text(paper, user_dir)

    # 截断过长的文本（避免超出 API 限制）
    if len(text) > 129024:
        logger.warning(f"文本内容过长，截断到前 129024 字符")
        text = text[:129024]

    # 方式3: 如果所有方法都失败，使用摘要
    if not text:
        logger.warning("所有提取方式失败，使用摘要")
        text = paper['abstract']

    return text


# ============================================================================
# AI 调用（过滤/总结）
# ============================================================================

def gpt_check_interest(abstract: str, interest_filter_prompt: str) -> Tuple[bool, Dict[str, int]]:
    """使用 AI 判断用户是否对论文感兴趣

    Args:
        abstract: 论文摘要
        interest_filter_prompt: 兴趣过滤提示词，需包含 {abstract} 占位符

    Returns:
        (是否感兴趣, token统计字典)
    """
    prompt = interest_filter_prompt.format(abstract=abstract)
    client = openai.OpenAI(base_url=AI_CONFIG["base_url"], api_key=AI_CONFIG["api_key"])

    try:
        response = client.chat.completions.create(
            model=AI_CONFIG["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        usage = response.usage
        token_stats = {
            'prompt_tokens': usage.prompt_tokens,
            'completion_tokens': usage.completion_tokens,
            'total_tokens': usage.total_tokens
        }

        answer = response.choices[0].message.content.strip().lower()
        logger.info(f"兴趣判断: {answer}")

        interested = any(kw in answer for kw in ['是', 'yes', '感兴趣', '有兴趣', 'interested'])
        not_interested = any(kw in answer for kw in ['否', 'no', '不感兴趣', '无兴趣', 'not interested'])

        if interested and not not_interested:
            return True, token_stats
        elif not_interested and not interested:
            return False, token_stats
        else:
            logger.warning(f"无法明确判断，默认感兴趣: {answer}")
            return True, token_stats

    except Exception as e:
        logger.error(f"兴趣判断失败: {e}，默认感兴趣")
        return True, {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

def gpt_summarize(text, custom_prompt=None):
    """使用 AI 对论文进行总结

    Args:
        text: 论文全文
        custom_prompt: 自定义提示词模板（可选）

    Returns:
        (总结内容, token统计字典)
    """
    prompt = custom_prompt.format(text=text) if custom_prompt else DEFAULT_PROMPT_TEMPLATE.format(text=text)
    client = openai.OpenAI(base_url=AI_CONFIG["base_url"], api_key=AI_CONFIG["api_key"])

    logger.info(f"请求 AI 总结论文，长度: {len(text)}")
    response = client.chat.completions.create(
        model=AI_CONFIG["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=1.5,
    )

    usage = response.usage
    token_stats = {
        'prompt_tokens': usage.prompt_tokens,
        'completion_tokens': usage.completion_tokens,
        'total_tokens': usage.total_tokens
    }
    logger.info(f"总结完成，响应长度: {len(response.choices[0].message.content)}")

    return response.choices[0].message.content, token_stats

def _log_token_cost(user_name, filter_input_tokens, filter_output_tokens,
                    generate_input_tokens, generate_output_tokens):
    """记录 token 使用情况和成本"""
    filter_total = filter_input_tokens + filter_output_tokens
    generate_total = generate_input_tokens + generate_output_tokens
    total_input = filter_input_tokens + generate_input_tokens
    total_output = filter_output_tokens + generate_output_tokens
    total = total_input + total_output

    # 计算成本（元）
    price_input = AI_CONFIG.get("price_per_million_input_tokens", 0)
    price_output = AI_CONFIG.get("price_per_million_output_tokens", 0)

    filter_cost = (filter_input_tokens * price_input + filter_output_tokens * price_output) / 1_000_000
    generate_cost = (generate_input_tokens * price_input + generate_output_tokens * price_output) / 1_000_000
    total_cost = filter_cost + generate_cost

    logger.info("=" * 80)
    logger.info(f"【{user_name}】Token 使用统计:")
    logger.info(f"过滤阶段: {filter_total:,} tokens (输入 {filter_input_tokens:,}, 输出 {filter_output_tokens:,}), 成本 ¥{filter_cost:.4f}")
    logger.info(f"生成阶段: {generate_total:,} tokens (输入 {generate_input_tokens:,}, 输出 {generate_output_tokens:,}), 成本 ¥{generate_cost:.4f}")
    logger.info(f"总计: {total:,} tokens (输入 {total_input:,}, 输出 {total_output:,}), 总成本 ¥{total_cost:.4f}")
    logger.info("=" * 80)

def build_filtered_papers_appendix(filtered_out_papers):
    """构建被过滤论文的附录"""
    if not filtered_out_papers:
        return ""

    appendix = ["\n\n" + "=" * 80, "\n## 📋 附录：其他论文（未通过兴趣过滤）\n"]

    for i, paper in enumerate(filtered_out_papers, 1):
        appendix.append(f"\n### {i}. {paper['title']}\n")
        appendix.append(f"**作者**: {', '.join(paper['authors'])}\n")
        appendix.append(f"**发表日期**: {paper['published'].strftime('%Y-%m-%d')}\n")
        appendix.append(f"**链接**: [{paper['url']}]({paper['url']})\n")
        appendix.append(f"**主要分类**: {paper.get('primary_category', '未知分类')}\n")
        appendix.append(f"\n**摘要**:\n{paper['abstract']}\n")
        appendix.append("\n" + "─" * 80 + "\n")

    return ''.join(appendix)

def process_user(user_config):
    """处理单个用户的论文获取和报告生成"""
    user_name = user_config["name"]
    user_email = user_config["email"]
    arxiv_categories = user_config["arxiv_categories"]
    custom_prompt = user_config.get("custom_prompt", None)
    interest_filter_prompt = user_config.get("interest_filter_prompt", None)

    logger.info(f"开始处理用户: {user_name}")

    # 初始化token统计 - 分阶段统计
    filter_input_tokens = 0
    filter_output_tokens = 0
    generate_input_tokens = 0
    generate_output_tokens = 0

    # 初始化论文数量统计
    papers_fetched = 0
    papers_filtered_count = 0
    papers_processed_count = 0

    # 为每个用户创建独立的临时目录
    user_dir = f"temp/{user_name.replace(' ', '_')}"
    os.makedirs(user_dir, exist_ok=True)

    # 获取该用户关注的论文
    papers = fetch_papers(arxiv_categories)
    papers_fetched = len(papers)

    if not papers:
        logger.info(f"用户 {user_name} 没有找到新论文")
        return

    # 第一步：如果配置了兴趣过滤提示词，先根据摘要过滤论文
    filtered_out_papers = []  # 存储被过滤掉的论文
    if interest_filter_prompt:
        logger.info(f"开始使用兴趣过滤（并发模式），共 {len(papers)} 篇论文待过滤")
        filtered_papers = []

        # 定义单个论文过滤任务
        def filter_single_paper(paper_with_index):
            i, paper = paper_with_index
            logger.info(f"过滤论文 {i+1}/{len(papers)}: {paper['title']}")
            try:
                is_interested, token_stats = gpt_check_interest(paper['abstract'], interest_filter_prompt)
                if is_interested:
                    logger.info(f"✓ 用户可能对此论文感兴趣")
                    return ('interested', paper, token_stats)
                else:
                    logger.info(f"✗ 用户可能对此论文不感兴趣，跳过")
                    return ('not_interested', paper, token_stats)
            except Exception as e:
                logger.error(f"过滤论文时出错: {str(e)}，保留该论文")
                return ('error', paper, None)

        # 使用线程池进行并发过滤（降低并发数避免API限流）
        with ThreadPoolExecutor(max_workers=3) as executor:
            # 提交所有任务
            future_to_paper = {executor.submit(filter_single_paper, (i, paper)): paper
                              for i, paper in enumerate(papers)}

            # 收集结果
            for future in as_completed(future_to_paper):
                try:
                    result_type, paper, token_stats = future.result()

                    # 累计token使用
                    if token_stats:
                        filter_input_tokens += token_stats['prompt_tokens']
                        filter_output_tokens += token_stats['completion_tokens']

                    if result_type == 'interested' or result_type == 'error':
                        filtered_papers.append(paper)
                    else:  # not_interested
                        filtered_out_papers.append(paper)

                except Exception as e:
                    logger.error(f"处理过滤结果时出错: {str(e)}")

        papers = filtered_papers
        papers_filtered_count = len(papers)
        logger.info(f"兴趣过滤完成，剩余 {len(papers)} 篇论文，过滤掉 {len(filtered_out_papers)} 篇论文")

        if not papers:
            logger.info(f"用户 {user_name} 经过兴趣过滤后没有感兴趣的论文")
            # 计算成本
            filter_input_cost = (filter_input_tokens / 1_000_000) * AI_CONFIG.get("price_per_million_input_tokens", 0)
            filter_output_cost = (filter_output_tokens / 1_000_000) * AI_CONFIG.get("price_per_million_output_tokens", 0)
            filter_cost = filter_input_cost + filter_output_cost

            # 记录到数据库
            try:
                db = get_db()
                db.record_usage(
                    user_name=user_name,
                    user_email=user_email,
                    arxiv_categories=arxiv_categories,
                    filter_input_tokens=filter_input_tokens,
                    filter_output_tokens=filter_output_tokens,
                    generate_input_tokens=0,
                    generate_output_tokens=0,
                    filter_cost=filter_cost,
                    generate_cost=0.0,
                    papers_fetched=papers_fetched,
                    papers_filtered=0,
                    papers_processed=0
                )
            except Exception as e:
                logger.error(f"记录数据库失败: {str(e)}")

            # 输出成本统计
            _log_token_cost(user_name, filter_input_tokens, filter_output_tokens,
                           generate_input_tokens, generate_output_tokens)
            # 即使没有感兴趣的论文，如果有被过滤的论文，也发送附录
            if filtered_out_papers:
                filtered_appendix = build_filtered_papers_appendix(filtered_out_papers)
                send_email(f"{date.today()} ArXiv 论文报告 - {user_name}", filtered_appendix, user_email)
            return
    else:
        # 没有配置兴趣过滤，所有论文都通过
        papers_filtered_count = len(papers)

    # 第二步：根据配置限制处理的论文数量（硬截断）
    max_papers = GENERAL_CONFIG.get("max_papers_per_user", None)
    if max_papers is not None and max_papers > 0:
        papers = papers[:max_papers]
        logger.info(f"应用硬截断，用户 {user_name} 最多处理 {max_papers} 篇论文")

    report = []
    papers_processed_count = 0
    for paper in papers:
        try:
            # 下载并处理PDF
            text = get_paper_text(paper, user_dir)

            # GPT总结（使用用户自定义提示词）
            summary, token_stats = gpt_summarize(text, custom_prompt)
            # 累计生成阶段token使用
            generate_input_tokens += token_stats['prompt_tokens']
            generate_output_tokens += token_stats['completion_tokens']
            papers_processed_count += 1

            # 构建报告
            report.append(f"""
## 📄论文标题

{paper['title']}

## 📊 论文信息
* **作者**: {', '.join(paper['authors'])}
* **发表日期**: {paper['published'].strftime('%Y-%m-%d')}
* **链接**: [{paper['url']}]({paper['url']})
* **主要分类**: {paper["primary_category"] if "primary_category" in paper else "未知分类"}
* **所属分类**: {paper["categories"] if "categories" in paper else "未知分类"}
* **摘要原文**:

{paper['abstract']}


## 📝 论文总结
{summary}

{'─' * 80}
""")
        except Exception as e:
            logger.error(f"处理论文失败: {paper['title']}，错误: {str(e)}")
            report.append(f"处理论文失败: {paper['title']}，错误: {str(e)}")

    # 输出用户的token使用统计和成本
    _log_token_cost(user_name, filter_input_tokens, filter_output_tokens,
                   generate_input_tokens, generate_output_tokens)

    # 计算成本
    filter_input_cost = (filter_input_tokens / 1_000_000) * AI_CONFIG.get("price_per_million_input_tokens", 0)
    filter_output_cost = (filter_output_tokens / 1_000_000) * AI_CONFIG.get("price_per_million_output_tokens", 0)
    filter_cost = filter_input_cost + filter_output_cost

    generate_input_cost = (generate_input_tokens / 1_000_000) * AI_CONFIG.get("price_per_million_input_tokens", 0)
    generate_output_cost = (generate_output_tokens / 1_000_000) * AI_CONFIG.get("price_per_million_output_tokens", 0)
    generate_cost = generate_input_cost + generate_output_cost

    # 记录到数据库
    try:
        db = get_db()
        db.record_usage(
            user_name=user_name,
            user_email=user_email,
            arxiv_categories=arxiv_categories,
            filter_input_tokens=filter_input_tokens,
            filter_output_tokens=filter_output_tokens,
            generate_input_tokens=generate_input_tokens,
            generate_output_tokens=generate_output_tokens,
            filter_cost=filter_cost,
            generate_cost=generate_cost,
            papers_fetched=papers_fetched,
            papers_filtered=papers_filtered_count,
            papers_processed=papers_processed_count
        )
    except Exception as e:
        logger.error(f"记录数据库失败: {str(e)}")

    if report:
        # 构建完整报告，包括被过滤论文的附录
        full_report = '\n'.join(report)

        # 如果有被过滤掉的论文，添加附录
        if filtered_out_papers:
            full_report += "\n\n" + build_filtered_papers_appendix(filtered_out_papers)

        # 发送给该用户
        send_email(f"{date.today()} ArXiv 论文报告 - {user_name}", full_report, user_email)

        # 保存报告到用户专属文件
        report_file = f"{user_dir}/report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(full_report)
        logger.success(f"用户 {user_name} 的报告已发送并保存到 {report_file}")

def daily_job():
    """每日任务：为所有配置的用户处理论文"""
    os.makedirs('temp', exist_ok=True)

    logger.info(f"开始每日任务，共有 {len(USERS_CONFIG)} 个用户")

    for i, user_config in enumerate(USERS_CONFIG):
        try:
            process_user(user_config)
            # 在处理用户之间添加延迟，避免ArXiv API限流
            if i < len(USERS_CONFIG) - 1:
                logger.info(f"等待60秒后处理下一个用户，避免API限流...")
                time.sleep(60)
        except Exception as e:
            logger.error(f"处理用户 {user_config['name']} 时发生错误: {str(e)}")

    logger.success("所有用户处理完成")

def run_scheduler():
    scheduler = BlockingScheduler()
    scheduler.add_job(
        daily_job, 
        trigger=CronTrigger(hour=16, minute=0),  # 每天下午4点运行
        id='daily_arxiv_job',
        name='Daily ArXiv paper collection and summary'
    )
    
    logger.info("定时任务已设置，每天下午4:00运行")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("定时任务调度器已停止")

if __name__ == "__main__":
    # 配置loguru
    logger.add(
        "arxiv_pusher.log",
        rotation="10 MB",
        level="INFO",
        encoding="utf-8"
    )
    # 如果需要立即运行一次，取消下面的注释
    daily_job()
    
    # 启动定时任务
    # run_scheduler()