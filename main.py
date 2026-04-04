import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

from loguru import logger

from ai_processing import gpt_check_interest, gpt_summarize, log_token_cost
from config import AI_CONFIG, GENERAL_CONFIG, USERS_CONFIG
from content_typsetting import build_filtered_papers_appendix
from email_utils import send_email
from paper_utils import fetch_papers, get_paper_text


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
    papers_processed_count = 0

    # 为每个用户创建独立的临时目录
    user_dir = f"temp/{user_name.replace(' ', '_')}"
    os.makedirs(user_dir, exist_ok=True)

    # 获取该用户关注的论文
    papers = fetch_papers(arxiv_categories)

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
            logger.info(f"过滤论文 {i + 1}/{len(papers)}: {paper['title']}")
            try:
                is_interested, token_stats = gpt_check_interest(
                    paper["abstract"], interest_filter_prompt
                )
                if is_interested:
                    logger.info("✓ 用户可能对此论文感兴趣")
                    return ("interested", paper, token_stats)
                else:
                    logger.info("✗ 用户可能对此论文不感兴趣，跳过")
                    return ("not_interested", paper, token_stats)
            except Exception as e:
                logger.error(f"过滤论文时出错: {str(e)}，保留该论文")
                return ("error", paper, None)

        # 使用线程池进行并发过滤（降低并发数避免API限流）
        with ThreadPoolExecutor(max_workers=3) as executor:
            # 提交所有任务
            future_to_paper = {
                executor.submit(filter_single_paper, (i, paper)): paper
                for i, paper in enumerate(papers)
            }

            # 收集结果
            for future in as_completed(future_to_paper):
                try:
                    result_type, paper, token_stats = future.result()

                    # 累计token使用
                    if token_stats:
                        filter_input_tokens += token_stats["prompt_tokens"]
                        filter_output_tokens += token_stats["completion_tokens"]

                    if result_type == "interested" or result_type == "error":
                        filtered_papers.append(paper)
                    else:  # not_interested
                        filtered_out_papers.append(paper)

                except Exception as e:
                    logger.error(f"处理过滤结果时出错: {str(e)}")

        papers = filtered_papers
        logger.info(
            f"兴趣过滤完成，剩余 {len(papers)} 篇论文，过滤掉 {len(filtered_out_papers)} 篇论文"
        )

        if not papers:
            logger.info(f"用户 {user_name} 经过兴趣过滤后没有感兴趣的论文")
            # 计算成本
            filter_input_cost = (filter_input_tokens / 1_000_000) * AI_CONFIG.get(
                "price_per_million_input_tokens", 0
            )
            filter_output_cost = (filter_output_tokens / 1_000_000) * AI_CONFIG.get(
                "price_per_million_output_tokens", 0
            )

            # 输出成本统计
            log_token_cost(
                user_name,
                filter_input_tokens,
                filter_output_tokens,
                generate_input_tokens,
                generate_output_tokens,
            )
            # 即使没有感兴趣的论文，如果有被过滤的论文，也发送附录
            if filtered_out_papers:
                filtered_appendix = build_filtered_papers_appendix(filtered_out_papers)
                send_email(
                    f"{date.today()} ArXiv 论文报告 - {user_name}",
                    filtered_appendix,
                    user_email,
                )
            return
        # 没有配置兴趣过滤，所有论文都通过

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
            generate_input_tokens += token_stats["prompt_tokens"]
            generate_output_tokens += token_stats["completion_tokens"]
            papers_processed_count += 1

            # 构建报告
            report.append(f"""
## 论文标题

{paper["title"]}

## 论文信息
* **作者**: {", ".join(paper["authors"])}
* **发表日期**: {paper["published"].strftime("%Y-%m-%d")}
* **链接**: [{paper["url"]}]({paper["url"]})
* **主要分类**: {paper["primary_category"] if "primary_category" in paper else "未知分类"}
* **所属分类**: {paper["categories"] if "categories" in paper else "未知分类"}
* **摘要原文**:

{paper["abstract"]}


## 论文总结
{summary}

---
""")
        except Exception as e:
            logger.error(f"处理论文失败: {paper['title']}，错误: {str(e)}")
            report.append(f"处理论文失败: {paper['title']}，错误: {str(e)}")

    # 输出用户的token使用统计和成本
    log_token_cost(
        user_name,
        filter_input_tokens,
        filter_output_tokens,
        generate_input_tokens,
        generate_output_tokens,
    )

    # 计算成本
    filter_input_cost = (filter_input_tokens / 1_000_000) * AI_CONFIG.get(
        "price_per_million_input_tokens", 0
    )
    filter_output_cost = (filter_output_tokens / 1_000_000) * AI_CONFIG.get(
        "price_per_million_output_tokens", 0
    )
    filter_cost = filter_input_cost + filter_output_cost

    if report:
        # 构建完整报告，包括被过滤论文的附录
        full_report = "\n".join(report)

        # 如果有被过滤掉的论文，添加附录
        if filtered_out_papers:
            full_report += "\n\n" + build_filtered_papers_appendix(filtered_out_papers)

        # 发送给该用户
        send_email(
            f"{date.today()} ArXiv 论文报告 - {user_name}", full_report, user_email
        )

        # 保存报告到用户专属文件
        report_file = f"{user_dir}/report.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(full_report)
        logger.success(f"用户 {user_name} 的报告已发送并保存到 {report_file}")


def daily_job():
    """每日任务：为所有配置的用户处理论文"""
    os.makedirs("temp", exist_ok=True)

    logger.info(f"开始每日任务，共有 {len(USERS_CONFIG)} 个用户")

    for i, user_config in enumerate(USERS_CONFIG):
        try:
            process_user(user_config)
            # 在处理用户之间添加延迟，避免ArXiv API限流
            if i < len(USERS_CONFIG) - 1:
                logger.info("等待60秒后处理下一个用户，避免API限流...")
                time.sleep(60)
        except Exception as e:
            logger.error(f"处理用户 {user_config['name']} 时发生错误: {str(e)}")

    logger.success("所有用户处理完成")


if __name__ == "__main__":
    # 配置loguru
    logger.add("arxiv_pusher.log", rotation="10 MB", level="INFO", encoding="utf-8")
    daily_job()
