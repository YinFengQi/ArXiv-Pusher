import os
import subprocess
import time
from typing import Any, Dict

import requests
from bs4 import BeautifulSoup
from loguru import logger
from PyPDF2 import PdfReader

from config import GENERAL_CONFIG
from arxiv import Client, Search, SortCriterion, SortOrder
from datetime import datetime, timedelta


def fetch_papers(arxiv_categories: list[str]) -> list[dict[str, Any]]:
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
        num_retries=5,
    )
    search = Search(
        query=search_query,
        sort_by=SortCriterion.SubmittedDate,
        sort_order=SortOrder.Descending,
        max_results=100,
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
            papers.append(
                {
                    "title": result.title,
                    "url": result.entry_id,
                    "pdf_url": result.pdf_url,
                    "abstract": result.summary,
                    "authors": [a.name for a in result.authors],
                    "published": result.published,
                    "categories": [c for c in result.categories],
                    "primary_category": result.primary_category
                    if result.primary_category
                    else None,
                }
            )

    logger.success(
        f"找到 {len(papers)} 篇论文（发表于 {target_date.strftime('%Y-%m-%d')} 之后）"
    )
    return papers


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
    if "arxiv.org" in url and not url.endswith(".pdf"):
        paper_id = url.split("/")[-1]
        url = f"https://arxiv.org/pdf/{paper_id}.pdf"

    logger.info(f"下载 PDF: {url}")

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                content_type = response.headers.get("Content-Type", "")
                if "pdf" not in content_type.lower() and len(response.content) < 10000:
                    logger.warning(
                        f"响应可能不是 PDF 文件 (Content-Type: {content_type})"
                    )

                with open(filename, "wb") as f:
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
            logger.warning(f"尝试 {attempt + 1}/{max_retries} 失败: {str(e)}")

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
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    logger.warning(f"无法提取第 {page_num + 1} 页: {str(e)}")
    except Exception as e:
        logger.error(f"PDF 解析失败: {str(e)}")

    return text


def download_pdf_and_extract_text(paper: Dict[str, Any], user_dir: str) -> str:
    """下载 PDF 并提取文本"""
    pdf_path = f"{user_dir}/{paper['title']}.pdf"
    if download_pdf(paper["pdf_url"], pdf_path):
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
        url = paper["url"]
        if "arxiv.org" in url:
            paper_id = url.split("/")[-1]
            html_url = f"https://arxiv.org/html/{paper_id}"
        else:
            html_url = url.replace(".pdf", ".html")

        logger.info(f"下载 HTML: {html_url}")
        response = requests.get(html_url, timeout=30)

        if response.status_code == 200:
            temp_html_path = f"{user_dir}/{paper['title']}_temp.html"
            with open(temp_html_path, "wb") as f:
                f.write(response.content)

            # 尝试使用 wkhtmltopdf 将 HTML 转换为 PDF
            pdf_path = f"{user_dir}/{paper['title']}_from_html.pdf"
            try:
                subprocess.run(
                    ["wkhtmltopdf", temp_html_path, pdf_path], check=True, timeout=60
                )
                logger.info(f"HTML 已转换为 PDF: {pdf_path}")

                pdf_text = extract_text_from_pdf(pdf_path)
                if pdf_text and len(pdf_text) > 1000:
                    return pdf_text
            except Exception as pdf_err:
                logger.error(f"HTML 转 PDF 失败: {str(pdf_err)}")

            # 如果 PDF 转换失败，直接从 HTML 提取
            soup = BeautifulSoup(response.content, "html.parser")

            for script in soup(["script", "style"]):
                script.extract()

            text = soup.get_text(separator="\n", strip=True)
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)

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
        logger.warning("文本内容过长，截断到前 129024 字符")
        text = text[:129024]

    # 方式3: 如果所有方法都失败，使用摘要
    if not text:
        logger.warning("所有提取方式失败，使用摘要")
        text = paper["abstract"]

    return text
