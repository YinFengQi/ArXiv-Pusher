from loguru import logger


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
        appendix.append("\n---\n")

    return "".join(appendix)
