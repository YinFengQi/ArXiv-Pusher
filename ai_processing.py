from typing import Dict, Tuple

import openai
from loguru import logger

from config import AI_CONFIG, DEFAULT_PROMPT_TEMPLATE


def gpt_check_interest(
    abstract: str, interest_filter_prompt: str
) -> Tuple[bool, Dict[str, int]]:
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
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }

        answer = response.choices[0].message.content.strip().lower()
        logger.info(f"兴趣判断: {answer}")

        interested = any(
            kw in answer for kw in ["是", "yes", "感兴趣", "有兴趣", "interested"]
        )
        not_interested = any(
            kw in answer for kw in ["否", "no", "不感兴趣", "无兴趣", "not interested"]
        )

        if interested and not not_interested:
            return True, token_stats
        elif not_interested and not interested:
            return False, token_stats
        else:
            logger.warning(f"无法明确判断，默认感兴趣: {answer}")
            return True, token_stats

    except Exception as e:
        logger.error(f"兴趣判断失败: {e}，默认感兴趣")
        return True, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def gpt_summarize(text, custom_prompt=None):
    """使用 AI 对论文进行总结

    Args:
        text: 论文全文
        custom_prompt: 自定义提示词模板（可选）

    Returns:
        (总结内容, token统计字典)
    """
    prompt = (
        custom_prompt.format(text=text)
        if custom_prompt
        else DEFAULT_PROMPT_TEMPLATE.format(text=text)
    )
    client = openai.OpenAI(base_url=AI_CONFIG["base_url"], api_key=AI_CONFIG["api_key"])

    logger.info(f"请求 AI 总结论文，长度: {len(text)}")
    response = client.chat.completions.create(
        model=AI_CONFIG["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=1.5,
    )

    usage = response.usage
    content = response.choices[0].message.content

    assert content is not None and usage is not None

    token_stats = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }
    logger.info(f"总结完成，响应长度: {len(content)}")

    return content, token_stats


def log_token_cost(
    user_name,
    filter_input_tokens,
    filter_output_tokens,
    generate_input_tokens,
    generate_output_tokens,
):
    """记录 token 使用情况和成本"""
    filter_total = filter_input_tokens + filter_output_tokens
    generate_total = generate_input_tokens + generate_output_tokens
    total_input = filter_input_tokens + generate_input_tokens
    total_output = filter_output_tokens + generate_output_tokens
    total = total_input + total_output

    # 计算成本（元）
    price_input = AI_CONFIG.get("price_per_million_input_tokens", 0)
    price_output = AI_CONFIG.get("price_per_million_output_tokens", 0)

    filter_cost = (
        filter_input_tokens * price_input + filter_output_tokens * price_output
    ) / 1_000_000
    generate_cost = (
        generate_input_tokens * price_input + generate_output_tokens * price_output
    ) / 1_000_000
    total_cost = filter_cost + generate_cost

    logger.info("=" * 80)
    logger.info(f"【{user_name}】Token 使用统计:")
    logger.info(
        f"过滤阶段: {filter_total:,} tokens (输入 {filter_input_tokens:,}, 输出 {filter_output_tokens:,}), 成本 ¥{filter_cost:.4f}"
    )
    logger.info(
        f"生成阶段: {generate_total:,} tokens (输入 {generate_input_tokens:,}, 输出 {generate_output_tokens:,}), 成本 ¥{generate_cost:.4f}"
    )
    logger.info(
        f"总计: {total:,} tokens (输入 {total_input:,}, 输出 {total_output:,}), 总成本 ¥{total_cost:.4f}"
    )
    logger.info("=" * 80)
