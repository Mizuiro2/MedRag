# -*- coding: utf-8 -*-
"""
LLM Reader - 论文 Section 4.4
将检索到的知识链条与用户问题组织成 prompt，调用 LLM 生成最终答案
"""

import requests
from typing import List, Dict

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, LLM_MAX_TOKENS


# 论文 Figure 3 的 Reader Prompt 格式
PROMPT_READER = """### Task Description: You are a medical expert. Based on relevant medical [Background Knowledge] and your medical knowledge, provide professional medical advice for [User Query] while adhering to [Answer Requirements].

### Answer Requirements:
1) Take time to think slowly, understand step by step, and answer questions.
2) Clearly state key information in the answer and provide direct and specific answers to user questions.

### {{ Background Knowledge }}
The retrieved knowledge chains are:
{knowledge_chains}

### {{ User Query }}
{query}
"""


def format_chains_for_prompt(chains: List[Dict]) -> str:
    """将链条格式化为 prompt 中的文本"""
    lines = []
    for i, c in enumerate(chains, 1):
        chain_str = c.get("chain", "")
        if chain_str:
            lines.append(f"{i}. {chain_str}")
    return "\n".join(lines) if lines else "（暂无相关链条）"


def get_answer(query: str, chains: List[Dict]) -> str:
    """
    调用 DeepSeek 生成答案
    :param query: 用户问题
    :param chains: 重排序后的 reasoning chains
    """
    knowledge_text = format_chains_for_prompt(chains)
    prompt = PROMPT_READER.format(
        knowledge_chains=knowledge_text,
        query=query
    )
    try:
        resp = requests.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": LLM_MAX_TOKENS,
                "temperature": 0.6
            },
            timeout=120
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    except Exception as e:
        return f"生成答案时出错: {e}"
    return "生成答案失败"
