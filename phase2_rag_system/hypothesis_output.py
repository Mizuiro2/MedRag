# -*- coding: utf-8 -*-
"""
Hypothesis Output Module (HOM) - 论文 Section 4.1.1
利用 LLM 的零样本推理能力生成假设输出，扩展 KG 检索方向
"""

import requests
from typing import Optional

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


# 论文 Figure 3 的 Prompt 格式
PROMPT_HO = """### Task Description: You are a medical expert. Please write a passage to answer [User Query] while adhering to [Answer Requirements].

### Answer Requirements:
1) Please take time to think slowly, understand step by step, and answer questions. Do not skip key steps.
2) Fully analyze the problem through thinking and exploratory analysis.

### {{ User Query }}
{query}
"""


def get_hypothesis_output(query: str) -> Optional[str]:
    """
    调用 DeepSeek API 生成假设输出
    :param query: 用户问题
    :return: 假设输出文本，失败返回 None
    """
    prompt = PROMPT_HO.format(query=query)
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
                "max_tokens": 500,
                "temperature": 0.6
            },
            timeout=60
        )
        if resp.status_code == 200:
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content.strip() if content else None
    except Exception as e:
        print(f"Hypothesis Output 调用失败: {e}")
    return None
