# -*- coding: utf-8 -*-
"""
多 LLM 调用：DeepSeek, Qwen3-max, Doubao, Ollama（本地）
统一接口，支持切换模型
"""

import requests
from typing import List, Optional
from openai import OpenAI

# Ollama：由 run_experiments 在启动时 set_ollama_config
_ollama_model: str = "qwen3.5:9b"
_ollama_host: str = "http://127.0.0.1:11434"
_ollama_think: bool = False


def set_ollama_config(model: str, host: str = "http://127.0.0.1:11434", think: bool = False) -> None:
    """设置 Ollama 模型名、服务地址、是否开启 think（Qwen3 等推理模型）"""
    global _ollama_model, _ollama_host, _ollama_think
    _ollama_model = model
    _ollama_host = host.rstrip("/")
    _ollama_think = think

# API 配置（来自 Instruction.md）
DEEPSEEK_API_KEY = "sk-bfd7d198f966480e99e2106d4e464b32"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

QWEN_API_KEY = "sk-f16800ff4f58416c9691bf325b30eae3"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen-max"  # 百炼兼容接口

DOUBAO_API_KEY = "46d21dc9-f8e3-4e5f-9157-0db7b713011c"
DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DOUBAO_MODEL = "doubao-seed-2-0-pro-260215"

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

PROMPT_HO = """### Task Description: You are a medical expert. Please write a passage to answer [User Query] while adhering to [Answer Requirements].

### Answer Requirements:
1) Please take time to think slowly, understand step by step, and answer questions. Do not skip key steps.
2) Fully analyze the problem through thinking and exploratory analysis.

### {{ User Query }}
{query}
"""


def _format_chains(chains: list, include_entity_desc: bool = True) -> str:
    """将检索链格式化为 Reader 背景；可选附带 KG 中头/尾实体描述（论文完整版）。"""
    lines = []
    for i, c in enumerate(chains, 1):
        chain_str = c.get("chain", "") if isinstance(c, dict) else str(c)
        if not chain_str:
            continue
        line = f"{i}. {chain_str}"
        if include_entity_desc and isinstance(c, dict):
            hd = (c.get("head_desc") or "").strip()
            td = (c.get("tail_desc") or "").strip()
            if hd or td:
                bits = []
                if hd:
                    bits.append(f"头实体描述: {hd[:400]}")
                if td:
                    bits.append(f"尾实体描述: {td[:400]}")
                line += " | " + " ".join(bits)
        lines.append(line)
    return "\n".join(lines) if lines else "（暂无相关链条）"


def _call_openai_compatible(
    base_url: str, api_key: str, model: str,
    messages: list, max_tokens: int = 2000, temperature: float = 0.6
) -> Optional[str]:
    """OpenAI 兼容接口（DeepSeek、Qwen 均支持）"""
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=120
        )
        content = resp.choices[0].message.content
        return content.strip() if content else None
    except Exception as e:
        print(f"LLM 调用失败: {e}")
        return None


def _call_ollama_chat(
    messages: List[dict],
    max_tokens: int = 2000,
    temperature: float = 0.6,
) -> Optional[str]:
    """调用 Ollama /api/chat（非流式），支持 think 开关"""
    url = f"{_ollama_host}/api/chat"
    body = {
        "model": _ollama_model,
        "messages": messages,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": temperature},
    }
    if not _ollama_think:
        body["think"] = False
    try:
        resp = requests.post(url, json=body, timeout=600)
        if resp.status_code != 200:
            if "think" in body:
                body.pop("think", None)
                resp = requests.post(url, json=body, timeout=600)
            if resp.status_code != 200:
                print(f"Ollama 调用失败: HTTP {resp.status_code} {resp.text[:200]}")
                return None
        data = resp.json()
        msg = data.get("message") or {}
        content = msg.get("content")
        return content.strip() if content else None
    except Exception as e:
        print(f"Ollama 调用失败: {e}")
        return None


def _call_doubao_text(messages: list, max_tokens: int = 2000, temperature: float = 0.6) -> Optional[str]:
    """Doubao 纯文本对话接口"""
    try:
        client = OpenAI(api_key=DOUBAO_API_KEY, base_url=DOUBAO_BASE_URL)
        # 使用 chat.completions（纯文本）
        resp = client.chat.completions.create(
            model=DOUBAO_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=120
        )
        content = resp.choices[0].message.content
        return content.strip() if content else None
    except Exception as e:
        print(f"Doubao 调用失败: {e}")
        return None


def call_llm(model_name: str, prompt: str, max_tokens: int = 2000, temperature: float = 0.6) -> Optional[str]:
    """
    统一 LLM 调用入口
    :param model_name: "deepseek" | "qwen3-max" | "doubao" | "ollama"
    """
    messages = [{"role": "user", "content": prompt}]
    if model_name == "deepseek":
        return _call_openai_compatible(
            DEEPSEEK_BASE_URL, DEEPSEEK_API_KEY, DEEPSEEK_MODEL,
            messages, max_tokens, temperature
        )
    elif model_name == "qwen3-max":
        return _call_openai_compatible(
            QWEN_BASE_URL, QWEN_API_KEY, QWEN_MODEL,
            messages, max_tokens, temperature
        )
    elif model_name == "doubao":
        return _call_doubao_text(messages, max_tokens, temperature)
    elif model_name == "ollama":
        return _call_ollama_chat(messages, max_tokens, temperature)
    else:
        raise ValueError(f"未知模型: {model_name}")


def get_hypothesis_output(model_name: str, query: str) -> Optional[str]:
    """生成假设输出（仅 HyKGE 使用）"""
    prompt = PROMPT_HO.format(query=query)
    return call_llm(model_name, prompt, max_tokens=500)


def get_answer(
    model_name: str,
    query: str,
    chains: list,
    include_entity_desc: bool = True,
) -> str:
    """基于检索知识生成答案。include_entity_desc=False 对应论文消融 w/o Description。"""
    knowledge_text = _format_chains(chains, include_entity_desc=include_entity_desc)
    prompt = PROMPT_READER.format(knowledge_chains=knowledge_text, query=query)
    result = call_llm(model_name, prompt, max_tokens=2000)
    return result if result else "生成答案失败"
