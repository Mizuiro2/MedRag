# -*- coding: utf-8 -*-
"""
纯 LLM 基线：无 KG、无 NER、无 HyKGE，仅把题目发给模型作答（Table 2 对比用）
"""

from llm_clients import call_llm


class BaselineRunner:
    """仅凭模型本身回答选择题或开放问答"""

    def __init__(self, model_name: str = "ollama", max_tokens: int = 512, temperature: float = 0.6):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    def close(self):
        pass

    def query(self, question: str) -> dict:
        """question 为已格式化的完整 prompt（选择题或病例问答）"""
        ans = call_llm(
            self.model_name, question, max_tokens=self.max_tokens, temperature=self.temperature
        )
        return {"answer": ans or "", "pruned_chains": []}
