# -*- coding: utf-8 -*-
"""
HyKGE 主流程 - 论文 Algorithm 1
完整 RAG 流程：HO -> NER -> Entity Linking -> KG Retrieval -> Rerank -> LLM Reader
"""

from typing import Dict, List, Optional

from config import KG_HOP_K, RERANK_TOP_K
from hypothesis_output import get_hypothesis_output
from ner_module import load_ner_model, extract_entities
from entity_linking import EntityLinker
from reranker import chunk_text, rerank_chains
from llm_reader import get_answer
from kg_client import DiseaseKGClient


class HyKGE:
    """HyKGE RAG 系统"""

    def __init__(self):
        self.kg_client = DiseaseKGClient()
        self.entity_linker = EntityLinker(self.kg_client)
        self.tokenizer, self.model, self.device, self.id2label = load_ner_model()

    def close(self):
        self.kg_client.close()

    def query(self, question: str) -> Dict:
        """
        完整 HyKGE 流程
        :return: {"answer": str, "ho": str, "entities": list, "anchors": list, "chains": list, "pruned_chains": list}
        """
        # 1. Hypothesis Output
        ho = get_hypothesis_output(question)
        if not ho:
            ho = question  # 若 HO 失败，仅用原问题

        # 2. NER: 从 Q ⊕ HO 提取实体
        combined = question + " " + ho
        ner_entities = extract_entities(
            combined, self.tokenizer, self.model, self.device, self.id2label
        )
        ner_entities = [e for e in ner_entities if e.get("entity")]

        # 3. Entity Linking: 实体链接到 KG 锚点
        anchors = self.entity_linker.link_entities(ner_entities)

        # 4. KG Retrieval: 检索 reasoning chains
        chains = self.kg_client.search_reasoning_chains(
            anchor_names=anchors,
            k_hops=KG_HOP_K,
            max_chains=100
        )

        # 5. HO Fragment Rerank
        fragments = chunk_text(combined)
        pruned_chains = rerank_chains(chains, fragments, top_k=RERANK_TOP_K)

        # 6. LLM Reader
        answer = get_answer(question, pruned_chains)

        return {
            "answer": answer,
            "ho": ho,
            "entities": [{"entity": e["entity"], "type": e["type"]} for e in ner_entities],
            "anchors": anchors,
            "chains": chains,
            "pruned_chains": pruned_chains
        }


def main():
    """命令行测试"""
    import sys
    h = HyKGE()
    q = sys.argv[1] if len(sys.argv) > 1 else "糖尿病有什么症状？"
    print("问题:", q)
    result = h.query(q)
    print("\n答案:", result["answer"])
    print("\n锚点实体:", result["anchors"])
    print("保留链条数:", len(result["pruned_chains"]))
    h.close()


if __name__ == "__main__":
    main()
