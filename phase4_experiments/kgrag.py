# -*- coding: utf-8 -*-
"""
KGRAG 基线：仅用 query 在 KG 中检索，无 HO、无 rerank
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase2_rag_system"))

from config import KG_HOP_K, RERANK_TOP_K
from ner_module import load_ner_model, extract_entities
from entity_linking import EntityLinker
from kg_client import DiseaseKGClient

from llm_clients import get_answer


class KGRAG:
    """KGRAG 基线：Query -> NER -> Entity Linking -> KG Retrieval -> LLM（无 HO、无 rerank）"""

    def __init__(self, model_name: str = "deepseek"):
        self.model_name = model_name
        self.kg_client = DiseaseKGClient()
        self.entity_linker = EntityLinker(self.kg_client)
        self.tokenizer, self.model, self.device, self.id2label = load_ner_model()

    def close(self):
        self.kg_client.close()

    def query(self, question: str) -> dict:
        """
        KGRAG 流程：仅用 question，无 HO，无 rerank
        """
        # 1. NER: 仅从 question 提取实体
        ner_entities = extract_entities(
            question, self.tokenizer, self.model, self.device, self.id2label
        )
        ner_entities = [e for e in ner_entities if e.get("entity")]

        # 2. Entity Linking
        anchors = self.entity_linker.link_entities(ner_entities)

        # 3. KG Retrieval
        chains = self.kg_client.search_reasoning_chains(
            anchor_names=anchors,
            k_hops=KG_HOP_K,
            max_chains=100
        )

        # 4. 无 rerank，直接取 topK
        pruned_chains = chains[:RERANK_TOP_K]

        # 5. LLM Reader
        answer = get_answer(
            self.model_name, question, pruned_chains, include_entity_desc=True
        )

        return {
            "answer": answer,
            "entities": [{"entity": e["entity"], "type": e["type"]} for e in ner_entities],
            "anchors": anchors,
            "chains": chains,
            "pruned_chains": pruned_chains
        }
