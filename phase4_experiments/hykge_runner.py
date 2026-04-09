# -*- coding: utf-8 -*-
"""
HyKGE 实验运行器：复用 phase2 逻辑，支持多 LLM
"""

import sys
import os
import time
from typing import Dict, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase2_rag_system"))

from config import KG_HOP_K, RERANK_TOP_K
from ner_module import load_ner_model, extract_entities
from entity_linking import EntityLinker
from reranker import chunk_text, rerank_chains
from kg_client import DiseaseKGClient

from llm_clients import get_hypothesis_output, get_answer


class HyKGERunner:
    """HyKGE 完整流程，支持切换 LLM"""

    def __init__(self, model_name: str = "deepseek"):
        self.model_name = model_name
        self.kg_client = DiseaseKGClient()
        self.entity_linker = EntityLinker(self.kg_client)
        self.tokenizer, self.model, self.device, self.id2label = load_ner_model()

    def close(self):
        self.kg_client.close()

    def query(self, question: str) -> dict:
        out, _ = self.query_with_timings(question)
        return out

    def query_with_timings(self, question: str) -> Tuple[dict, Dict[str, float]]:
        """
        与 query 相同，额外返回各步骤耗时（秒，perf_counter）。
        键：hypothesis_output, ner, entity_linking, kg_retrieval, rerank, llm_reader, total
        """
        timings: Dict[str, float] = {}
        t_all = time.perf_counter()

        t0 = time.perf_counter()
        ho = get_hypothesis_output(self.model_name, question)
        if not ho:
            ho = question
        timings["hypothesis_output"] = time.perf_counter() - t0

        combined = question + " " + ho

        t0 = time.perf_counter()
        ner_entities = extract_entities(
            combined, self.tokenizer, self.model, self.device, self.id2label
        )
        ner_entities = [e for e in ner_entities if e.get("entity")]
        timings["ner"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        anchors = self.entity_linker.link_entities(ner_entities)
        timings["entity_linking"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        chains = self.kg_client.search_reasoning_chains(
            anchor_names=anchors,
            k_hops=KG_HOP_K,
            max_chains=100,
        )
        timings["kg_retrieval"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        fragments = chunk_text(combined)
        pruned_chains = rerank_chains(chains, fragments, top_k=RERANK_TOP_K)
        timings["rerank"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        answer = get_answer(
            self.model_name, question, pruned_chains, include_entity_desc=True
        )
        timings["llm_reader"] = time.perf_counter() - t0

        timings["total"] = time.perf_counter() - t_all

        result = {
            "answer": answer,
            "ho": ho,
            "entities": [{"entity": e["entity"], "type": e["type"]} for e in ner_entities],
            "anchors": anchors,
            "chains": chains,
            "pruned_chains": pruned_chains,
        }
        return result, timings
