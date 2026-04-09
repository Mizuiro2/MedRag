# -*- coding: utf-8 -*-
"""
HO Fragment Granularity-aware Rerank Module - 论文 Section 4.3
将 Q ⊕ HO 分块，用 reranker 对 reasoning chains 重排序，保留 topK
"""

import re
from typing import List, Dict

from config import CHUNK_WINDOW, CHUNK_OVERLAP, RERANK_TOP_K


def chunk_text(text: str, window: int = None, overlap: int = None) -> List[str]:
    """
    分块：论文 Eq.5, Chunk(Q ⊕ HO)
    :param text: 输入文本
    :param window: 窗口大小 lc
    :param overlap: 重叠大小 oc
    """
    window = window or CHUNK_WINDOW
    overlap = overlap or CHUNK_OVERLAP
    text = _remove_stopwords(text)
    if not text or len(text) <= window:
        return [text] if text else []
    chunks = []
    start = 0
    while start < len(text):
        end = start + window
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


def _remove_stopwords(text: str) -> str:
    """去除停用词（论文提到使用 jieba 和 chinese_word_cut）"""
    # 简单过滤：去除多余空白和常见语气词
    stop = {"的", "了", "是", "在", "和", "与", "或", "及", "等", "之", "这", "那"}
    words = []
    for c in text:
        if c.strip() and c not in stop:
            words.append(c)
    return "".join(words)


def rerank_chains(
    chains: List[Dict],
    fragments: List[str],
    top_k: int = None,
    reranker_model=None
) -> List[Dict]:
    """
    对 reasoning chains 重排序
    Rerank(RC, {C}; topK) - 论文 Eq.6
    :param chains: 检索到的链条列表
    :param fragments: 分块后的 Q ⊕ HO
    :param top_k: 保留数量
    :param reranker_model: 重排序模型，若为 None 则用简单规则
    """
    top_k = top_k or RERANK_TOP_K
    if not chains:
        return []
    if not fragments:
        return chains[:top_k]

    # 若有 BGE reranker，使用它
    if reranker_model is not None:
        try:
            return _rerank_with_model(chains, fragments, top_k, reranker_model)
        except Exception as e:
            print(f"Reranker 模型失败: {e}，使用规则排序")

    # 规则排序：fragment 与 chain 的重叠度
    def score(chain: Dict) -> float:
        chain_str = chain.get("chain", "")
        s = 0.0
        for frag in fragments:
            if frag and any(w in chain_str for w in frag if len(w) >= 2):
                s += 1.0
        return s

    scored = [(c, score(c)) for c in chains]
    scored.sort(key=lambda x: -x[1])
    return [c for c, _ in scored[:top_k]]


def _rerank_with_model(chains, fragments, top_k, model) -> List[Dict]:
    """使用 BGE reranker 重排序"""
    try:
        from FlagEmbedding import FlagReranker
        reranker = FlagReranker("BAAI/bge-reranker-large", use_fp16=False)
    except ImportError:
        try:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder("BAAI/bge-reranker-large")
        except Exception:
            return chains[:top_k]

    # 每个 (fragment, chain) 对打分，chain 取各 fragment 下的最高分
    chain_scores = {i: 0.0 for i in range(len(chains))}
    for frag in fragments:
        if not frag:
            continue
        pairs = [[frag, c.get("chain", "")] for c in chains]
        try:
            scores = reranker.predict(pairs) if hasattr(reranker, "predict") else reranker.compute_score(pairs)
            if isinstance(scores, (int, float)):
                scores = [scores] * len(pairs)
            for i, s in enumerate(scores):
                chain_scores[i] = max(chain_scores[i], float(s))
        except Exception:
            pass

    sorted_indices = sorted(chain_scores.keys(), key=lambda i: -chain_scores[i])
    return [chains[i] for i in sorted_indices[:top_k]]
