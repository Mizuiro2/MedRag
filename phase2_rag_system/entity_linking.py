# -*- coding: utf-8 -*-
"""
Entity Linking - 论文 Section 4.2.1
使用 GTE 编码器将 NER 实体链接到 KG 实体
sim(ui, ej) = <enc(ui), enc(ej)>, 取最高相似度且 > δ 的匹配
"""

import os
import json
import time
import numpy as np
from typing import List, Dict, Optional

from config import GTE_MODEL_NAME, SIMILARITY_THRESHOLD, ENTITY_EMBED_CACHE, ENTITY_MAP_CACHE


def _load_gte_sentence_transformer(model_name: str):
    """
    加载 thenlper/gte-large-zh，尽量走本地缓存，减少访问 huggingface.co 时的 SSL/EOF 问题。
    联网失败时会多次重试；仍失败请设置镜像，例如：
      set HF_ENDPOINT=https://hf-mirror.com
    （PowerShell: $env:HF_ENDPOINT=\"https://hf-mirror.com\"）
    """
    from sentence_transformers import SentenceTransformer

    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")

    # 1) 仅本地：不发起 HEAD/GET，可避免 SSL: UNEXPECTED_EOF 与 client has been closed
    try:
        st = SentenceTransformer(
            model_name,
            local_files_only=True,
            trust_remote_code=True,
        )
        print("[实体链接] 已从本机 HuggingFace 缓存加载 GTE（未访问 huggingface.co）")
        return st
    except Exception as e:
        print(f"[实体链接] 无法仅用本地缓存加载（首次下载或缓存不完整）: {str(e)[:240]}")

    # 2) 联网拉取 + 重试（瞬时 SSL/网络错误常见）
    last_err: Optional[BaseException] = None
    for attempt in range(1, 6):
        try:
            st = SentenceTransformer(model_name, trust_remote_code=True)
            if attempt > 1:
                print(f"[实体链接] GTE 在线加载第 {attempt} 次尝试成功")
            return st
        except Exception as e:
            last_err = e
            msg = str(e)
            print(f"[实体链接] GTE 在线加载失败（{attempt}/5）: {msg[:320]}")
            if attempt < 5:
                wait = min(2 ** (attempt - 1), 45)
                print(f"[实体链接] {wait}s 后重试…")
                time.sleep(wait)
    assert last_err is not None
    print(
        "[实体链接] 若持续出现 SSL/EOF：可设置镜像后再运行，例如 "
        "HF_ENDPOINT=https://hf-mirror.com（中国大陆）或检查代理/VPN。"
    )
    raise last_err


class EntityLinker:
    def __init__(self, kg_client):
        self.kg_client = kg_client
        self.encoder = None
        self.entity_embeddings = None  # shape: (N, dim)
        self.entity_map = []           # [{name, label, desc}, ...]
        # 当前使用的编码器标识：gte | bge-flagfallback
        self.encoder_backend: str = ""
        self._load_encoder()
        self._load_or_build_entity_embeddings()

    def _load_encoder(self):
        """
        加载 GTE 编码器（论文实体链接；中文实验用 thenlper/gte-large-zh）。
        优先读本地 HF 缓存；需联网时带重试，减轻 SSL UNEXPECTED_EOF / client closed。
        """
        try:
            print(f"[实体链接] 正在加载 GTE：{GTE_MODEL_NAME}（SentenceTransformer）…")
            self.encoder = _load_gte_sentence_transformer(GTE_MODEL_NAME)
            self.encoder_backend = "gte"
            print(
                f"[实体链接] 已就绪：GTE（{GTE_MODEL_NAME}），用于 sim(ui,ej)=<enc(ui),enc(ej)>"
            )
        except Exception as e:
            print(f"[实体链接] GTE（SentenceTransformer）加载失败: {e}")
            print("[实体链接] 回退到 FlagEmbedding：BAAI/bge-large-zh-v1.5（非论文默认 GTE，仅作备用）")
            try:
                from FlagEmbedding import FlagModel

                self.encoder = FlagModel(
                    "BAAI/bge-large-zh-v1.5",
                    query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                )
                self.encoder_backend = "bge-flagfallback"
            except Exception as e2:
                raise RuntimeError(f"无法加载编码器: {e}, {e2}") from e2

    def _encode(self, texts: List[str]) -> np.ndarray:
        """编码文本为向量"""
        if hasattr(self.encoder, "encode"):
            return self.encoder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return self.encoder.encode_queries(texts)

    def _load_or_build_entity_embeddings(self):
        """加载或构建 KG 实体嵌入"""
        if os.path.exists(ENTITY_EMBED_CACHE) and os.path.exists(ENTITY_MAP_CACHE):
            try:
                self.entity_embeddings = np.load(ENTITY_EMBED_CACHE)
                with open(ENTITY_MAP_CACHE, "r", encoding="utf-8") as f:
                    self.entity_map = json.load(f)
                print(
                    f"已加载 {len(self.entity_map)} 个实体的嵌入缓存 "
                    f"（向量由 {self.encoder_backend or '?'} 生成；若曾换过编码器请删 cache 重建）"
                )
                return
            except Exception as e:
                print(f"加载缓存失败: {e}，重新构建")
        self._build_entity_embeddings()

    def _build_entity_embeddings(self):
        """从 Neo4j 构建实体嵌入"""
        entities = self.kg_client.get_all_entities_with_labels()
        if not entities:
            raise RuntimeError("KG 中无实体，请先运行 phase1 导入")
        texts = []
        for e in entities:
            t = e["name"]
            if e.get("desc"):
                t += " " + e["desc"][:100]
            texts.append(t)
        self.entity_embeddings = self._encode(texts).astype(np.float32)
        self.entity_map = entities
        os.makedirs(os.path.dirname(ENTITY_EMBED_CACHE), exist_ok=True)
        np.save(ENTITY_EMBED_CACHE, self.entity_embeddings)
        with open(ENTITY_MAP_CACHE, "w", encoding="utf-8") as f:
            json.dump(self.entity_map, f, ensure_ascii=False, indent=2)
        print(f"已构建 {len(self.entity_map)} 个实体的嵌入（编码器: {self.encoder_backend}）")

    def link_entities(self, ner_entities: List[Dict], delta: float = None) -> List[str]:
        """
        将 NER 实体链接到 KG，返回匹配的 KG 实体名称列表（锚点）
        :param ner_entities: [{"entity": str, "type": str, "kg_label": str}, ...]
        :param delta: 相似度阈值，默认使用 config
        """
        delta = delta or SIMILARITY_THRESHOLD
        anchor_names = []
        seen = set()

        for ne in ner_entities:
            entity_text = ne.get("entity", "").strip()
            if not entity_text or entity_text in seen:
                continue
            # 精确匹配优先
            if self.kg_client.find_entity_by_name(entity_text):
                anchor_names.append(entity_text)
                seen.add(entity_text)
                continue
            # 向量相似度匹配
            query_emb = self._encode([entity_text])[0]
            sims = np.dot(self.entity_embeddings, query_emb) / (
                np.linalg.norm(self.entity_embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-9
            )
            best_idx = np.argmax(sims)
            if sims[best_idx] > delta:
                matched_name = self.entity_map[best_idx]["name"]
                if matched_name not in seen:
                    anchor_names.append(matched_name)
                    seen.add(matched_name)
        return anchor_names
