# -*- coding: utf-8 -*-
"""
HyKGE RAG 系统配置文件
按照论文 HyKGE (ACL 2025) 的结构设计
"""

import os

# ============ Neo4j (disease-kb) ============
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "qingquan666"

# ============ DeepSeek API (LLM) ============
DEEPSEEK_API_KEY = "sk-bfd7d198f966480e99e2106d4e464b32"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

# ============ W2NER 模型 ============
W2NER_MODEL_PATH = r"G:\W2NER\output"

# ============ GTE 编码器 (Entity Linking) ============
# 论文使用 gte_sentence-embedding，此处用中文版
GTE_MODEL_NAME = "thenlper/gte-large-zh"
GTE_EMBED_DIM = 1024

# ============ BGE Reranker ============
BGE_RERANKER_MODEL = "BAAI/bge-reranker-large"

# ============ 论文超参数 (Section 5.1.6) ============
SIMILARITY_THRESHOLD = 0.7   # δ: 实体链接相似度阈值
KG_HOP_K = 3                # k: 知识图谱检索跳数
RERANK_TOP_K = 10            # topK: 重排序保留的链条数
CHUNK_WINDOW = 10            # lc: 分块窗口大小（字符数）
CHUNK_OVERLAP = 4            # oc: 分块重叠大小

# LLM 回答长度
LLM_MAX_TOKENS = 2000

# ============ 路径 ============
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 实体向量索引缓存（预计算 KG 实体嵌入）
ENTITY_EMBED_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "entity_embeddings.npy")
ENTITY_MAP_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "entity_map.json")
