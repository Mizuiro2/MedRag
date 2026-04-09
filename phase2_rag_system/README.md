# Phase 2: HyKGE RAG 问答系统

按照论文 **HyKGE: A Hypothesis Knowledge Graph Enhanced RAG Framework for Accurate and Reliable Medical LLMs Responses** (ACL 2025) 实现。

## 流程概览

1. **Hypothesis Output Module (HOM)**：DeepSeek 生成假设输出，扩展检索方向
2. **NER Module**：W2NER 从 Q ⊕ HO 提取医学实体
3. **Entity Linking**：GTE 编码器将实体链接到 disease-kb
4. **KG Retrieval**：检索 reasoning chains（邻居关系 + 锚点对路径）
5. **HO Fragment Rerank**：分块 + 重排序，保留 topK 链条
6. **LLM Reader**：基于检索知识生成最终答案

## 依赖

- Phase 1 已导入 disease-kb 到 Neo4j
- W2NER 模型：`G:\W2NER\output`
- DeepSeek API Key（已配置）

## 安装

```bash
G:\Anaconda\envs\rag_fyp\python.exe -m pip install -r requirements.txt
```

首次运行会下载 GTE 模型并构建实体嵌入缓存（需数分钟）。

## 运行

```bash
cd h:\MedRAG\phase2_rag_system
G:\Anaconda\envs\rag_fyp\python.exe hykge.py "糖尿病有什么症状？"
```

## 配置

编辑 `config.py` 修改：
- Neo4j 连接
- DeepSeek API
- 超参数：δ=0.7, k=3, topK=10, lc=10, oc=4

## 实体类型映射

| CMeEE (W2NER) | disease-kb |
|---------------|------------|
| dis           | Disease    |
| dru           | Drug       |
| sym           | Symptom    |
| dep           | Department |
| equ, ite      | Check      |
| pro           | Cure       |
| bod, mic      | Disease    |
