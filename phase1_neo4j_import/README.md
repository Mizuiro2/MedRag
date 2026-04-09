# Phase 1: Neo4j 疾病知识图谱导入

本阶段实现从 `disease-kb` 数据到 Neo4j 的知识图谱导入。

## 数据来源

- 数据文件：`../disease-kb/data/medical.json`
- 参考项目：disease-kb（常见疾病相关信息构建 knowledge graph）

## 实体与关系

### 8 类实体

| 实体类型   | 中文含义 | 举例           |
| ---------- | -------- | -------------- |
| Disease    | 疾病     | 急性肺脓肿     |
| Drug       | 药品     | 布林佐胺滴眼液 |
| Food       | 食物     | 芝麻           |
| Check      | 检查项目 | 胸部CT检查     |
| Department | 科室     | 内科           |
| Producer   | 在售药品 | 青阳醋酸地塞米松片 |
| Symptom    | 疾病症状 | 乏力           |
| Cure       | 治疗方法 | 抗生素药物治疗 |

### 11 类关系

| 关系类型       | 中文含义     |
| -------------- | ------------ |
| belongs_to     | 属于（科室） |
| common_drug    | 疾病常用药品 |
| do_eat         | 疾病宜吃食物 |
| drugs_of       | 药品在售药品 |
| need_check     | 疾病所需检查 |
| no_eat         | 疾病忌吃食物 |
| recommand_drug| 疾病推荐药品 |
| recommand_eat  | 疾病推荐食谱 |
| has_symptom    | 疾病症状     |
| acompany_with  | 疾病并发疾病 |
| cure_way       | 疾病治疗方法 |

## 环境要求

- Python 3.7+
- Neo4j 4.x 或 5.x（需已启动）
- conda 环境：`rag_fyp`

## 配置

编辑 `config.py` 修改连接信息：

```python
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "qingquan666"
```

## 安装依赖

```bash
G:\Anaconda\envs\rag_fyp\python.exe -m pip install -r requirements.txt
```

## 运行导入

```bash
# 在 phase1_neo4j_import 目录下执行
G:\Anaconda\envs\rag_fyp\python.exe build_medicalgraph.py
```

**首次导入或需要重建时，先清空数据库：**

```bash
G:\Anaconda\envs\rag_fyp\python.exe build_medicalgraph.py --clear
```

**指定数据文件路径：**

```bash
G:\Anaconda\envs\rag_fyp\python.exe build_medicalgraph.py --data "H:\MedRAG\disease-kb\data\medical.json"
```

## 注意事项

1. 确保 Neo4j 已启动，且端口 7687 可访问
2. 使用 `--clear` 会删除图中所有节点和关系，请谨慎使用
3. 导入约 4.4 万实体、31 万关系，可能需要数分钟
