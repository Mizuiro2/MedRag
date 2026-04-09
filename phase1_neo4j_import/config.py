# -*- coding: utf-8 -*-
"""
Neo4j 疾病知识图谱导入 - 配置文件
"""

import os

# Neo4j 连接配置
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "qingquan666"

# 数据路径：指向 disease-kb 的 medical.json
# 默认使用项目根目录下的 disease-kb
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "disease-kb", "data", "medical.json")
