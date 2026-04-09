# -*- coding: utf-8 -*-
"""
disease-kb 知识图谱客户端
适配 phase1 导入的 schema: Disease, Drug, Food, Check, Department, Producer, Symptom, Cure
节点属性: name (必需), Disease 另有 desc, prevent, cause, easy_get, cure_lasttime, cured_prob
"""

from typing import List, Dict, Optional, Tuple
from neo4j import GraphDatabase

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


class DiseaseKGClient:
    """disease-kb Neo4j 客户端"""

    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def get_all_entities_with_labels(self) -> List[Dict]:
        """
        获取所有实体（含标签和描述），用于预计算嵌入
        返回: [{"name": str, "label": str, "desc": str}, ...]
        """
        labels = ["Disease", "Drug", "Food", "Check", "Department", "Producer", "Symptom", "Cure"]
        entities = []
        with self.driver.session() as session:
            for label in labels:
                try:
                    if label == "Disease":
                        result = session.run(
                            f"MATCH (n:{label}) RETURN n.name as name, "
                            "COALESCE(n.desc, '') as desc"
                        )
                    else:
                        result = session.run(
                            f"MATCH (n:{label}) RETURN n.name as name"
                        )
                    for record in result:
                        entities.append({
                            "name": record["name"],
                            "label": label,
                            "desc": record.get("desc", "") or ""
                        })
                except Exception as e:
                    print(f"获取 {label} 实体时出错: {e}")
        return entities

    def get_entity_description(self, name: str, label: str) -> str:
        """获取实体描述（仅 Disease 有 desc）"""
        with self.driver.session() as session:
            result = session.run(
                f"MATCH (n:{label} {{name: $name}}) RETURN n",
                name=name
            )
            record = result.single()
            if record and record["n"]:
                node = record["n"]
                return node.get("desc", "") or ""
        return ""

    def search_reasoning_chains(
        self,
        anchor_names: List[str],
        k_hops: int = 3,
        max_chains: int = 100
    ) -> List[Dict]:
        """
        检索 reasoning chains（HyKGE 论文：path / chainCA / chainCO，在 k 跳内，k≥2 时常取 k=3）。

        **存储**：disease-kb 在 Neo4j 中为**有向边**（导入时 ``CREATE (p)-[r]->(q)``）。

        - **path**：两锚点间 ``-[*1..k]-`` **无向遍历**（沿关系正反均可走）。因图中多为
          ``Disease→Symptom`` 等单向边，两实体间合理解释链常需「经中间点双向衔接」，
          仅用 ``(a)-[*]->(b)`` 有向路径会大量缺失。
        - **chainCA（共汇）**：优先 **有向** ``(a)-[*]->(e)`` 且 ``(b)-[*]->(e)``；若无结果再回退无向汇合模式。
        - **chainCO（共源）**：优先 **有向** ``(e)-[*]->(a)`` 且 ``(e)-[*]->(b)``；若无结果再回退无向分叉模式。

        另保留单锚点 1-hop 邻居（``MATCH (a)-[r]->(b)``，尊重存储方向）。
        """
        if not anchor_names:
            return []

        chains = []
        seen = set()

        def _add(batch: List[Dict]) -> bool:
            nonlocal chains
            for c in batch:
                key = c["chain"]
                if key not in seen:
                    seen.add(key)
                    chains.append(c)
                    if len(chains) >= max_chains:
                        return True
            return False

        k = max(2, int(k_hops))

        with self.driver.session() as session:
            # 单锚点 1-hop（补充）
            for name in anchor_names:
                if _add(self._get_entity_neighbor_chains(session, name)):
                    return chains[:max_chains]

            # 锚点对：三种论文链模式
            for i, name_i in enumerate(anchor_names):
                for name_j in anchor_names[i + 1:]:
                    if name_i == name_j:
                        continue
                    if _add(self._get_pair_path_chains(session, name_i, name_j, k)):
                        return chains[:max_chains]
                    if _add(self._get_pair_chain_ca(session, name_i, name_j, k)):
                        return chains[:max_chains]
                    if _add(self._get_pair_chain_co(session, name_i, name_j, k)):
                        return chains[:max_chains]

        return chains[:max_chains]

    def _get_entity_neighbor_chains(self, session, entity_name: str) -> List[Dict]:
        """获取实体的邻居关系链条 (head)-[rel]->(tail)"""
        query = """
        MATCH (a)-[r]->(b)
        WHERE a.name = $name OR b.name = $name
        RETURN a.name as head, type(r) as rel, b.name as tail,
               a.desc as head_desc, b.desc as tail_desc
        LIMIT 20
        """
        try:
            result = session.run(query, name=entity_name)
            chains = []
            for record in result:
                head = record["head"]
                tail = record["tail"]
                rel = record["rel"]
                chain_str = f"{head} --[{rel}]--> {tail}"
                head_desc = record["head_desc"] or "" if record["head_desc"] is not None else ""
                tail_desc = record["tail_desc"] or "" if record["tail_desc"] is not None else ""
                chains.append({
                    "chain": chain_str,
                    "type": "neighbor",
                    "head": head,
                    "tail": tail,
                    "head_desc": head_desc,
                    "tail_desc": tail_desc
                })
            return chains
        except Exception:
            return []

    def _get_pair_path_chains(self, session, head: str, tail: str, k: int) -> List[Dict]:
        """path_ij：两锚点之间长度 1..k 的简单路径（无向遍历 ``-[*]-``，见 search_reasoning_chains 说明）。"""
        query = f"""
        MATCH path = (a)-[*1..{k}]-(b)
        WHERE a.name = $head AND b.name = $tail AND a <> b
        RETURN path
        LIMIT 10
        """
        try:
            result = session.run(query, head=head, tail=tail)
            chains = []
            for record in result:
                path = record["path"]
                chain_str = self._path_to_chain_str(path)
                if chain_str:
                    head_name = path.start_node["name"]
                    tail_name = path.end_node["name"]
                    head_desc = path.start_node.get("desc", "") or ""
                    tail_desc = path.end_node.get("desc", "") or ""
                    chains.append({
                        "chain": chain_str,
                        "type": "path",
                        "head": head_name,
                        "tail": tail_name,
                        "head_desc": head_desc,
                        "tail_desc": tail_desc,
                    })
            return chains
        except Exception:
            return []

    def _get_pair_chain_ca(
        self, session, name_i: str, name_j: str, k: int
    ) -> List[Dict]:
        """
        chainCA_ij：共汇于 e。优先有向 (a)-*->(e) 与 (b)-*->(e)；无结果时用无向 ``-[*]-`` 回退。
        """
        kh = max(1, k)
        q_dir = f"""
        MATCH (a {{name: $name_i}}), (b {{name: $name_j}})
        WHERE a <> b
        MATCH p1 = (a)-[*1..{kh}]->(e)
        MATCH p2 = (b)-[*1..{kh}]->(e)
        WHERE e <> a AND e <> b
        RETURN p1, p2, e
        LIMIT 8
        """
        q_undir = f"""
        MATCH (a {{name: $name_i}}), (b {{name: $name_j}})
        WHERE a <> b
        MATCH p1 = (a)-[*1..{kh}]-(e)
        MATCH p2 = (b)-[*1..{kh}]-(e)
        WHERE e <> a AND e <> b
        RETURN p1, p2, e
        LIMIT 8
        """
        try:
            r = session.run(q_dir, name_i=name_i, name_j=name_j)
            out = self._records_from_ca_paths(r, "chain_ca")
            if out:
                return out
            r2 = session.run(q_undir, name_i=name_i, name_j=name_j)
            return self._records_from_ca_paths(r2, "chain_ca")
        except Exception:
            return []

    def _get_pair_chain_co(
        self, session, name_i: str, name_j: str, k: int
    ) -> List[Dict]:
        """
        chainCO_ij：共源于 e。优先有向 (e)-*->(a) 与 (e)-*->(b)；无结果时用无向回退。
        """
        kh = max(1, k)
        q_dir = f"""
        MATCH (a {{name: $name_i}}), (b {{name: $name_j}})
        WHERE a <> b
        MATCH p1 = (e)-[*1..{kh}]->(a)
        MATCH p2 = (e)-[*1..{kh}]->(b)
        WHERE e <> a AND e <> b
        RETURN p1, p2, e
        LIMIT 8
        """
        q_undir = f"""
        MATCH (a {{name: $name_i}}), (b {{name: $name_j}})
        WHERE a <> b
        MATCH p1 = (e)-[*1..{kh}]-(a)
        MATCH p2 = (e)-[*1..{kh}]-(b)
        WHERE e <> a AND e <> b
        RETURN p1, p2, e
        LIMIT 8
        """
        try:
            r = session.run(q_dir, name_i=name_i, name_j=name_j)
            out = self._records_from_co_paths(r, "chain_co")
            if out:
                return out
            r2 = session.run(q_undir, name_i=name_i, name_j=name_j)
            return self._records_from_co_paths(r2, "chain_co")
        except Exception:
            return []

    def _records_from_ca_paths(self, result, chain_type: str) -> List[Dict]:
        chains: List[Dict] = []
        for record in result:
            p1 = record["p1"]
            p2 = record["p2"]
            e = record["e"]
            e_name = e.get("name", "")
            s1 = self._path_to_chain_str(p1)
            s2 = self._path_to_chain_str(p2)
            if not s1 or not s2:
                continue
            chain_str = f"[chainCA @ {e_name}] {s1}  ||  {s2}"
            head_name = p1.start_node.get("name", "")
            tail_name = p2.start_node.get("name", "")
            chains.append({
                "chain": chain_str,
                "type": chain_type,
                "head": head_name,
                "tail": tail_name,
                "head_desc": p1.start_node.get("desc", "") or "",
                "tail_desc": p2.start_node.get("desc", "") or "",
            })
        return chains

    def _records_from_co_paths(self, result, chain_type: str) -> List[Dict]:
        chains: List[Dict] = []
        for record in result:
            p1 = record["p1"]
            p2 = record["p2"]
            e = record["e"]
            e_name = e.get("name", "")
            s1 = self._path_to_chain_str(p1)
            s2 = self._path_to_chain_str(p2)
            if not s1 or not s2:
                continue
            chain_str = f"[chainCO @ {e_name}] {s1}  ||  {s2}"
            head_name = p1.end_node.get("name", "")
            tail_name = p2.end_node.get("name", "")
            chains.append({
                "chain": chain_str,
                "type": chain_type,
                "head": head_name,
                "tail": tail_name,
                "head_desc": p1.end_node.get("desc", "") or "",
                "tail_desc": p2.end_node.get("desc", "") or "",
            })
        return chains

    def _path_to_chain_str(self, path) -> str:
        """将 Neo4j Path 转为字符串: n0 -> r1 -> n1 -> r2 -> n2"""
        if not path:
            return ""
        nodes = list(path.nodes)
        rels = list(path.relationships)
        if len(nodes) < 2:
            return nodes[0]["name"] if nodes else ""
        parts = []
        for i, rel in enumerate(rels):
            parts.append(nodes[i]["name"])
            parts.append(rel.type)
        parts.append(nodes[-1]["name"])
        return " → ".join(parts)

    def find_entity_by_name(self, name: str, label: Optional[str] = None) -> Optional[Dict]:
        """按名称查找实体（精确匹配）"""
        if label:
            labels = [label]
        else:
            labels = ["Disease", "Drug", "Food", "Check", "Department", "Producer", "Symptom", "Cure"]
        with self.driver.session() as session:
            for l in labels:
                result = session.run(
                    f"MATCH (n:{l} {{name: $name}}) RETURN n.name as name, '{l}' as label",
                    name=name
                )
                record = result.single()
                if record:
                    return {"name": record["name"], "label": record["label"]}
        return None
