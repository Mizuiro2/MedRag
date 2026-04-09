# -*- coding: utf-8 -*-
"""
疾病知识图谱 Neo4j 导入脚本
基于 disease-kb 项目，适配新版依赖和 Neo4j 连接方式
数据来源：disease-kb/data/medical.json
"""

import os
import json
from py2neo import Graph, Node

# 导入配置
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, DATA_PATH


class MedicalGraph:
    """医疗知识图谱构建类"""

    def __init__(self, data_path=None):
        """
        初始化
        :param data_path: medical.json 路径，默认使用 config 中的路径
        """
        self.data_path = data_path or DATA_PATH
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")

        # 使用 neo4j:// 协议连接（适配 Neo4j 4.x/5.x）
        self.g = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def read_nodes(self):
        """从 medical.json 读取节点和关系"""
        drugs = []
        foods = []
        checks = []
        departments = []
        producers = []
        diseases = []
        symptoms = []
        cures = []

        disease_infos = []

        rels_department = []
        rels_noteat = []
        rels_doeat = []
        rels_recommandeat = []
        rels_commonddrug = []
        rels_recommanddrug = []
        rels_check = []
        rels_drug_producer = []
        rels_cureway = []

        rels_symptom = []
        rels_acompany = []
        rels_category = []

        count = 0
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                count += 1
                try:
                    data_json = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"第 {count} 行 JSON 解析失败: {e}")
                    continue

                disease = data_json.get("name", "")
                if not disease:
                    continue

                disease_dict = {
                    "name": disease,
                    "desc": "",
                    "prevent": "",
                    "cause": "",
                    "easy_get": "",
                    "cure_lasttime": "",
                    "cured_prob": "",
                }
                diseases.append(disease)

                if "symptom" in data_json:
                    symptoms.extend(data_json["symptom"])
                    for symptom in data_json["symptom"]:
                        rels_symptom.append([disease, symptom])

                if "acompany" in data_json:
                    for acompany in data_json["acompany"]:
                        rels_acompany.append([disease, acompany])

                if "desc" in data_json:
                    disease_dict["desc"] = data_json["desc"]
                if "prevent" in data_json:
                    disease_dict["prevent"] = data_json["prevent"]
                if "cause" in data_json:
                    disease_dict["cause"] = data_json["cause"]
                if "easy_get" in data_json:
                    disease_dict["easy_get"] = data_json["easy_get"]
                if "cure_department" in data_json:
                    cure_department = data_json["cure_department"]
                    if len(cure_department) == 1:
                        rels_category.append([disease, cure_department[0]])
                    if len(cure_department) == 2:
                        big, small = cure_department[0], cure_department[1]
                        rels_department.append([small, big])
                        rels_category.append([disease, small])
                    departments.extend(cure_department)

                if "cure_way" in data_json:
                    cure_way = data_json["cure_way"]
                    cures.extend(cure_way)
                    for cure in cure_way:
                        rels_cureway.append([disease, cure])

                if "cure_lasttime" in data_json:
                    disease_dict["cure_lasttime"] = data_json["cure_lasttime"]
                if "cured_prob" in data_json:
                    disease_dict["cured_prob"] = data_json["cured_prob"]

                if "common_drug" in data_json:
                    for drug in data_json["common_drug"]:
                        rels_commonddrug.append([disease, drug])
                    drugs.extend(data_json["common_drug"])

                if "recommand_drug" in data_json:
                    for drug in data_json["recommand_drug"]:
                        rels_recommanddrug.append([disease, drug])
                    drugs.extend(data_json["recommand_drug"])

                if "not_eat" in data_json:
                    for _not in data_json["not_eat"]:
                        rels_noteat.append([disease, _not])
                    foods.extend(data_json["not_eat"])
                    for _do in data_json.get("do_eat", []):
                        rels_doeat.append([disease, _do])
                    foods.extend(data_json.get("do_eat", []))
                    for _recommand in data_json.get("recommand_eat", []):
                        rels_recommandeat.append([disease, _recommand])
                    foods.extend(data_json.get("recommand_eat", []))

                if "check" in data_json:
                    for _check in data_json["check"]:
                        rels_check.append([disease, _check])
                    checks.extend(data_json["check"])

                if "drug_detail" in data_json:
                    for i in data_json["drug_detail"]:
                        parts = i.split("(")
                        if len(parts) >= 2:
                            producer = parts[0]
                            drug = parts[-1].replace(")", "").strip()
                            if producer and drug:
                                rels_drug_producer.append([producer, drug])
                                producers.append(producer)

                disease_infos.append(disease_dict)

        return (
            set(drugs),
            set(foods),
            set(checks),
            set(departments),
            set(producers),
            set(symptoms),
            set(diseases),
            set(cures),
            disease_infos,
            rels_check,
            rels_recommandeat,
            rels_noteat,
            rels_doeat,
            rels_department,
            rels_commonddrug,
            rels_drug_producer,
            rels_recommanddrug,
            rels_symptom,
            rels_acompany,
            rels_category,
            rels_cureway,
        )

    def create_node(self, label, nodes):
        """创建普通实体节点"""
        nodes = list(nodes)
        total = len(nodes)
        for i, node_name in enumerate(nodes):
            if not node_name or not isinstance(node_name, str):
                continue
            node = Node(label, name=node_name.strip())
            self.g.create(node)
            if (i + 1) % 500 == 0 or i + 1 == total:
                print(f"  {label}: {i + 1}/{total}")

    def create_diseases_nodes(self, disease_infos):
        """创建疾病节点（含属性）"""
        total = len(disease_infos)
        for i, d in enumerate(disease_infos):
            node = Node(
                "Disease",
                name=self._safe_str(d.get("name", "")),
                desc=self._safe_str(d.get("desc", "")),
                prevent=self._safe_str(d.get("prevent", "")),
                cause=self._safe_str(d.get("cause", "")),
                easy_get=self._safe_str(d.get("easy_get", "")),
                cure_lasttime=self._safe_str(d.get("cure_lasttime", "")),
                cured_prob=self._safe_str(d.get("cured_prob", "")),
            )
            self.g.create(node)
            if (i + 1) % 500 == 0 or i + 1 == total:
                print(f"  Disease: {i + 1}/{total}")

    @staticmethod
    def _safe_str(s):
        """确保字符串非 None，且截断过长文本（Neo4j 属性建议不超过一定长度）"""
        if s is None:
            return ""
        s = str(s).strip()
        # 超长描述可截断，避免性能问题（可选）
        if len(s) > 32000:
            s = s[:32000] + "..."
        return s

    def create_graphnodes(self):
        """创建所有节点"""
        print("正在读取数据...")
        (
            drugs,
            foods,
            checks,
            departments,
            producers,
            symptoms,
            diseases,
            cures,
            disease_infos,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = self.read_nodes()

        print("创建 Disease 节点...")
        self.create_diseases_nodes(disease_infos)
        print("创建其他实体节点...")
        self.create_node("Drug", drugs)
        self.create_node("Food", foods)
        self.create_node("Check", checks)
        self.create_node("Department", departments)
        self.create_node("Producer", producers)
        self.create_node("Symptom", symptoms)
        self.create_node("Cure", cures)
        print("节点创建完成。")

    def create_relationship(self, start_node, end_node, edges, rel_type, rel_name):
        """创建关系（使用参数化查询，避免注入和特殊字符问题）"""
        set_edges = []
        for edge in edges:
            if len(edge) >= 2 and edge[0] and edge[1]:
                set_edges.append("###".join([str(edge[0]), str(edge[1])]))
        unique_edges = list(set(set_edges))
        total = len(unique_edges)
        count = 0

        # 使用参数化 Cypher，避免 name 中的单引号等问题
        query = (
            f"MATCH (p:{start_node}), (q:{end_node}) "
            "WHERE p.name = $p_name AND q.name = $q_name "
            f"CREATE (p)-[r:{rel_type} {{name: $rel_name}}]->(q)"
        )

        for edge_str in unique_edges:
            parts = edge_str.split("###")
            if len(parts) != 2:
                continue
            p_name, q_name = parts[0], parts[1]
            try:
                self.g.run(
                    query,
                    parameters={
                        "p_name": p_name,
                        "q_name": q_name,
                        "rel_name": rel_name,
                    },
                )
                count += 1
                if count % 1000 == 0 or count == total:
                    print(f"  {rel_type}: {count}/{total}")
            except Exception as e:
                print(f"  关系创建失败 [{p_name}]-[{rel_type}]->[{q_name}]: {e}")

    def create_graphrels(self):
        """创建所有关系"""
        print("正在读取关系数据...")
        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            rels_check,
            rels_recommandeat,
            rels_noteat,
            rels_doeat,
            rels_department,
            rels_commonddrug,
            rels_drug_producer,
            rels_recommanddrug,
            rels_symptom,
            rels_acompany,
            rels_category,
            rels_cureway,
        ) = self.read_nodes()

        print("创建关系...")
        self.create_relationship("Disease", "Food", rels_recommandeat, "recommand_eat", "推荐食谱")
        self.create_relationship("Disease", "Food", rels_noteat, "no_eat", "忌吃")
        self.create_relationship("Disease", "Food", rels_doeat, "do_eat", "宜吃")
        self.create_relationship("Department", "Department", rels_department, "belongs_to", "属于")
        self.create_relationship("Disease", "Drug", rels_commonddrug, "common_drug", "常用药品")
        self.create_relationship("Producer", "Drug", rels_drug_producer, "drugs_of", "生产药品")
        self.create_relationship("Disease", "Drug", rels_recommanddrug, "recommand_drug", "好评药品")
        self.create_relationship("Disease", "Check", rels_check, "need_check", "诊断检查")
        self.create_relationship("Disease", "Symptom", rels_symptom, "has_symptom", "症状")
        self.create_relationship("Disease", "Disease", rels_acompany, "acompany_with", "并发症")
        self.create_relationship("Disease", "Department", rels_category, "belongs_to", "所属科室")
        self.create_relationship("Disease", "Cure", rels_cureway, "cure_way", "治疗方法")
        print("关系创建完成。")

    def clear_graph(self):
        """清空图数据库（慎用！）"""
        self.g.run("MATCH (n) DETACH DELETE n")
        print("图数据库已清空。")

    def run_import(self, clear_first=False):
        """
        执行完整导入
        :param clear_first: 是否先清空数据库
        """
        if clear_first:
            print("清空现有数据...")
            self.clear_graph()
        print("开始导入疾病知识图谱...")
        self.create_graphnodes()
        self.create_graphrels()
        print("导入完成！")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="疾病知识图谱 Neo4j 导入")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="导入前清空数据库（慎用）",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="medical.json 路径（默认使用 config）",
    )
    args = parser.parse_args()

    handler = MedicalGraph(data_path=args.data)
    handler.run_import(clear_first=args.clear)
