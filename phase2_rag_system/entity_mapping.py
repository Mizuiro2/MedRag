# -*- coding: utf-8 -*-
"""
CMeEE-V2 / W2NER 实体类型 -> disease-kb 节点类型映射
用于 Entity Linking 时限定检索的节点类型
"""

# CMeEE-V2 标签 -> disease-kb 节点标签
# W2NER 在 CMeEE 上训练，标签: pro, dis, dru, bod, sym, mic, equ, ite, dep
CMEEE_TO_KG_LABEL = {
    "dis": "Disease",      # 疾病
    "dru": "Drug",        # 药物
    "sym": "Symptom",     # 症状
    "dep": "Department",  # 科室
    "equ": "Check",       # 医疗设备 -> 检查项目
    "ite": "Check",       # 医学检验项目
    "pro": "Cure",        # 医疗程序 -> 治疗方法
    "bod": "Disease",     # 身体部位 -> 可尝试疾病相关
    "mic": "Disease",     # 微生物 -> 疾病相关
}

# disease-kb 所有节点类型（用于实体链接时检索）
KG_NODE_LABELS = [
    "Disease", "Drug", "Food", "Check",
    "Department", "Producer", "Symptom", "Cure"
]


def get_kg_label_for_ner_type(ner_type: str) -> str:
    """
    将 NER 实体类型映射到 KG 节点类型
    :param ner_type: CMeEE 类型 (dis, sym, dru, ...)
    :return: disease-kb 节点标签
    """
    return CMEEE_TO_KG_LABEL.get(ner_type, "Disease")  # 默认尝试 Disease
