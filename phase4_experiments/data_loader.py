# -*- coding: utf-8 -*-
"""
实验数据加载：MMCU-Medical, CMB-Exam, CMB-Clin
"""

import os
import json
import random
import pandas as pd
from typing import List, Dict, Any, Optional

# 路径
DATASETS_DIR = os.path.join(os.path.dirname(__file__), "datasets")
MMCU_PATH = os.path.join(DATASETS_DIR, "MMCU-Medical.xlsx")
CMB_EXAM_PATH = os.path.join(DATASETS_DIR, "CMB-Exam.json")
CMB_EXAM_ANSWERS_PATH = os.path.join(DATASETS_DIR, "CMB-Exam-answers.json")
CMB_CLIN_PATH = os.path.join(DATASETS_DIR, "CMB-Clin-qa.json")


def _parse_mmcu_sheet(df: pd.DataFrame, items: List[Dict], limit: Optional[int]) -> bool:
    """解析单个 sheet，按位置取列：0=题目, 1-4=选项A-D, 5=正确答案。返回是否已达 limit"""
    cols = df.columns.tolist()
    q_col = cols[0] if len(cols) > 0 else df.columns[0]
    a_col = cols[5] if len(cols) > 5 else df.columns[5]
    opt_cols = cols[1:5] if len(cols) >= 5 else list(df.columns[1:5])
    for _, row in df.iterrows():
        question = str(row[q_col]).strip() if pd.notna(row[q_col]) else ""
        if not question:
            continue
        options = {}
        for i, c in enumerate(opt_cols):
            key = chr(65 + i)
            val = str(row[c]).strip() if pd.notna(row[c]) else ""
            options[key] = val
        answer = str(row[a_col]).strip().upper() if pd.notna(row[a_col]) else ""
        if not answer:
            continue
        # MMCU 无题型标注，根据答案长度推断：多字母为多选
        is_multi = len(answer) > 1
        items.append({"question": question, "options": options, "answer": answer, "is_multi_choice": is_multi})
        if limit and len(items) >= limit:
            return True
    return False


def load_mmcu_medical(limit: Optional[int] = None) -> List[Dict]:
    """
    加载 MMCU-Medical 选择题，读取所有 sheet（医学三基299题、药理学200题、护理学690题等）
    返回: [{"question": str, "options": {A,B,C,D}, "answer": str}, ...]
    """
    if not os.path.exists(MMCU_PATH):
        return []
    xl = pd.ExcelFile(MMCU_PATH)
    items = []
    for sheet_name in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=sheet_name)
        if _parse_mmcu_sheet(df, items, limit):
            break
    return items[:limit] if limit else items


def load_cmb_exam(sample_size: int = 4000, seed: int = 42, limit: Optional[int] = None) -> List[Dict]:
    """
    加载 CMB-Exam
    :param limit: 若指定，按 id 顺序取前 limit 题（调试用，保证含 id 1,2,3,4,5 等）
    :param sample_size: limit 未指定时，随机抽样数量
    返回: [{"id": int, "question": str, "options": dict, "answer": str, "question_type": str, "is_multi_choice": bool}, ...]
    """
    with open(CMB_EXAM_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
    with open(CMB_EXAM_ANSWERS_PATH, "r", encoding="utf-8") as f:
        answers = json.load(f)
    
    answer_map = {a["id"]: a["answer"] for a in answers}
    
    # 合并题目与答案（保持 id 顺序）
    merged = []
    for q in questions:
        aid = q.get("id")
        ans = answer_map.get(aid, "")
        if not ans:
            continue
        qtype = q.get("question_type", "单项选择题")
        merged.append({
            "id": aid,
            "question": q.get("question", ""),
            "options": q.get("option", {}),
            "answer": ans.strip().upper(),
            "question_type": qtype,
            "is_multi_choice": "多项" in qtype or "多选" in qtype
        })
    
    if limit is not None:
        return merged[:limit]
    if len(merged) <= sample_size:
        return merged
    random.seed(seed)
    return random.sample(merged, sample_size)


def load_cmb_clin(limit: Optional[int] = None) -> List[Dict]:
    """
    加载 CMB-Clin 病例问答
    返回: [{"case_id": str, "description": str, "question": str, "answer": str}, ...]
    """
    with open(CMB_CLIN_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    items = []
    for case in data:
        desc = case.get("description", "")
        for qa in case.get("QA_pairs", []):
            items.append({
                "case_id": case.get("id", ""),
                "description": desc,
                "question": qa.get("question", ""),
                "answer": qa.get("answer", "")
            })
            if limit and len(items) >= limit:
                return items
    return items


def format_cmb_clin_input(description: str, question: str) -> str:
    """CMB-Clin 输入格式：[病例描述]\n\n问题：{question}"""
    return f"{description}\n\n问题：{question}"
