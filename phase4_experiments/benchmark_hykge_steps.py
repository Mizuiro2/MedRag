# -*- coding: utf-8 -*-
"""
HyKGE 各模块耗时测试（单次 query 内分步计时）。

步骤与 hykge_runner.HyKGERunner.query_with_timings 一致：
  hypothesis_output → ner → entity_linking → kg_retrieval → rerank → llm_reader

用法示例::
    G:\\Anaconda\\envs\\rag_fyp\\python.exe benchmark_hykge_steps.py -m ollama --ollama-model qwen3.5:4b
    G:\\Anaconda\\envs\\rag_fyp\\python.exe benchmark_hykge_steps.py -m ollama --from-mmcu --limit 3
"""

import argparse
import os
import sys
from typing import Dict, List

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, "..", "phase2_rag_system"))

from data_loader import load_mmcu_medical
from hykge_runner import HyKGERunner
from llm_clients import set_ollama_config

MODEL_MAP = {
    "ds": "deepseek",
    "qwen": "qwen3-max",
    "doubao": "doubao",
    "ollama": "ollama",
}

STEP_LABELS = {
    "hypothesis_output": "1. Hypothesis Output (HO)",
    "ner": "2. NER (Q⊕HO)",
    "entity_linking": "3. Entity Linking (GTE)",
    "kg_retrieval": "4. KG Retrieval",
    "rerank": "5. Rerank (chunk + rerank)",
    "llm_reader": "6. LLM Reader",
    "total": "总计 (pipeline)",
}


def format_mcq_prompt(item: dict) -> str:
    q = item.get("question", "")
    opts = item.get("options", {})
    is_multi = item.get("is_multi_choice", False)
    if not opts:
        return q
    if is_multi:
        prefix = "【多选题】请从下列选项中选择所有正确答案，直接输出选项字母如 A、AB 等。\n\n"
    else:
        prefix = "【单选题】请从下列选项中选择唯一正确答案，直接输出选项字母。\n\n"
    parts = [prefix + q]
    for k in sorted(opts.keys()):
        v = opts[k]
        if v:
            parts.append(f"{k}. {v}")
    return "\n".join(parts)


def _print_timings(timings: Dict[str, float], title: str) -> None:
    print(f"\n--- {title} ---")
    for key in (
        "hypothesis_output",
        "ner",
        "entity_linking",
        "kg_retrieval",
        "rerank",
        "llm_reader",
        "total",
    ):
        if key in timings:
            label = STEP_LABELS.get(key, key)
            print(f"  {label:<42} {timings[key]:10.3f} s")
    over = sum(
        timings[k]
        for k in (
            "hypothesis_output",
            "ner",
            "entity_linking",
            "kg_retrieval",
            "rerank",
            "llm_reader",
        )
        if k in timings
    )
    if "total" in timings:
        gap = timings["total"] - over
        if abs(gap) > 0.001:
            print(f"  {'(total − 分步之和，调度开销)':<42} {gap:10.3f} s")


def _aggregate(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    n = len(rows)
    return {k: sum(r[k] for r in rows) / n for k in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="HyKGE 分模块耗时（HyKGERunner.query_with_timings）")
    parser.add_argument(
        "-m",
        "--model",
        choices=list(MODEL_MAP.keys()),
        default="ollama",
        help="与 run_experiments 相同的后端名",
    )
    parser.add_argument("--ollama-model", default="qwen3.5:9b", help="Ollama 模型名")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--think", action="store_true", help="Ollama 开启 think")
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="单条测试用用户问题（与 --from-mmcu 二选一；默认内置医学问句）",
    )
    parser.add_argument(
        "--from-mmcu",
        dest="use_mmcu",
        action="store_true",
        help="从 MMCU-Medical 取题（需 format 成选择题 prompt）",
    )
    parser.add_argument("--limit", type=int, default=1, help="--from-mmcu 时抽取条数；单题模式忽略")
    args = parser.parse_args()

    model_name = MODEL_MAP[args.model]
    if args.model == "ollama":
        set_ollama_config(model=args.ollama_model, host=args.ollama_host, think=args.think)

    questions: List[str] = []
    if args.use_mmcu:
        items = load_mmcu_medical(limit=max(1, args.limit))
        for it in items:
            questions.append(format_mcq_prompt(it))
        if not questions:
            print("MMCU 无数据，退出")
            return
    else:
        q = args.question or (
            "2型糖尿病患者出现多饮、多尿，首选应考虑哪类检查？"
            "（测试用：请简要说明，不必真选选项。）"
        )
        questions = [q]

    print(f"后端: {args.model} -> {model_name}" + (f"  ollama:{args.ollama_model}" if args.model == "ollama" else ""))
    print(f"题目数: {len(questions)}")
    print("首次运行会加载 NER / GTE / Neo4j，首题总耗时会偏大。")

    runner = HyKGERunner(model_name=model_name)
    all_timings: List[Dict[str, float]] = []
    try:
        for i, prompt in enumerate(questions):
            _, timings = runner.query_with_timings(prompt)
            all_timings.append(timings)
            title = f"第 {i+1}/{len(questions)} 题" if len(questions) > 1 else "单题分步耗时"
            _print_timings(timings, title)
    finally:
        runner.close()

    if len(questions) > 1:
        avg = _aggregate(all_timings)
        _print_timings(avg, f"各题平均（共 {len(questions)} 题）")


if __name__ == "__main__":
    main()
