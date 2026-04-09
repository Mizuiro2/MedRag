# -*- coding: utf-8 -*-
"""
Phase4 实验主脚本 - Table 2
HyKGE、KGRAG、Base（纯 LLM）在 MMCU-Medical、CMB-Exam 上测试 EM、PCR

Ollama 本地示例（关闭 think，与 Qwen3 推理链无关）::

    python run_experiments.py --ollama-qwen35-2b   # qwen3.5:2b
    python run_experiments.py --ollama-qwen35-4b   # qwen3.5:4b
    python run_experiments.py --ollama-llama32-3b  # llama3.2:3b
    # 等价于 -m ollama --ollama-model …（默认不带 --think，即 think=false）
"""

import os
import sys
import argparse
from datetime import datetime
from typing import List, Dict

_dir = os.path.dirname(os.path.abspath(__file__))
# phase4 本地模块 + phase2 RAG 依赖
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, "..", "phase2_rag_system"))

from data_loader import load_mmcu_medical, load_cmb_exam
from hykge_runner import HyKGERunner
from kgrag import KGRAG
from baseline_runner import BaselineRunner
from llm_clients import set_ollama_config
from metrics import em_batch, pcr_batch

MODEL_MAP = {
    "ds": "deepseek",
    "qwen": "qwen3-max",
    "doubao": "doubao",
    "ollama": "ollama",
}
# Base 放最前：不加载 NER/GTE，先跑完轻量基线
METHODS_ORDER = ["Base", "HyKGE", "KGRAG"]
MCQ_DATASETS = ["MMCU-Medical", "CMB-Exam"]
DEFAULT_TABLE2 = os.path.join(_dir, "table2_results.txt")


def format_mcq_prompt(item: Dict) -> str:
    """将选择题格式化为完整 prompt，根据单选/多选使用不同提示"""
    q = item.get("question", "")
    opts = item.get("options", {})
    is_multi = item.get("is_multi_choice", False)
    if not opts:
        return q
    if is_multi:
        prefix = "【多选题】请从下列选项中选择所有正确答案（可能不止一个），直接输出选项字母如 A、AB、ABC 等。\n\n"
    else:
        prefix = "【单选题】请从下列选项中选择唯一正确答案，直接输出选项字母如 A、B、C 或 D。\n\n"
    parts = [prefix + q]
    for k in sorted(opts.keys()):
        v = opts[k]
        if v:
            parts.append(f"{k}. {v}")
    return "\n".join(parts)


def run_mcq(method_runner, items: List[Dict]) -> tuple:
    preds, golds, options_list, is_multi_choice = [], [], [], []
    for i, item in enumerate(items):
        prompt = format_mcq_prompt(item)
        gold = item["answer"]
        opts = item.get("options", {})
        is_multi = item.get("is_multi_choice", False)
        try:
            res = method_runner.query(prompt)
            pred = res.get("answer", "")
        except Exception as e:
            print(f"  [{i+1}] 错误: {e}")
            pred = ""
        preds.append(pred)
        golds.append(gold)
        options_list.append(opts)
        is_multi_choice.append(is_multi)
    return preds, golds, options_list, is_multi_choice


def _make_runner(method: str, model_name: str):
    if method == "HyKGE":
        return HyKGERunner(model_name=model_name)
    if method == "KGRAG":
        return KGRAG(model_name=model_name)
    return BaselineRunner(model_name=model_name)


def append_table2(
    out_path: str,
    table2: Dict,
    model_label: str,
    methods_used: List[str],
) -> None:
    """追加写入 table2_results.txt，不覆盖历史记录"""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(out_path, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("=" * 60 + "\n")
        f.write(f"Table 2 记录 | {ts}\n")
        f.write(f"模型: {model_label}\n")
        f.write(f"方法: {', '.join(methods_used)}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Method':<10} {'Dataset':<15} {'EM':>10} {'PCR':>10}\n")
        f.write("-" * 50 + "\n")
        for ds in MCQ_DATASETS:
            for method in methods_used:
                k = (method, ds)
                if k not in table2:
                    continue
                v = table2[k]
                pcr_str = f"{v['PCR']:.4f}" if v["PCR"] is not None else "N/A"
                f.write(f"{method:<10} {ds:<15} {v['EM']:>10.4f} {pcr_str:>10}\n")
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Table 2: HyKGE / KGRAG / Base",
        epilog=(
            "Ollama 快捷方式（think=false）：--ollama-qwen35-2b / --ollama-qwen35-4b / --ollama-llama32-3b。"
            " 也可手写：-m ollama --ollama-model llama3.2:3b（勿加 --think）。"
        ),
    )
    parser.add_argument(
        "--model", "-m",
        choices=["ds", "qwen", "doubao", "ollama"],
        default=None,
        help="ds / qwen / doubao / ollama(本地)；与 ollama 快捷参数二选一即可",
    )
    parser.add_argument(
        "--ollama-qwen35-2b",
        action="store_true",
        help="使用 Ollama 模型 qwen3.5:2b，且关闭 think（think=false；会设置 -m ollama）",
    )
    parser.add_argument(
        "--ollama-qwen35-4b",
        action="store_true",
        help="使用 Ollama 模型 qwen3.5:4b，且关闭 think（think=false；会设置 -m ollama）",
    )
    parser.add_argument(
        "--ollama-llama32-3b",
        action="store_true",
        help="使用 Ollama 模型 llama3.2:3b，且关闭 think（think=false；会设置 -m ollama）",
    )
    parser.add_argument(
        "--ollama-model",
        default="qwen3.5:9b",
        help="Ollama 模型名，如 qwen3.5:9b、qwen3.5:4b、llama3.2:3b",
    )
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434", help="Ollama 服务地址")
    parser.add_argument("--think", action="store_true", help="Ollama 开启 think（默认关闭，等价 think=false）")
    parser.add_argument("--limit", type=int, default=None, help="每个数据集最多测试条数（调试用）")
    parser.add_argument("--cmb-exam-size", type=int, default=4000, help="CMB-Exam 抽样数量（无 --limit 时）")
    parser.add_argument("--output", "-o", type=str, default=None, help="结果文件（默认 table2_results.txt）")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="清空结果文件后再写入本次运行（默认追加）",
    )
    parser.add_argument(
        "--no-base",
        action="store_true",
        help="不跑纯 LLM 基线 Base，仅 HyKGE + KGRAG",
    )
    parser.add_argument(
        "--rag-only",
        action="store_true",
        help="仅跑 HyKGE + KGRAG（与 --no-base 相同）",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="仅跑纯 LLM 基线 Base（不加载 NER/KG，用于快速验证 Ollama 与结果文件追加）",
    )
    args = parser.parse_args()

    _ollama_presets = (
        args.ollama_qwen35_2b,
        args.ollama_qwen35_4b,
        args.ollama_llama32_3b,
    )
    if sum(1 for p in _ollama_presets if p) > 1:
        parser.error(
            "以下参数仅可同时使用一个："
            "--ollama-qwen35-2b、--ollama-qwen35-4b、--ollama-llama32-3b"
        )

    if args.ollama_qwen35_2b:
        if args.model is not None and args.model != "ollama":
            parser.error("--ollama-qwen35-2b 仅可与 -m ollama 联用（或未指定 -m）")
        if args.think:
            parser.error(
                "--ollama-qwen35-2b 固定 think=false；若需 qwen3.5:2b 且 think，请使用："
                " -m ollama --ollama-model qwen3.5:2b --think"
            )
        args.model = "ollama"
        args.ollama_model = "qwen3.5:2b"
        args.think = False
    elif args.ollama_qwen35_4b:
        if args.model is not None and args.model != "ollama":
            parser.error("--ollama-qwen35-4b 仅可与 -m ollama 联用（或未指定 -m）")
        if args.think:
            parser.error(
                "--ollama-qwen35-4b 固定 think=false；若需 qwen3.5:4b 且 think，请使用："
                " -m ollama --ollama-model qwen3.5:4b --think"
            )
        args.model = "ollama"
        args.ollama_model = "qwen3.5:4b"
        args.think = False
    elif args.ollama_llama32_3b:
        if args.model is not None and args.model != "ollama":
            parser.error("--ollama-llama32-3b 仅可与 -m ollama 联用（或未指定 -m）")
        if args.think:
            parser.error(
                "--ollama-llama32-3b 固定 think=false；若需 llama3.2:3b 且 think，请使用："
                " -m ollama --ollama-model llama3.2:3b --think"
            )
        args.model = "ollama"
        args.ollama_model = "llama3.2:3b"
        args.think = False
    if args.model is None:
        parser.error(
            "请指定 -m/--model，或使用 --ollama-qwen35-2b / --ollama-qwen35-4b / --ollama-llama32-3b"
        )

    model_name = MODEL_MAP[args.model]
    if args.model == "ollama":
        set_ollama_config(
            model=args.ollama_model,
            host=args.ollama_host,
            think=args.think,
        )
        model_label = f"ollama:{args.ollama_model} (think={args.think})"
    else:
        model_label = f"{args.model} -> {model_name}"

    print(f"使用模型: {model_label}")

    if args.base_only:
        methods = ["Base"]
    else:
        methods = [
            m for m in METHODS_ORDER
            if m != "Base" or (not args.no_base and not args.rag_only)
        ]

    print("加载数据集...")
    mmcu = load_mmcu_medical(limit=args.limit)
    cmb_exam = load_cmb_exam(sample_size=args.cmb_exam_size, limit=args.limit)
    print(f"  MMCU-Medical: {len(mmcu)} 题")
    print(f"  CMB-Exam: {len(cmb_exam)} 题")
    if not mmcu:
        print("警告: MMCU-Medical 为空")

    table2 = {}
    for method in methods:
        print(f"\n{'='*60}")
        print(f"运行 {method} + {model_name}")
        print("=" * 60)
        runner = _make_runner(method, model_name)
        try:
            for ds_name, items in [("MMCU-Medical", mmcu), ("CMB-Exam", cmb_exam)]:
                if not items:
                    continue
                print(f"  [{ds_name}] {len(items)} 题...")
                preds, golds, opts, is_multi = run_mcq(runner, items)
                em = em_batch(preds, golds, opts)
                pcr = pcr_batch(preds, golds, opts, is_multi_choice=is_multi)
                table2[(method, ds_name)] = {"EM": em, "PCR": pcr}
                pcr_str = f"{pcr:.4f}" if pcr is not None else "N/A(无多选)"
                print(f"    EM={em:.4f}, PCR={pcr_str}")
        finally:
            runner.close()

    out_path = args.output or DEFAULT_TABLE2
    mode = "w" if args.overwrite else "a"
    if args.overwrite:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("Table 2 实验结果（追加模式：默认每次运行追加一段；使用 --overwrite 可清空）\n")
    append_table2(out_path, table2, model_label, methods)
    print(f"\nTable 2 已追加写入: {out_path}")


if __name__ == "__main__":
    main()
