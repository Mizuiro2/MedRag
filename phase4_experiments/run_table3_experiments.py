# -*- coding: utf-8 -*-
"""
Table 3：MMCU-Medical、CMB-Exam、CMB-Clin 各取 limit 题（默认每集 50，共 150 题）
指标按数据集分工（与论文 Table 3 安排一致）：
- MMCU-Medical、CMB-Exam：ACJ、PPL、ROUGE-R
- CMB-Clin：BLEU-1、BLEU-4、PPL、ROUGE-R
生成模型默认 Ollama qwen3.5:9b；方法顺序：Base → KGRAG → HyKGE（150×3=450 次答题）
"""

import os
import sys
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Any

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, "..", "phase2_rag_system"))

from data_loader import load_mmcu_medical, load_cmb_exam, load_cmb_clin, format_cmb_clin_input
from hykge_runner import HyKGERunner
from kgrag import KGRAG
from baseline_runner import BaselineRunner
from llm_clients import set_ollama_config
from metrics import (
    bleu1,
    bleu4,
    rouge_r,
    perplexity,
    judge_acj_batch,
    mean_acj,
    ACJ_JUDGE_MODEL,
    DEFAULT_PPL_HF_MODEL,
    format_chains_for_acj,
)

MODEL_MAP = {
    "ds": "deepseek",
    "qwen": "qwen3-max",
    "doubao": "doubao",
    "ollama": "ollama",
}
# 与用户需求一致：Base → KGRAG → HyKGE；共 150 题 × 3 = 450 次生成
METHODS = ["Base", "KGRAG", "HyKGE"]
DATASETS_ORDER = ["MMCU-Medical", "CMB-Exam", "CMB-Clin"]
DEFAULT_TABLE3 = os.path.join(_dir, "table3_results.txt")


def format_mcq_prompt(item: Dict) -> str:
    """选择题 prompt（与 run_experiments 一致）"""
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


def format_clin_prompt(item: Dict) -> str:
    body = format_cmb_clin_input(item.get("description", ""), item.get("question", ""))
    return (
        "【题型】开放式病例问答题（请根据病例信息与问题作答，使用专业医学表述；"
        "直接作答，勿重复整段病例全文。）\n\n"
        + body
    )


def format_item_prompt(dataset: str, item: Dict) -> str:
    if dataset == "CMB-Clin":
        return format_clin_prompt(item)
    return format_mcq_prompt(item)


def load_table3_corpus(per_dataset_limit: int) -> List[Dict[str, Any]]:
    """
    三个数据集各取 per_dataset_limit 条，共 3×per_dataset_limit 题（默认 150）。
    """
    rows: List[Dict[str, Any]] = []
    for it in load_mmcu_medical(limit=per_dataset_limit):
        rows.append({"dataset": "MMCU-Medical", "item": it, "is_mcq": True})
    for it in load_cmb_exam(limit=per_dataset_limit):
        rows.append({"dataset": "CMB-Exam", "item": it, "is_mcq": True})
    for it in load_cmb_clin(limit=per_dataset_limit):
        rows.append({"dataset": "CMB-Clin", "item": it, "is_mcq": False})
    return rows


def build_static_fields(corpus: List[Dict]) -> Tuple[List[str], List[str]]:
    """每题固定：发给模型的完整 prompt、标准答案。"""
    case_qs, refs = [], []
    for row in corpus:
        item = row["item"]
        ds = row["dataset"]
        case_qs.append(format_item_prompt(ds, item))
        refs.append(item.get("answer", "") or "")
    return case_qs, refs


def run_method(runner, corpus: List[Dict]) -> Tuple[List[str], List[str]]:
    """返回 (预测, 检索知识文本)；供 ACJ（论文 A.2.2 对检索知识打分）使用。"""
    preds: List[str] = []
    know_texts: List[str] = []
    for i, row in enumerate(corpus):
        ds = row["dataset"]
        item = row["item"]
        prompt = format_item_prompt(ds, item)
        try:
            res = runner.query(prompt)
            pred = res.get("answer", "") or ""
            chains = res.get("pruned_chains")
            if chains is None:
                chains = res.get("chains") or []
            know_texts.append(format_chains_for_acj(chains))
        except Exception as e:
            print(f"  [{i+1}] 错误: {e}")
            pred = ""
            know_texts.append("")
        preds.append(pred)
    return preds, know_texts


def _indices_for_dataset(corpus: List[Dict], dataset_name: str) -> List[int]:
    return [i for i, r in enumerate(corpus) if r["dataset"] == dataset_name]


def _take(lst: List[Any], indices: List[int]) -> List[Any]:
    return [lst[i] for i in indices]


def compute_metrics_by_dataset(
    corpus: List[Dict],
    preds: List[str],
    refs: List[str],
    case_qs: List[str],
    skip_acj: bool,
    ppl_hf_model: str,
    retrieved_knowledge: List[str],
) -> Dict[str, Dict]:
    """
    按数据集计算：MMCU / CMB-Exam → ACJ、PPL、ROUGE-R；CMB-Clin → BLEU-1/4、PPL、ROUGE-R。
    ACJ 按附录 A.2.2 对「检索知识」打 -1/0/1（见 judge_acj_batch）。
    """
    out: Dict[str, Dict] = {}
    for ds in DATASETS_ORDER:
        idx = _indices_for_dataset(corpus, ds)
        r = _take(refs, idx)
        p = _take(preds, idx)
        cq = _take(case_qs, idx)
        rk = _take(retrieved_knowledge, idx)
        rr = rouge_r(r, p)
        ppl = perplexity(p, hf_model_id=ppl_hf_model)
        ppl_v = ppl if isinstance(ppl, (int, float)) and ppl >= 0 else None

        if ds in ("MMCU-Medical", "CMB-Exam"):
            if skip_acj:
                acj = None
            else:
                print(f"    ACJ（Qwen3-max，检索知识 -1/0/1）[{ds}]...")
                acj = mean_acj(judge_acj_batch(cq, r, p, rk))
            out[ds] = {
                "ACJ": acj,
                "PPL": ppl_v,
                "ROUGE-R": rr,
                "BLEU-1": None,
                "BLEU-4": None,
            }
        else:
            out[ds] = {
                "ACJ": None,
                "PPL": ppl_v,
                "ROUGE-R": rr,
                "BLEU-1": bleu1(r, p),
                "BLEU-4": bleu4(r, p),
            }
    return out


def _fmt_m(v: Any, is_float: bool = True) -> str:
    if v is None:
        return "N/A"
    if is_float and isinstance(v, (int, float)):
        return f"{v:.4f}"
    return str(v)


def append_table3(
    out_path: str,
    rows: Dict[str, Dict[str, Dict]],
    model_label: str,
    judge_label: str,
    per_ds_limit: int,
    ppl_hf_model: str,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_total = 3 * per_ds_limit
    with open(out_path, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("=" * 90 + "\n")
        f.write(f"Table 3 记录 | {ts}\n")
        f.write(f"生成模型: {model_label}\n")
        f.write(f"ACJ 评判: {judge_label}\n")
        f.write(
            f"数据: MMCU-Medical / CMB-Exam / CMB-Clin 各 {per_ds_limit} 题，"
            f"合计 {n_total} 题；方法 Base→KGRAG→HyKGE，共 {n_total * 3} 次生成。\n"
        )
        f.write(
            "指标（均按论文 Appendix A.2.2）："
            "MMCU与CMB-Exam → ACJ,PPL,ROUGE-R；CMB-Clin → BLEU-1,BLEU-4,PPL,ROUGE-R。\n"
        )
        f.write(
            f"ACJ：检索知识相关性均值（-1/0/1）；PPL：生成序列在因果 LM 下困惑度（HF: {ppl_hf_model}）；"
            "ROUGE-R / BLEU：式 (10)(11–12)。\n"
        )
        f.write("=" * 90 + "\n\n")
        hdr = (
            f"{'Method':<10} {'Dataset':<16} {'ACJ':>8} {'PPL':>10} {'ROUGE-R':>9} "
            f"{'BLEU-1':>9} {'BLEU-4':>9}\n"
        )
        f.write(hdr)
        f.write("-" * 86 + "\n")
        for method in METHODS:
            if method not in rows:
                continue
            for ds in DATASETS_ORDER:
                v = rows[method].get(ds, {})
                f.write(
                    f"{method:<10} {ds:<16} "
                    f"{_fmt_m(v.get('ACJ')):>8} {_fmt_m(v.get('PPL')):>10} "
                    f"{_fmt_m(v.get('ROUGE-R')):>9} {_fmt_m(v.get('BLEU-1')):>9} "
                    f"{_fmt_m(v.get('BLEU-4')):>9}\n"
                )
        f.write("\n")


def _run_one_method(
    name: str,
    runner,
    corpus: List[Dict],
    case_qs: List[str],
    refs: List[str],
    skip_acj: bool,
    ppl_hf_model: str,
) -> Dict[str, Dict]:
    print(f"\n--- {name} ---")
    preds, know = run_method(runner, corpus)
    n_k = sum(1 for t in know if (t or "").strip())
    print(
        f"  检索知识非空条数（附录 A.2.2 ACJ 用）: {n_k}/{len(know)}；"
        "无检索文本时单题 ACJ 记为 -1（Base 无链故常为 0 条）"
    )
    by_ds = compute_metrics_by_dataset(
        corpus,
        preds,
        refs,
        case_qs,
        skip_acj=skip_acj,
        ppl_hf_model=ppl_hf_model,
        retrieved_knowledge=know,
    )
    for ds in DATASETS_ORDER:
        m = by_ds[ds]
        if ds in ("MMCU-Medical", "CMB-Exam"):
            print(
                f"  [{ds}] ACJ={_fmt_m(m.get('ACJ'))} PPL={_fmt_m(m.get('PPL'))} "
                f"ROUGE-R={_fmt_m(m.get('ROUGE-R'))}"
            )
        else:
            print(
                f"  [{ds}] BLEU-1={_fmt_m(m.get('BLEU-1'))} BLEU-4={_fmt_m(m.get('BLEU-4'))} "
                f"PPL={_fmt_m(m.get('PPL'))} ROUGE-R={_fmt_m(m.get('ROUGE-R'))}"
            )
    return by_ds


def main():
    parser = argparse.ArgumentParser(
        description="Table 3: 三数据集各 limit 题，Base / KGRAG / HyKGE"
    )
    parser.add_argument(
        "--model", "-m",
        choices=["ds", "qwen", "doubao", "ollama"],
        default="ollama",
        help="答题用 LLM（默认 ollama 本地）",
    )
    parser.add_argument("--ollama-model", default="qwen3.5:9b", help="Ollama 模型名")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--think", action="store_true", help="Ollama 开启 think（默认关）")
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="每个数据集抽取题数（默认 50，三集合计 150 题；每题 3 方法共 450 次答题）",
    )
    parser.add_argument("--max-tokens", type=int, default=2048, help="Base 生成最大 token")
    parser.add_argument(
        "--skip-acj",
        action="store_true",
        help="跳过 MMCU/CMB-Exam 的 ACJ（其余按数据集照常计算）",
    )
    parser.add_argument("-o", "--output", default=None, help="结果文件，默认 table3_results.txt")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="清空结果文件后再追加本次运行",
    )
    parser.add_argument(
        "--ppl-hf-model",
        type=str,
        default=DEFAULT_PPL_HF_MODEL,
        help=(
            "PPL 所用 HuggingFace 因果 LM（论文 A.2.2：对 LLM 输出文本的困惑度；"
            "API/Ollama 无 token logprob 时用同系列中文 LM 近似，默认 Qwen2.5-1.5B）"
        ),
    )
    args = parser.parse_args()

    model_name = MODEL_MAP[args.model]
    if args.model == "ollama":
        set_ollama_config(model=args.ollama_model, host=args.ollama_host, think=args.think)
        model_label = f"ollama:{args.ollama_model} (think={args.think})"
    else:
        model_label = f"{args.model} -> {model_name}"

    judge_label = (
        f"Qwen3-max API（{ACJ_JUDGE_MODEL}，附录 A.2.2：对检索知识 ACJ，取值 -1/0/1；"
        f"仅 MMCU/CMB-Exam）"
        if not args.skip_acj
        else "已跳过（MMCU/CMB-Exam 无 ACJ）"
    )

    print("加载数据（MMCU-Medical、CMB-Exam、CMB-Clin 各取 limit 条）...")
    corpus = load_table3_corpus(args.limit)
    case_qs, refs = build_static_fields(corpus)
    n = len(corpus)
    print(f"  合计 {n} 题（期望 3×{args.limit}={3 * args.limit}）；每题 3 方法 → {n * 3} 次生成")
    print(f"  PPL 计算用 HF 模型: {args.ppl_hf_model}")

    if not corpus:
        print("无数据，退出")
        return

    rows: Dict[str, Dict] = {}

    # Base
    base = BaselineRunner(model_name=model_name, max_tokens=args.max_tokens, temperature=0.6)
    try:
        rows["Base"] = _run_one_method(
            "Base",
            base,
            corpus,
            case_qs,
            refs,
            args.skip_acj,
            args.ppl_hf_model,
        )
    finally:
        base.close()

    # KGRAG
    kgrag = KGRAG(model_name=model_name)
    try:
        rows["KGRAG"] = _run_one_method(
            "KGRAG",
            kgrag,
            corpus,
            case_qs,
            refs,
            args.skip_acj,
            args.ppl_hf_model,
        )
    finally:
        kgrag.close()

    # HyKGE
    hykge = HyKGERunner(model_name=model_name)
    try:
        rows["HyKGE"] = _run_one_method(
            "HyKGE",
            hykge,
            corpus,
            case_qs,
            refs,
            args.skip_acj,
            args.ppl_hf_model,
        )
    finally:
        hykge.close()

    out_path = args.output or DEFAULT_TABLE3
    if args.overwrite:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(
                "Table 3 实验结果（默认每次运行追加；--overwrite 可清空后写入）\n"
            )
    append_table3(out_path, rows, model_label, judge_label, args.limit, args.ppl_hf_model)
    print(f"\nTable 3 已写入: {out_path}")


if __name__ == "__main__":
    main()
