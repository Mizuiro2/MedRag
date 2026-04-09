# -*- coding: utf-8 -*-
"""
HyKGE 消融实验（Table2 同指标 EM / PCR）：固定答题模型，逐模块关闭各跑一次。

模式（一次只关一块）：
  full              完整 HyKGE
  no_ho             无 HO：不调用假设生成，NER/分块仅用题干
  no_rerank         无重排序：直接取前 RERANK_TOP_K 条链
  no_kg             无图谱检索：仍跑 HO/NER/链接，Reader 无背景链
  no_entity_linking 无 GTE 链接，无检索
  no_ner            无 NER，无检索

默认 Ollama qwen3.5:2b、think=false；结果写入 hykge_ablation_results.txt（可 -o）。

用法::
    G:\\Anaconda\\envs\\rag_fyp\\python.exe run_hykge_ablation.py
    G:\\Anaconda\\envs\\rag_fyp\\python.exe run_hykge_ablation.py --limit 100 --modes full,no_ho,no_kg
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, "..", "phase2_rag_system"))

from config import KG_HOP_K, RERANK_TOP_K
from ner_module import extract_entities
from reranker import chunk_text, rerank_chains

from data_loader import load_mmcu_medical, load_cmb_exam
from hykge_runner import HyKGERunner
from llm_clients import get_hypothesis_output, get_answer, set_ollama_config
from metrics import em_batch, pcr_batch

# ---------------------------------------------------------------------------
# 消融定义与运行器（单文件内，便于维护）
# ---------------------------------------------------------------------------

ABLATION_MODES_ORDER: List[str] = [
    "full",
    "no_ho",
    "no_rerank",
    "no_kg",
    "no_entity_linking",
    "no_ner",
]

ABLATION_LABELS = {
    "full": "HyKGE（完整）",
    "no_ho": "−HO（假设生成）",
    "no_rerank": "−Rerank（重排序）",
    "no_kg": "−KG（图谱检索）",
    "no_entity_linking": "−Entity Linking（GTE 链接）",
    "no_ner": "−NER",
}


class HyKGEAblationRunner(HyKGERunner):
    """在 HyKGERunner 上增加 query_ablation(mode)。"""

    def query_ablation(self, question: str, mode: str) -> dict:
        if mode not in ABLATION_MODES_ORDER:
            raise ValueError(f"未知消融模式: {mode}，可选: {ABLATION_MODES_ORDER}")
        if mode == "full":
            return self.query(question)
        return self._query_ablation(question, mode)

    def _query_ablation(self, question: str, mode: str) -> dict:
        if mode == "no_ho":
            ho = question
            combined = question
        else:
            ho = get_hypothesis_output(self.model_name, question)
            if not ho:
                ho = question
            combined = question + " " + ho

        if mode == "no_ner":
            ner_entities: List = []
        else:
            ner_entities = extract_entities(
                combined, self.tokenizer, self.model, self.device, self.id2label
            )
            ner_entities = [e for e in ner_entities if e.get("entity")]

        if mode == "no_entity_linking":
            anchors: List[str] = []
        else:
            anchors = self.entity_linker.link_entities(ner_entities)

        if mode == "no_kg":
            chains = []
        elif mode in ("no_ner", "no_entity_linking") or not anchors:
            chains = []
        else:
            chains = self.kg_client.search_reasoning_chains(
                anchor_names=anchors,
                k_hops=KG_HOP_K,
                max_chains=100,
            )

        if mode == "no_rerank":
            pruned_chains = chains[:RERANK_TOP_K]
        else:
            fragments = chunk_text(combined)
            pruned_chains = rerank_chains(chains, fragments, top_k=RERANK_TOP_K)

        answer = get_answer(self.model_name, question, pruned_chains)

        return {
            "answer": answer,
            "ho": ho,
            "entities": [{"entity": e["entity"], "type": e["type"]} for e in ner_entities],
            "anchors": anchors,
            "chains": chains,
            "pruned_chains": pruned_chains,
            "ablation": mode,
        }


# ---------------------------------------------------------------------------
# 实验主流程
# ---------------------------------------------------------------------------

MCQ_DATASETS = ["MMCU-Medical", "CMB-Exam"]
DEFAULT_OUT = os.path.join(_dir, "hykge_ablation_results.txt")


def format_mcq_prompt(item: Dict) -> str:
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


def run_mcq_ablation(runner: HyKGEAblationRunner, mode: str, items: List[Dict]) -> tuple:
    preds, golds, options_list, is_multi_choice = [], [], [], []
    for i, item in enumerate(items):
        prompt = format_mcq_prompt(item)
        gold = item["answer"]
        opts = item.get("options", {})
        is_multi = item.get("is_multi_choice", False)
        try:
            res = runner.query_ablation(prompt, mode)
            pred = res.get("answer", "")
        except Exception as e:
            print(f"  [{mode}] [{i+1}] 错误: {e}")
            pred = ""
        preds.append(pred)
        golds.append(gold)
        options_list.append(opts)
        is_multi_choice.append(is_multi)
    return preds, golds, options_list, is_multi_choice


def append_ablation_report(
    out_path: str,
    rows: Dict[str, Dict[str, Dict]],
    model_label: str,
    modes_order: List[str],
    per_ds_limit_note: str,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(out_path, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("=" * 72 + "\n")
        f.write(f"HyKGE 消融实验 | {ts}\n")
        f.write(f"答题模型: {model_label}\n")
        f.write(f"数据: {per_ds_limit_note}\n")
        f.write("指标: EM（式6）、PCR（式7，多选题子集）\n")
        f.write("=" * 72 + "\n\n")
        hdr = f"{'Mode':<22} {'Dataset':<16} {'EM':>10} {'PCR':>10}\n"
        f.write(hdr)
        f.write("-" * 64 + "\n")
        for mode in modes_order:
            if mode not in rows:
                continue
            label = ABLATION_LABELS.get(mode, mode)
            for ds in MCQ_DATASETS:
                if ds not in rows[mode]:
                    continue
                v = rows[mode][ds]
                pcr_str = f"{v['PCR']:.4f}" if v["PCR"] is not None else "N/A"
                f.write(f"{label:<22} {ds:<16} {v['EM']:>10.4f} {pcr_str:>10}\n")
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="HyKGE 消融：各模块缺失对 EM/PCR 的影响")
    parser.add_argument(
        "--ollama-model",
        default="qwen3.5:2b",
        help="Ollama 答题模型（默认 qwen3.5:2b）",
    )
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--think", action="store_true", help="Ollama 开启 think（默认关）")
    parser.add_argument("--limit", type=int, default=None, help="每个数据集最多题数（调试）")
    parser.add_argument("--cmb-exam-size", type=int, default=4000, help="CMB-Exam 抽样上限（无 --limit 时）")
    parser.add_argument(
        "--modes",
        type=str,
        default=None,
        help="逗号分隔，如 full,no_ho,no_kg；默认跑全部模式",
    )
    parser.add_argument("-o", "--output", default=DEFAULT_OUT, help="结果 txt（追加）")
    parser.add_argument("--overwrite", action="store_true", help="运行前清空输出文件")
    args = parser.parse_args()

    set_ollama_config(model=args.ollama_model, host=args.ollama_host, think=args.think)
    model_label = f"ollama:{args.ollama_model} (think={args.think})"

    if args.modes:
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        for m in modes:
            if m not in ABLATION_MODES_ORDER:
                parser.error(f"未知模式: {m}，可选: {', '.join(ABLATION_MODES_ORDER)}")
    else:
        modes = list(ABLATION_MODES_ORDER)

    print("加载数据集…")
    mmcu = load_mmcu_medical(limit=args.limit)
    cmb_exam = load_cmb_exam(sample_size=args.cmb_exam_size, limit=args.limit)
    print(f"  MMCU-Medical: {len(mmcu)} 题；CMB-Exam: {len(cmb_exam)} 题")

    if args.overwrite and os.path.isfile(args.output):
        open(args.output, "w", encoding="utf-8").close()

    rows: Dict[str, Dict[str, Dict]] = {}
    runner = HyKGEAblationRunner(model_name="ollama")
    try:
        for mode in modes:
            print(f"\n{'='*60}\n消融模式: {mode} — {ABLATION_LABELS.get(mode, mode)}\n{'='*60}")
            rows[mode] = {}
            for ds_name, items in [("MMCU-Medical", mmcu), ("CMB-Exam", cmb_exam)]:
                if not items:
                    continue
                print(f"  [{ds_name}] {len(items)} 题…")
                preds, golds, opts, is_multi = run_mcq_ablation(runner, mode, items)
                em = em_batch(preds, golds, opts)
                pcr = pcr_batch(preds, golds, opts, is_multi_choice=is_multi)
                rows[mode][ds_name] = {"EM": em, "PCR": pcr}
                pcr_str = f"{pcr:.4f}" if pcr is not None else "N/A"
                print(f"    EM={em:.4f}, PCR={pcr_str}")
    finally:
        runner.close()

    lim_note = (
        f"MMCU / CMB-Exam 各 {args.limit} 题"
        if args.limit
        else f"MMCU 全量；CMB-Exam 抽样至多 {args.cmb_exam_size}（实际 {len(cmb_exam)}）"
    )
    append_ablation_report(args.output, rows, model_label, modes, lim_note)
    print(f"\n结果已追加写入: {args.output}")


if __name__ == "__main__":
    main()
