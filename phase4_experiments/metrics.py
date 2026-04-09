# -*- coding: utf-8 -*-
"""
实验指标 — ACL 2025 HyKGE（2025.acl-long.580）Appendix A.2.2

EM（式 6）、PCR（式 7）、ACJ（式 8）、PPL（式 9）、ROUGE（式 10）、BLEU（式 11–12）。
实现说明见各函数 docstring；ACJ 用 API 代专家，对「检索知识」按 {−1,0,1} 打分（无检索则 −1）。
"""

import math
import re
import warnings
from typing import Any, Dict, List, Optional, Sequence

from llm_clients import call_llm


ACJ_JUDGE_MODEL = "qwen3-max"


def format_chains_for_acj(chains: Optional[Sequence]) -> str:
    """将 KG 检索链格式化为一段文本，供 ACJ（附录 A.2.2）使用。"""
    if not chains:
        return ""
    lines = []
    for i, c in enumerate(chains[:40], 1):
        if isinstance(c, dict):
            s = (c.get("chain") or "").strip()
        else:
            s = str(c).strip()
        if s:
            lines.append(f"{i}. {s}")
    return "\n".join(lines)


ACJ_PROMPT_KNOWLEDGE = """你是医学信息检索评估员。请根据以下标准，
对「检索到的知识」与「问题」的相关性**只**打一个整数分。

评分取值：
• 1：检索知识与问题相关且有助于作答（Correlation）
• 0：知识与问题相关但对作答帮助不大（Relevant but useless）
• -1：知识与问题不相关（Irrelevant）

只输出一个数字：1、0 或 -1，不要解释、不要其它文字。

【问题 / 题目上下文】
{case_question}

【检索到的知识】
{knowledge}

【模型最终回答】（辅助判断知识是否可能被利用）
{hypothesis}
"""


def _parse_acj_triple(text: str) -> float:
    """解析 {-1, 0, 1}，与附录式 (8) 的专家打分一致。"""
    if not text:
        return -1.0
    t = text.strip()
    if re.search(r"-\s*1", t):
        return -1.0
    if re.search(r"\b0\b", t) or t == "0":
        return 0.0
    if re.search(r"\b1\b", t) or t == "1":
        return 1.0
    return -1.0


def judge_acj_single(
    case_question: str,
    reference: str,
    hypothesis: str,
    retrieved_knowledge: str = "",
) -> float:
    """
    单条 ACJ，取值 ∈ {{−1,0,1}}，对应附录式 (8)。
    若无检索知识（Base 或无链），记为 -1，不调 API。
    """
    hyp = (hypothesis or "")[:8000]
    cq = (case_question or "")[:12000]
    know = (retrieved_knowledge or "").strip()
    if not know:
        return -1.0
    _ = reference
    prompt = ACJ_PROMPT_KNOWLEDGE.format(
        case_question=cq,
        knowledge=know[:12000],
        hypothesis=hyp,
    )
    out = call_llm(ACJ_JUDGE_MODEL, prompt, max_tokens=32, temperature=0.0)
    return _parse_acj_triple(out or "")


def judge_acj_batch(
    case_questions: List[str],
    references: List[str],
    hypotheses: List[str],
    retrieved_knowledge: List[str],
    log_every: int = 5,
) -> List[float]:
    """批量 ACJ；retrieved_knowledge 与样本一一对应。"""
    scores = []
    n = len(references)
    for i in range(n):
        if log_every and (i + 1) % log_every == 0:
            print(f"    ACJ 评判进度 {i+1}/{n}")
        rk = retrieved_knowledge[i] if i < len(retrieved_knowledge) else ""
        s = judge_acj_single(
            case_questions[i],
            references[i],
            hypotheses[i],
            retrieved_knowledge=rk,
        )
        scores.append(s)
    return scores


def mean_acj(scores: List[float]) -> float:
    """式 (8)：样本均值，范围为 [-1, 1]。"""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _tokenize_zh(text: str) -> List[str]:
    """论文实验使用 jieba 分词（正文 5.1.6 / A.2.3）。"""
    try:
        import jieba

        text = (text or "").strip()
        if not text:
            return []
        return [w for w in jieba.lcut(text) if w.strip()]
    except Exception:
        return list((text or "").replace(" ", ""))


def normalize_answer(s: str) -> str:
    """规范化答案：去空格、转大写、提取选项字母"""
    if not s or not isinstance(s, str):
        return ""
    s = str(s).strip().upper()
    letters = re.findall(r'[A-E]', s)
    return "".join(sorted(set(letters))) if letters else s.strip()


def extract_predicted_option(pred: str, options: Optional[dict] = None) -> str:
    """从模型输出中提取选项。支持格式：A、答案A、选A、A. xxx 等"""
    if not pred:
        return ""
    pred = str(pred).strip()
    letters = re.findall(r'\b[A-E]\b', pred)
    if letters:
        return "".join(sorted(set(letters)))
    m = re.search(r'(?:答案|选择?|正确选项)[:：]?\s*([A-E]+)', pred, re.I)
    if m:
        return "".join(sorted(set(m.group(1).upper())))
    return ""


def em_single(pred: str, gold: str, options: Optional[dict] = None) -> float:
    """单题 EM：式 (6)，预测与标准答案形式完全一致则 1，否则 0（等价于 Accuracy）。"""
    pred_norm = extract_predicted_option(pred, options) or normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    return 1.0 if pred_norm == gold_norm else 0.0


def pcr_single(pred: str, gold: str, options: Optional[dict] = None) -> float:
    """
    单题 PCR：式 (7)。多选题上允许漏选但不得有错选；实现为 pred ⊆ gold 且 pred 非空。
    正文亦说明「仅有漏选而无错误选项」则 PCR 判对。
    """
    pred_norm = extract_predicted_option(pred, options) or normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    if not gold_norm:
        return 0.0
    gold_set = set(gold_norm)
    pred_set = set(pred_norm)
    # 有错误答案（选了不该选的）→ 0
    if pred_set - gold_set:
        return 0.0
    # 空预测不算正确
    if not pred_set:
        return 0.0
    # 无错误答案且 pred ⊆ gold（含漏选）→ 正确
    return 1.0


def em_batch(preds: List[str], golds: List[str], options_list: Optional[List[dict]] = None) -> float:
    """批量 EM 平均"""
    n = len(preds)
    if n == 0:
        return 0.0
    opts = options_list or [None] * n
    return sum(em_single(p, g, o) for p, g, o in zip(preds, golds, opts)) / n


def pcr_batch(
    preds: List[str],
    golds: List[str],
    options_list: Optional[List[dict]] = None,
    is_multi_choice: Optional[List[bool]] = None,
) -> Optional[float]:
    """
    批量 PCR 平均，仅针对多选题。
    :param is_multi_choice: 每题为多选则为 True，否则不参与 PCR 计算
    :return: 多选题的 PCR，若无多选题则返回 None
    """
    n = len(preds)
    if n == 0:
        return None
    opts = options_list or [None] * n
    multi_flags = is_multi_choice or [False] * n
    multi_preds = [p for p, m in zip(preds, multi_flags) if m]
    multi_golds = [g for g, m in zip(golds, multi_flags) if m]
    multi_opts = [o for o, m in zip(opts, multi_flags) if m]
    if not multi_preds:
        return None
    return sum(pcr_single(p, g, o) for p, g, o in zip(multi_preds, multi_golds, multi_opts)) / len(multi_preds)


def _char_ngrams(s: str, n: int):
    chars = list(s) if s else []
    return [tuple(chars[i:i+n]) for i in range(len(chars)-n+1)] if len(chars) >= n else []


def _bleu_nltk_sentence_bleu(
    references: List[str],
    hypotheses: List[str],
    max_n: int,
) -> float:
    """
    论文式 (11)(12)：BLEU = BP·exp(Σ w_n log P_n)；对中文参考/假设用 jieba 词元。
    max_n=1 对应 BLEU-1（仅一元精度）；max_n=4 对应 BLEU-4（1–4 元加权）。
    """
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    except ImportError:
        return _bleu_char_fallback(references, hypotheses, max_n)

    smooth = SmoothingFunction().method1
    weights_map = {
        1: (1.0, 0.0, 0.0, 0.0),
        4: (0.25, 0.25, 0.25, 0.25),
    }
    weights = weights_map.get(max_n)
    if not weights:
        weights = (0.25, 0.25, 0.25, 0.25)
    scores = []
    for ref, hyp in zip(references, hypotheses):
        r = _tokenize_zh(ref or "")
        h = _tokenize_zh(hyp or "")
        if not r:
            scores.append(1.0 if not h else 0.0)
            continue
        try:
            s = sentence_bleu([r], h, weights=weights, smoothing_function=smooth)
        except Exception:
            s = 0.0
        scores.append(float(s))
    return sum(scores) / len(scores) if scores else 0.0


def _bleu_char_fallback(
    references: List[str], hypotheses: List[str], max_n: int
) -> float:
    """无 NLTK 时的字符级近似（不推荐）。"""
    scores = []
    for ref, hyp in zip(references, hypotheses):
        ref = ref or ""
        hyp = hyp or ""
        ref_ngrams = _char_ngrams(ref, n=1 if max_n == 1 else 4)
        hyp_ngrams = _char_ngrams(hyp, n=1 if max_n == 1 else 4)
        if not ref_ngrams:
            scores.append(1.0 if not hyp and not ref else 0.0)
            continue
        ref_cnt = {}
        for g in ref_ngrams:
            ref_cnt[g] = ref_cnt.get(g, 0) + 1
        match = 0
        hyp_cnt = {}
        for g in hyp_ngrams:
            hyp_cnt[g] = hyp_cnt.get(g, 0) + 1
        for g, c in hyp_cnt.items():
            match += min(c, ref_cnt.get(g, 0))
        scores.append(match / len(hyp_ngrams) if hyp_ngrams else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def bleu1(references: List[str], hypotheses: List[str]) -> float:
    """BLEU-1：论文 Table 3 中用于 answer precision（式 11，取 n=1 情形）。"""
    return _bleu_nltk_sentence_bleu(references, hypotheses, max_n=1)


def bleu4(references: List[str], hypotheses: List[str]) -> float:
    """BLEU-4：论文 Table 3 中用于 fluency（式 11–12，1–4 元）。"""
    return _bleu_nltk_sentence_bleu(references, hypotheses, max_n=4)


def _rouge_r_char_recall(ref: str, hyp: str) -> float:
    """
    字级召回（无 rouge 库或词级不适用时的回退）：参考串中字符在预测中出现的比例。
    适用于中文与极短参考答案（如选择题选项字母）。
    """
    if not ref or not hyp:
        return 0.0
    ref = str(ref).strip()
    hyp = str(hyp)
    if not ref:
        return 0.0
    hit = sum(1 for c in ref if c in hyp)
    return hit / len(ref)


def rouge_r(references: List[str], hypotheses: List[str]) -> float:
    """
    式 (10)：ROUGE-R = 参考中 n-gram 被生成覆盖的比例；实现为 rouge1 **recall**。
    中文参考/生成都先 jieba 分词再拼成空格分隔串，供 rouge 包分词。
    """
    if not references:
        return 0.0
    use_fallback = False
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        try:
            from rouge import Rouge
            rouge = Rouge()
            scores = []
            for ref, hyp in zip(references, hypotheses):
                if not ref or not hyp:
                    scores.append(0.0)
                    continue
                rt = " ".join(_tokenize_zh(ref))
                ht = " ".join(_tokenize_zh(hyp))
                if not rt.strip():
                    scores.append(0.0)
                    continue
                try:
                    r = rouge.get_scores(ht, rt)[0]
                    scores.append(r.get("rouge-1", {}).get("r", 0.0))
                except Exception:
                    scores.append(_rouge_r_char_recall(ref, hyp))
            return sum(scores) / len(scores) if scores else 0.0
        except ImportError:
            use_fallback = True
    if use_fallback:
        scores = [_rouge_r_char_recall(r, h) for r, h in zip(references, hypotheses)]
        return sum(scores) / len(scores) if scores else 0.0
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=False)
    scores = []
    for ref, hyp in zip(references, hypotheses):
        if not ref or not hyp:
            scores.append(0.0)
            continue
        rt = " ".join(_tokenize_zh(ref))
        ht = " ".join(_tokenize_zh(hyp))
        if not rt.strip():
            scores.append(0.0)
            continue
        try:
            s = scorer.score(rt, ht)
            r_val = float(s["rouge1"].recall)
            if r_val == 0.0 and (len(str(ref).strip()) <= 8 or len(str(hyp)) > 20):
                r_val = _rouge_r_char_recall(ref, hyp)
            scores.append(r_val)
        except Exception:
            scores.append(_rouge_r_char_recall(ref, hyp))
    return sum(scores) / len(scores) if scores else 0.0


# 论文 Table 3：与中文医学实验一致，默认用 Qwen2.5 系列因果 LM 对「生成文本」打分（非英文 GPT-2）。
DEFAULT_PPL_HF_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
_ppl_lm_cache: Dict[str, Any] = {}


def _get_ppl_lm(hf_model_id: str):
    """缓存 tokenizer + causal LM，避免 Table3 重复加载。"""
    if hf_model_id not in _ppl_lm_cache:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
        if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
            tok.pad_token_id = tok.eos_token_id
        load_kw: Dict[str, Any] = {"trust_remote_code": True}
        if torch.cuda.is_available():
            import transformers as _tf

            major = int(_tf.__version__.split(".")[0])
            if major >= 5:
                load_kw["dtype"] = torch.float16
            else:
                load_kw["torch_dtype"] = torch.float16
        mod = AutoModelForCausalLM.from_pretrained(hf_model_id, **load_kw)
        mod.eval()
        _ppl_lm_cache[hf_model_id] = (tok, mod)
    return _ppl_lm_cache[hf_model_id]


def perplexity(
    generated: List[str],
    hf_model_id: Optional[str] = None,
    reference_model_name: Optional[str] = None,
) -> float:
    """
    论文 **ACL 2025 HyKGE（2025.acl-long.580）Appendix A.2.2 / 正文 5.1.5**：
    对 LLM **输出** 的困惑度（Perplexity of LLMs' output）。

    附录 Eq. (9) 写法（对预测 token 序列 w_1…w_N）：
    PPL = 2^{ - (1/N) * sum_i log2 P(w_i) }，
    其中 P(w_i) 为模型对第 i 个 token 的概率；即用 **平均负对数似然**（以 bit 计）再对 2 取指数。
    与 PyTorch/HuggingFace 中 **自然对数** 下的平均 token 负对数似然 L 满足：exp(L) = 2^{(1/ln2)·(...)}，
    标准实现里 **exp(outputs.loss)**（loss 为 mean -ln P，单位 nats）与上式在数学上对应同一困惑度定义。

    若答题为 API/Ollama 且无法取得**同一**生成模型的 token 概率，则用 HF 上**同系列**中文因果 LM
   （默认 ``DEFAULT_PPL_HF_MODEL``）对生成文本做 teacher-forcing 打分，而非无关的英文 GPT-2。

    :param hf_model_id: HuggingFace 模型 id；可用 ``--ppl-hf-model`` 改为 Qwen2.5-7B 等（需显存）。
    :param reference_model_name: 兼容旧参数名；若设置且未传 hf_model_id，则视为 hf_model_id。
    :return: 有效 PPL，失败时 -1（调用方常显示为 N/A）。
    """
    mid = hf_model_id or reference_model_name or DEFAULT_PPL_HF_MODEL
    try:
        tokenizer, model = _get_ppl_lm(mid)
        import torch
    except Exception as e:
        warnings.warn(
            f"PPL：无法加载 HF 因果语言模型 {mid!r}（{e}）。"
            "请执行 pip install -U \"transformers>=4.40\"，或改用 --ppl-hf-model 指定本机可加载的模型。",
            stacklevel=2,
        )
        return -1.0

    total_nll = 0.0
    total_tokens = 0
    device = next(model.parameters()).device

    for text in generated:
        text = (text or "").strip()
        if not text:
            continue
        try:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )
            input_ids = inputs["input_ids"].to(device)
            ntok = int(input_ids.size(1))
            if ntok < 1:
                continue
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            if hasattr(loss, "item"):
                loss = loss.item()
            if math.isnan(loss) or math.isinf(loss):
                continue
            total_nll += float(loss) * ntok
            total_tokens += ntok
        except Exception:
            continue

    if total_tokens == 0:
        warnings.warn(
            "PPL：有效 token 数为 0（生成文本全空或逐条前向均失败），无法计算困惑度。",
            stacklevel=2,
        )
        return -1.0
    avg = total_nll / total_tokens
    if math.isnan(avg) or math.isinf(avg):
        return -1.0
    return math.exp(avg)
