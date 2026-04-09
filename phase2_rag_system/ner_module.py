# -*- coding: utf-8 -*-
"""
NER Module - 论文 Section 4.1.2
使用 W2NER 从 Q ⊕ HO 中提取医学实体
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict

from config import W2NER_MODEL_PATH
from entity_mapping import get_kg_label_for_ner_type


def load_ner_model():
    """加载 W2NER 模型"""
    if not os.path.exists(W2NER_MODEL_PATH):
        raise FileNotFoundError(f"W2NER 模型路径不存在: {W2NER_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(W2NER_MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(W2NER_MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 加载 id2label
    label_config_path = os.path.join(W2NER_MODEL_PATH, "label_config.json")
    if os.path.exists(label_config_path):
        with open(label_config_path, "r", encoding="utf-8") as f:
            id2label = {int(k): v for k, v in json.load(f).get("id2label", {}).items()}
    else:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(W2NER_MODEL_PATH)
        id2label = getattr(config, "id2label", {}) or {}

    return tokenizer, model, device, id2label


def extract_entities(
    text: str,
    tokenizer,
    model,
    device,
    id2label: Dict,
    max_length: int = 512
) -> List[Dict]:
    """
    从文本中提取实体（与 test_trained_model 逻辑一致）
    返回: [{"entity": str, "type": str, "kg_label": str}, ...]
    """
    if not text or not text.strip():
        return []

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True
    )
    offset_mapping = inputs.pop("offset_mapping")[0].cpu().numpy()
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = predictions[0].cpu().numpy()
    attention_mask = inputs["attention_mask"][0].cpu().numpy()
    actual_length = attention_mask.sum().item()

    entities = []
    current_entity_tokens = []
    current_token_indices = []
    current_label_id = None

    for idx in range(1, min(actual_length - 1, len(tokens))):
        token = tokens[idx]
        label_id = int(labels[idx])
        label_name = id2label.get(label_id, f"LABEL_{label_id}") if label_id in id2label else (id2label.get(label_id) or f"LABEL_{label_id}")

        if token in ["[CLS]", "[SEP]", "[PAD]", "[UNK]"]:
            continue

        if label_name != "O" and label_id != 0:
            if current_label_id is None or current_label_id != label_id:
                if current_entity_tokens and current_label_id is not None and current_token_indices:
                    start_idx = current_token_indices[0]
                    end_idx = current_token_indices[-1]
                    if start_idx < len(offset_mapping) and end_idx < len(offset_mapping):
                        start_char = int(offset_mapping[start_idx][0])
                        end_char = int(offset_mapping[end_idx][1])
                        entity_text = text[start_char:end_char]
                        if entity_text:
                            ner_type = id2label.get(current_label_id, "")
                            entities.append({
                                "entity": entity_text,
                                "type": ner_type,
                                "kg_label": get_kg_label_for_ner_type(ner_type)
                            })
                current_entity_tokens = [token]
                current_token_indices = [idx]
                current_label_id = label_id
            else:
                current_entity_tokens.append(token)
                current_token_indices.append(idx)
        else:
            if current_entity_tokens and current_label_id is not None and current_token_indices:
                start_idx = current_token_indices[0]
                end_idx = current_token_indices[-1]
                if start_idx < len(offset_mapping) and end_idx < len(offset_mapping):
                    start_char = int(offset_mapping[start_idx][0])
                    end_char = int(offset_mapping[end_idx][1])
                    entity_text = text[start_char:end_char]
                    if entity_text:
                        ner_type = id2label.get(current_label_id, "")
                        entities.append({
                            "entity": entity_text,
                            "type": ner_type,
                            "kg_label": get_kg_label_for_ner_type(ner_type)
                        })
            current_entity_tokens = []
            current_token_indices = []
            current_label_id = None

    if current_entity_tokens and current_label_id is not None and current_token_indices:
        start_idx = current_token_indices[0]
        end_idx = current_token_indices[-1]
        if start_idx < len(offset_mapping) and end_idx < len(offset_mapping):
            start_char = int(offset_mapping[start_idx][0])
            end_char = int(offset_mapping[end_idx][1])
            entity_text = text[start_char:end_char]
            if entity_text:
                ner_type = id2label.get(current_label_id, "")
                entities.append({
                    "entity": entity_text,
                    "type": ner_type,
                    "kg_label": get_kg_label_for_ner_type(ner_type)
                })

    return entities
