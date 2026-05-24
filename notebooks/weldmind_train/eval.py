"""Greedy-decode evaluation + label parsing.

After the model generates a short completion, we look for the first canonical
label string ("cracks (CR)", "porosity (PO)", "lack of penetration (LP)",
"no defect (ND)") and use that as the prediction. If none of the four match,
we also try the parenthesised acronym (CR/PO/LP/ND) as a fallback.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from .collator import build_inference_messages
from .config import LABEL_LIST, LABEL_TO_INDEX

log = logging.getLogger(__name__)

# Lowercased canonical labels for matching.
_LABEL_LOWER = {lbl.lower(): lbl for lbl in LABEL_LIST}
_ACRONYM_RE = re.compile(r"\b(CR|PO|LP|ND)\b")
_ACRONYM_TO_LABEL = {
    "CR": "cracks (CR)",
    "PO": "porosity (PO)",
    "LP": "lack of penetration (LP)",
    "ND": "no defect (ND)",
}


def parse_label(generated_text: str) -> str | None:
    """Return the canonical label string the generation refers to, or None."""
    if not generated_text:
        return None
    lower = generated_text.lower()
    # Pick the canonical label with the earliest occurrence.
    best_pos = None
    best_lbl = None
    for lbl_lower, lbl_orig in _LABEL_LOWER.items():
        idx = lower.find(lbl_lower)
        if idx >= 0 and (best_pos is None or idx < best_pos):
            best_pos = idx
            best_lbl = lbl_orig
    if best_lbl is not None:
        return best_lbl
    m = _ACRONYM_RE.search(generated_text)
    if m:
        return _ACRONYM_TO_LABEL[m.group(1)]
    return None


def _generate_one(model, processor, image_path, max_new_tokens: int) -> str:
    from qwen_vl_utils import process_vision_info

    messages = build_inference_messages(image_path)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        gen = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id
                or processor.tokenizer.eos_token_id,
        )
    # Strip the prompt portion.
    new_tokens = gen[0, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True)


def evaluate_classification(
    model,
    processor,
    eval_records: Iterable,
    max_new_tokens: int = 16,
    desc: str = "eval",
) -> dict:
    """Run greedy classification across `eval_records` and return metrics.

    Returns a dict with: accuracy, macro_f1, per_class_f1 (list aligned with
    LABEL_LIST), n_total, n_parse_failed.
    """
    y_true: list[int] = []
    y_pred: list[int] = []
    n_parse_failed = 0

    for ex in tqdm(list(eval_records), desc=desc):
        text = _generate_one(model, processor, ex.image_path, max_new_tokens)
        parsed = parse_label(text)
        if parsed is None:
            n_parse_failed += 1
            # Treat as an arbitrary wrong class so the sample still counts.
            y_pred.append((ex.class_idx + 1) % len(LABEL_LIST))
        else:
            y_pred.append(LABEL_TO_INDEX[parsed])
        y_true.append(ex.class_idx)

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    acc = accuracy_score(y_true_arr, y_pred_arr)
    macro = f1_score(y_true_arr, y_pred_arr, average="macro",
                     labels=list(range(len(LABEL_LIST))), zero_division=0)
    per_class = f1_score(
        y_true_arr, y_pred_arr, average=None,
        labels=list(range(len(LABEL_LIST))), zero_division=0,
    )

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro),
        "per_class_f1": [float(x) for x in per_class],
        "n_total": int(len(y_true)),
        "n_parse_failed": int(n_parse_failed),
    }
