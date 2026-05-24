"""Collator that tokenizes Qwen3-VL chat messages and builds the per-token
loss-weight tensor used by `WeightedCETrainer`.

Per-variant assistant target:

- V_cls:       label only
- V_joint:     label + separator + description (uniform weight)
- V_weighted:  same target as V_joint but label-region tokens get weight λ

Why this is tricky for Qwen-VL: the chat template inserts `<|image_pad|>`
placeholders that the **processor** then expands into the right number of
tokens based on the image grid (vision tokens). So `tokenizer.encode(text)`
alone does NOT match the actual `input_ids` the model receives. We therefore
run the processor on the *prompt only* (with images) to get the true prompt
`input_ids`, then concatenate our hand-built target tokens on the right —
that way `labels` and `loss_weight` stay aligned with `input_ids`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from .config import SYSTEM_PROMPT, USER_INSTRUCTION

log = logging.getLogger(__name__)

# Separator between the label and the description.
SEPARATOR = "\n\n"


def _build_target(
    tokenizer,
    label_str: str,
    description: str | None,
    variant: str,
    lambda_label: float,
) -> tuple[list[int], list[float]]:
    eos_str = tokenizer.eos_token or "<|im_end|>"

    label_ids = tokenizer.encode(label_str, add_special_tokens=False)
    eos_ids = tokenizer.encode(eos_str, add_special_tokens=False)

    if variant == "V_cls":
        target_ids = label_ids + eos_ids
        weights = [1.0] * len(label_ids) + [1.0] * len(eos_ids)
        return target_ids, weights

    if description is None:
        raise ValueError(f"variant {variant!r} needs a description")

    sep_ids = tokenizer.encode(SEPARATOR, add_special_tokens=False)
    desc_ids = tokenizer.encode(description, add_special_tokens=False)

    if variant == "V_joint":
        w_label = 1.0
    elif variant == "V_weighted":
        w_label = float(lambda_label)
    else:
        raise ValueError(f"unknown variant: {variant!r}")

    target_ids = label_ids + sep_ids + desc_ids + eos_ids
    weights = (
        [w_label] * len(label_ids)
        + [w_label] * len(sep_ids)
        + [1.0] * len(desc_ids)
        + [1.0] * len(eos_ids)
    )
    return target_ids, weights


def _build_prompt_messages(image_path) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": USER_INSTRUCTION},
            ],
        },
    ]


@dataclass
class Qwen3VLCollator:
    processor: Any
    variant: str
    lambda_label: float = 3.0
    max_length: int = 2048

    def __post_init__(self):
        self.tokenizer = self.processor.tokenizer
        self.pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

    def _encode_prompt(self, image_path) -> dict[str, torch.Tensor]:
        """Run processor on prompt+image — returns the model-shaped tensors
        (input_ids, attention_mask, pixel_values, image_grid_thw) for this
        single example."""
        from qwen_vl_utils import process_vision_info

        messages = _build_prompt_messages(image_path)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )
        return inputs

    def __call__(self, batch: list) -> dict[str, torch.Tensor]:
        per_sample = []
        for ex in batch:
            prompt = self._encode_prompt(ex.image_path)
            prompt_ids = prompt["input_ids"][0].tolist()

            target_ids, target_w = _build_target(
                self.tokenizer, ex.label, ex.description,
                self.variant, self.lambda_label,
            )

            full_ids = prompt_ids + target_ids
            full_labels = [-100] * len(prompt_ids) + target_ids
            full_weights = [0.0] * len(prompt_ids) + target_w

            if len(full_ids) > self.max_length:
                overflow = len(full_ids) - self.max_length
                # Truncate from the *front* of the prompt to preserve target.
                full_ids = full_ids[overflow:]
                full_labels = full_labels[overflow:]
                full_weights = full_weights[overflow:]

            per_sample.append({
                "input_ids": full_ids,
                "labels": full_labels,
                "loss_weight": full_weights,
                "pixel_values": prompt.get("pixel_values"),
                "image_grid_thw": prompt.get("image_grid_thw"),
            })

        max_len = max(len(s["input_ids"]) for s in per_sample)

        def _pad(seq, value):
            return seq + [value] * (max_len - len(seq))

        input_ids = torch.tensor(
            [_pad(s["input_ids"], self.pad_id) for s in per_sample],
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [[1] * len(s["input_ids"]) + [0] * (max_len - len(s["input_ids"]))
             for s in per_sample],
            dtype=torch.long,
        )
        labels = torch.tensor(
            [_pad(s["labels"], -100) for s in per_sample], dtype=torch.long,
        )
        loss_weight = torch.tensor(
            [_pad(s["loss_weight"], 0.0) for s in per_sample],
            dtype=torch.float,
        )

        out: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_weight": loss_weight,
        }

        # Vision features: Qwen-VL packs pixel_values as a 2D tensor of
        # (sum_of_image_tokens_in_batch, feature_dim); image_grid_thw is one
        # row per image. So we concatenate directly across the batch.
        pvs = [s["pixel_values"] for s in per_sample if s["pixel_values"] is not None]
        if pvs:
            out["pixel_values"] = torch.cat(pvs, dim=0)
        grids = [s["image_grid_thw"] for s in per_sample if s["image_grid_thw"] is not None]
        if grids:
            out["image_grid_thw"] = torch.cat(grids, dim=0)

        return out


def build_inference_messages(image_path) -> list[dict]:
    """Same prompt schema used during training, but with no assistant turn —
    used for greedy generation at eval time."""
    return _build_prompt_messages(image_path)
