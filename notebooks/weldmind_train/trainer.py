"""Custom Trainer that applies a per-token loss weight on top of causal LM CE.

The collator emits a `loss_weight` tensor of shape (B, T) parallel to
`input_ids`. We pop it from `inputs`, compute the standard CE per position
(with `ignore_index=-100`), then weight and average.

Reduction matches "sum of weighted CE / sum of weights" so that varying
sequence lengths within a batch don't bias the loss — equivalent to mean over
contributing tokens with each token's contribution scaled by its weight.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import Trainer


class WeightedCETrainer(Trainer):
    """Causal LM trainer with per-token loss weights.

    Expects `inputs["loss_weight"]` of shape (B, T), float, where 0.0 marks
    positions that should not contribute (the same positions already marked
    `-100` in `labels`).
    """

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        weights = inputs.pop("loss_weight")

        outputs = model(**inputs)
        logits = outputs.logits

        # Standard causal-LM shift: predict token t+1 from positions ≤ t.
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = inputs["labels"][:, 1:].contiguous()
        shifted_weights = weights[:, 1:].contiguous()

        flat_logits = shifted_logits.view(-1, shifted_logits.size(-1))
        flat_labels = shifted_labels.view(-1)
        flat_weights = shifted_weights.view(-1).to(flat_logits.dtype)

        flat_ce = F.cross_entropy(
            flat_logits,
            flat_labels,
            reduction="none",
            ignore_index=-100,
        )
        # CE returns 0 at ignored positions; the weight mask also zeros them,
        # but be defensive — coerce any NaN/inf to 0.
        flat_ce = torch.nan_to_num(flat_ce, nan=0.0, posinf=0.0, neginf=0.0)

        loss = (flat_ce * flat_weights).sum() / flat_weights.sum().clamp_min(1.0)

        return (loss, outputs) if return_outputs else loss
