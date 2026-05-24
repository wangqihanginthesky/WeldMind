"""Dataset construction for the joint-loss sweep.

`RiawelcExample` is the per-sample record. `build_train_pool` assembles all
training records from `data/train_batch/` + `train_results_merged.json`.
`stratified_subset` produces fixed, reproducible class-balanced subsets.
`build_eval_records` reads images from an extracted RIAWELC split directory.

These all produce lightweight records; tokenization happens in the collator.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import (
    DEFECT_FOLDER_TO_LABEL,
    LABEL_TO_INDEX,
    MERGED_JSON_PATH,
    RIAWELC_FULL_DIR,
    TRAIN_BATCH_DIR,
)

log = logging.getLogger(__name__)


@dataclass
class RiawelcExample:
    """One image with its label and optional pre-generated description."""

    image_path: Path
    label: str            # full string, e.g. "cracks (CR)"
    class_idx: int
    description: str | None  # may be None for eval-only records


def _iter_class_dir(class_dir: Path, label: str) -> Iterable[Path]:
    if not class_dir.is_dir():
        return ()
    return sorted(class_dir.glob("*.png"))


def build_train_pool() -> list[RiawelcExample]:
    """Join on-disk train_batch/*.png with train_results_merged.json.

    Drops images that lack a description in the merged json.
    """
    if not MERGED_JSON_PATH.exists():
        raise FileNotFoundError(f"merged json missing: {MERGED_JSON_PATH}")
    if not TRAIN_BATCH_DIR.exists():
        raise FileNotFoundError(f"train_batch dir missing: {TRAIN_BATCH_DIR}")

    merged = json.loads(MERGED_JSON_PATH.read_text(encoding="utf-8"))

    pool: list[RiawelcExample] = []
    skipped_no_desc = 0
    for folder, label in DEFECT_FOLDER_TO_LABEL.items():
        for p in _iter_class_dir(TRAIN_BATCH_DIR / folder, label):
            rec = merged.get(p.name)
            if rec is None or not rec.get("description"):
                skipped_no_desc += 1
                continue
            pool.append(RiawelcExample(
                image_path=p,
                label=label,
                class_idx=LABEL_TO_INDEX[label],
                description=rec["description"],
            ))

    if skipped_no_desc:
        log.warning("dropped %d train images without a description",
                    skipped_no_desc)
    log.info("train pool: %d examples", len(pool))
    return pool


def stratified_subset(
    pool: list[RiawelcExample],
    per_class: int,
    seed: int,
) -> list[RiawelcExample]:
    """Return `per_class` examples per defect class, deterministically."""
    rng = random.Random(seed)

    by_class: dict[int, list[RiawelcExample]] = {}
    for ex in pool:
        by_class.setdefault(ex.class_idx, []).append(ex)

    chosen: list[RiawelcExample] = []
    for cls_idx, items in sorted(by_class.items()):
        items_sorted = sorted(items, key=lambda e: e.image_path.name)
        if per_class > len(items_sorted):
            log.warning(
                "class %d only has %d examples, requested %d — taking all",
                cls_idx, len(items_sorted), per_class,
            )
            chosen.extend(items_sorted)
            continue
        chosen.extend(rng.sample(items_sorted, per_class))

    rng.shuffle(chosen)
    return chosen


def build_eval_records(
    split: str,
    root: Path | None = None,
) -> list[RiawelcExample]:
    """Read eval images from an extracted RIAWELC split directory.

    Args:
        split: one of "training", "validation", "testing"
        root: base dir; defaults to `RIAWELC_FULL_DIR` (data/riawelc_full/).
              The expected layout is `{root}/DB/{split}/collect/{class}/*.png`.
              If that path is missing, falls back to `{root}/{split}/{class}/*.png`
              (some RIAWELC packs use the flatter form).
    """
    base = (root or RIAWELC_FULL_DIR)
    candidates = [
        base / "DB" / split / "collect",
        base / split,
        base / split / "collect",
    ]
    split_dir = next((c for c in candidates if c.is_dir()), None)
    if split_dir is None:
        raise FileNotFoundError(
            f"could not find split '{split}' under {base}. "
            f"Tried: {[str(c) for c in candidates]}"
        )

    records: list[RiawelcExample] = []
    per_cls: dict[str, int] = {}
    for folder, label in DEFECT_FOLDER_TO_LABEL.items():
        cls_dir = split_dir / folder
        if not cls_dir.is_dir():
            log.warning("missing class folder in %s: %s", split, folder)
            continue
        for p in sorted(cls_dir.glob("*.png")):
            records.append(RiawelcExample(
                image_path=p, label=label,
                class_idx=LABEL_TO_INDEX[label], description=None,
            ))
            per_cls[folder] = per_cls.get(folder, 0) + 1

    log.info("eval split=%s loaded from %s: %s (total %d)",
             split, split_dir, per_cls, len(records))
    return records
