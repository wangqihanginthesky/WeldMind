"""Paths, label maps, and experiment grids for the joint-loss sweep.

The defect label list mirrors `preprocessing/data/config.py:DEFECT_MAPPINGS`
verbatim — change one, change both.
"""

from __future__ import annotations

import os
from pathlib import Path


# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_DATA_DIR = PROJECT_ROOT / "data"

TRAIN_BATCH_DIR = REPO_DATA_DIR / "train_batch"
MERGED_JSON_PATH = REPO_DATA_DIR / "train_results_merged.json"

# Where the unpacked RIAWELC archive lives once `rar_extractor.py` is run.
# Expected layout after extraction:
#   {RIAWELC_FULL_DIR}/DB/training/collect/{Difetto1,Difetto2,Difetto4,NoDifetto}/*.png
#   {RIAWELC_FULL_DIR}/DB/validation/collect/...
#   {RIAWELC_FULL_DIR}/DB/testing/collect/...
RIAWELC_FULL_DIR = REPO_DATA_DIR / "riawelc_full"
RIAWELC_RAR_DIR = Path(
    os.environ.get(
        "RIAWELC_RAR_DIR",
        Path.home() / "Downloads" / "RIAWELC-main" / "Dataset_partitioned",
    )
)

# Where ModelScope caches model snapshots.
MODEL_CACHE_DIR = Path(
    os.environ.get("MODELSCOPE_CACHE", Path.home() / "models")
)

# Run outputs.
RUNS_DIR = PROJECT_ROOT / "notebooks" / "runs"
SPLITS_DIR = PROJECT_ROOT / "notebooks" / "splits"


# === Defect classes ===
DEFECT_FOLDER_TO_LABEL = {
    "Difetto1": "cracks (CR)",
    "Difetto2": "porosity (PO)",
    "Difetto4": "lack of penetration (LP)",
    "NoDifetto": "no defect (ND)",
}
LABEL_LIST = list(DEFECT_FOLDER_TO_LABEL.values())
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABEL_LIST)}


# === Model ===
MODEL_ID_PRIMARY = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_ID_FALLBACK = "Qwen/Qwen3-VL-4B"


# === Experiment grid (user-confirmed: small) ===
RUN_CONFIG = {
    "subset_sizes_per_class": [50, 200, 1000],
    "variants": ["V_cls", "V_joint", "V_weighted"],
    "seed": 20260523,
    "lambda_label_weighted": 3.0,           # weight on label tokens in V_weighted
    "epochs_by_size": {50: 8, 200: 5, 1000: 3},
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "lr_scheduler": "cosine",
    "warmup_ratio": 0.03,
    "weight_decay": 0.0,
    "logging_steps": 10,
    "lora": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    },
    "max_new_tokens_eval": 16,                # greedy decode budget for label parse
    "heldout_per_class": 200,                 # only used if real val unavailable
}


# === Chat template ===
SYSTEM_PROMPT = (
    "You are a precise welding inspection assistant. Given a radiographic "
    "image of a weld, name the defect class first, then describe what you see."
)
USER_INSTRUCTION = (
    "Classify this radiographic weld image into one of: "
    "cracks (CR), porosity (PO), lack of penetration (LP), no defect (ND). "
    "Reply with the class label, then a short inspection description."
)
