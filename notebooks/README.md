# WeldMind Training Notebook

Joint classification + caption-generation fine-tuning of `Qwen3-VL-4B` on
RIAWELC weld-defect radiographs. Tests the hypothesis that the auxiliary
generation objective improves classification accuracy in the small-data regime.

## What's here

```
notebooks/
├── README.md                       <- this file
├── requirements-train.txt          <- pinned env (CUDA 12.4 / H20)
├── train_qwen3vl_4b.ipynb          <- main driver (10 cells)
├── runs/                           <- adapter checkpoints + results.csv (created at run time)
├── splits/                         <- cached deterministic splits (created at run time)
└── weldmind_train/                 <- helper package, imported by the notebook
    ├── __init__.py
    ├── config.py                   <- paths, label maps, experiment grid
    ├── data.py                     <- dataset, stratified subset sampler, eval loader
    ├── collator.py                 <- Qwen-VL chat tokenization + loss-weight tensor
    ├── trainer.py                  <- WeightedCETrainer (per-token weighted CE)
    └── eval.py                     <- greedy decode + label parser + sklearn metrics
```

## Prerequisites

- 1× NVIDIA H20-NVLink (or any 24 GB+ SM90 GPU; bf16 + flash-attn-2 required).
- `data/train_batch/` with the 15,863 RIAWELC training PNGs.
- `data/train_results_merged.json` with `{filename: {label, description, gen_text}}`.
- `~/Downloads/RIAWELC-main/Dataset_partitioned/*.partNN.rar` for the validation
  + test splits (extracted by cell 2 of the notebook).

## Setup

```bash
cd /Users/wangqihang/Projects/paper/WeldMind
python -m venv .venv-train && source .venv-train/bin/activate
pip install -r notebooks/requirements-train.txt
pip install flash-attn==2.6.3 --no-build-isolation
```

If flash-attn build fails: install the prebuilt wheel matching your `torch`
ABI from https://github.com/Dao-AILab/flash-attention/releases.

## Run

```bash
jupyter lab notebooks/train_qwen3vl_4b.ipynb
```

Execute cells in order. Expected wall-clock on one H20: ~4.5 h for the full
9-run sweep + validation eval.

## Experiment matrix (configured in `weldmind_train/config.py`)

| | | |
| --- | --- | --- |
| Subset sizes per class | 50, 200, 1000 | 3 |
| Loss variants | V_cls, V_joint, V_weighted | 3 |
| Seeds | 1 (`20260523`) | 1 |
| **Total runs** | | **9** |

## Loss variants

- **V_cls**: assistant target is the label string alone. Pure classification.
- **V_joint**: assistant target is `<label>\n\n<description>`, all tokens
  weighted equally.
- **V_weighted**: same target as V_joint, but label tokens get weight λ=3
  while description tokens stay at weight 1. Pushes the model to *use* the
  description as scaffolding without letting its CE dominate.

The custom `WeightedCETrainer.compute_loss` reduces with
`sum(w · CE) / sum(w)` so varying sequence lengths don't bias the loss across
runs.

## Outputs

- `runs/n{size}_v{variant}/` – LoRA adapter for each run.
- `runs/results.csv` – per-run metrics: accuracy, macro F1, per-class F1,
  n_parse_failed.
- Plot in the final notebook cell: accuracy vs subset size, one line per
  variant.

## For the paper table

After the sweep, re-run the eval cell with `eval_split="testing"` to score
all 9 adapters on the official RIAWELC test split (2,441 imgs).
