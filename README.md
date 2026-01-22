# Robust Image-to-Expression Regression (K=50)

This repository demonstrates a **reproducible and scalable ML pipeline** for
**image → high-dimensional regression (K=50)**, inspired by transcriptomic prediction from pathology.
The emphasis is **not** on chasing SOTA, but on building a pipeline that is:

- **Reproducible** (deterministic seeds, fixed splits, saved configs)
- **Data-integrity aware** (group-aware splits to prevent leakage)
- **Evaluation-robust** (rank-based metrics for log-scale gene expression)
- **Scale-ready** (clean separation of data/model/training; DDP-ready structure)

## Why Spearman correlation?
Gene expression targets are typically log-transformed (e.g., log1p / log2(TPM+1)) and often heavy-tailed.
For such targets, **relative ordering** is often more meaningful than absolute scale.
Therefore, we use **mean Spearman correlation** as the **primary evaluation metric**.

- **Primary metric:** mean Spearman correlation (gene-wise average)
- **Secondary metric:** mean Pearson correlation (for reference)
- **Training loss:** MSE (optimize continuous targets; evaluate with rank consistency)

---

## Project structure

- `train.py` — main training entrypoint (config-driven)
- `make_splits.py` — creates and saves group-aware splits
- `engine/`
  - `seed.py` — deterministic seeding (Python/NumPy/PyTorch/worker)
  - `data.py` — dataset, transforms, target normalization (train-only stats)
  - `model.py` — backbone + regression head
  - `trainer.py` — train/val loop (single-GPU in Week1; DDP-ready hooks)
  - `metrics.py` — Spearman/Pearson implementations
  - `io.py` — run directory + config/metrics serialization
- `configs/` — YAML configs (data/model/train)
- `splits/` — persisted split CSVs for exact reproducibility
- `results/` — run artifacts (gitignored)

---

## Data format (public/abstracted)

This repo expects a single CSV with paths, group identifiers, and K=50 targets.

**Required columns:**
- `path` (string): path to an image file
- `group_id` (string/int): grouping key to prevent leakage (e.g., patient_id)
- `y0 ... y49` (float): target vector (K=50), typically log-scale

Example schema: see `data/metadata_schema.md` and `data/example_metadata.csv`.

> Note: Real pathology WSIs and expression labels are often non-shareable.  
> This repo is designed so the **same pipeline** applies to private WSI pipelines,
> while using a public/abstracted dataset for demonstration.

---

## Setup

### Option A: pip (simple)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```
## Quick Start

The following commands run the full pipeline end-to-end using a small example setup.

```bash
# install dependencies
pip install -r requirements.txt

# generate group-aware train/validation splits
python make_splits.py \
  --metadata_csv data/metadata.csv \
  --out_csv splits/split_seed42.csv \
  --seed 42

# run an end-to-end dry run (sanity check)
python train.py --config configs/base.yaml --dry_run
```
This executes:
```kotlin
data loading → model forward → training → evaluation
```
and saves artifacts under `results/run_*/`.

## Running training

### Single GPU / CPU
```bash
python train.py --config configs/base.yaml
```

### Distributed Data Parallel (DDP)
- Requires CUDA; DDP on MPS is blocked. Use a Linux/GPU box for real DDP testing.
- Two processes on one node:
```bash
torchrun --nproc_per_node=2 train.py --config configs/base.yaml
```
- Quick sanity only: add `--dry_run`. Set `OMP_NUM_THREADS=1` if you want to silence torchrun’s advisory warning.

### Resume training (single/DDP)
Resumes model/optimizer/scheduler/AMP state, current epoch, and best metric.
```bash
# single process
python train.py --config configs/base.yaml --resume results/<run_name>/checkpoints/last.pt

# multi-process (same flags)
torchrun --nproc_per_node=2 train.py --config configs/base.yaml \
  --resume results/<run_name>/checkpoints/last.pt
```

## Run naming rule
By default a descriptive run name is auto-generated and used as `results/<run_name>/`:
```
<backbone>__bs<batch>x<accum>x<world>__lr<lr>__seed<seed>__<timestamp>
```
Example: `resnet18__bs4x4x1__lr0.0003__seed42__20260122_170000`  
Override with `logging.run_name` in `configs/base.yaml` if you need a custom name.

## Aggregating results
Aggregate all run `metrics.json` files into a Markdown table.
```bash
python scripts/aggregate_results.py \
  --results_dir results \
  --out_md results/summary.md
```
The table is also printed to stdout; omit `--out_md` if you just want the console output.

## Gradient accumulation

To support large effective batch sizes under limited GPU memory,
this pipeline implements **gradient accumulation**.

Effective global batch size is defined as:

```text
effective_batch = micro_batch × grad_accum_steps × world_size
```

Example:
- `batch_size = 4`
- `grad_accum_steps = 4`
- `world_size = 1`

→ effective global batch = **16**

Configure via YAML:
```yaml
train:
  grad_accum_steps: 4
```

During training, the effective batch size is logged for transparency.

## Outputs

Each run creates a self-contained directory under `results/`:


```text
results/run_YYYYMMDD_HHMMSS/
├── checkpoints/
│ ├── last.pt # latest checkpoint (resume-safe)
│ └── best.pt # best by primary metric
├── config_resolved.yaml
├── metrics.json
└── log.txt
```

This design ensures experiments are **fully reproducible and auditable**.


## Reproducibility

Reproducibility is treated as a first-class design goal in this repository.

Key measures include:

- Deterministic seeding across Python, NumPy, and PyTorch
- Group-aware train/validation splits persisted to disk
- Train-only target normalization to prevent data leakage
- Config-driven experiments via YAML
- Metrics, checkpoints, and resolved configs saved for every run

Given the same config file and split CSV, results are expected to be reproducible
up to hardware-level nondeterminism.
