# Face Re-Identification for Retail Surveillance

A hybrid deep learning pipeline for identity retrieval and unknown-person rejection.

## Highlights
- A two-model face re-identification system using FaceNet + Vision Transformer with late fusion.
- Reached **95.44%** accuracy on hidden testset
- Part of a national competition solution that won **2nd place** and a **50K EGP** prize from a **200K EGP** total prize pool.

## Pipeline Design
![alt text](assets/architecture.jpeg)



## Project overview

- FaceNet and ViT produce 1024-D embeddings.
- reference embeddings are built from train identities.
- distances are fused at inference.
- unknown identities are rejected with a threshold.


## Table of Contents

- [Repository Layout](#repository-layout)
- [Environment Setup](#environment-setup)
- [Data Format](#data-format)
- [Configuration](#configuration)
- [Inference (Late Fusion)](#inference-late-fusion)
- [Training](#training)
  - [FaceNet Stage 1](#facenet-stage-1)
  - [FaceNet Stage 2 (Resume)](#facenet-stage-2-resume)
  - [ViT Stage 1 (5-fold)](#vit-stage-1-5-fold)
  - [ViT Stage 2 (Aggressive Finetune)](#vit-stage-2-aggressive-finetune)
- [Outputs](#outputs)

---

## Repository Layout

High-level structure:

- `src/` — Python package (datasets, models, training, inference)
- `configs/` — YAML configs (data paths, training hyperparameters, inference settings)
- `data/` — Local CSVs + images
- `outputs/` — Checkpoints, logs, and generated predictions
- `commands.txt` — Quick command cheatsheet

---

## Environment Setup

### Option A — Conda (recommended)

```bash
conda create -n face_id python=3.10 -y
conda activate face_id
pip install -r requirements.txt
```

---

## Data Format

The refactored code expects a local `data/` directory with:

- `data/trainset.csv` (training labels)
  - Supported formats:
    - With header: columns `image_path`, `gt`
    - Or without header: 2 columns (image path, label)
  - `image_path` is typically relative to `data/` (e.g. `train/person_0/0.jpg`).

- `data/eval_set.csv` (test/eval images)
  - Supported formats:
    - With header: column `image_path`
    - Or 1-column CSV

- Images:
  - `data/train/...`
  - `data/test/...`

- Optional scoring file: `data/gt.csv`
  - This project supports more than one ground-truth format.
  - In this workspace, `gt.csv` may store labels inside an `objects` column containing Python-literal dict/list strings.

---

## Configuration

### Dataset config

The default dataset config is:
- `configs/data/retail_faces.yaml`

It is repo-relative by default:
- `data.base_dir: data`
- `data.train_csv: data/trainset.csv`
- `data.test_csv: data/eval_set.csv`

### Training configs

- FaceNet:
  - `configs/facenet/stage1.yaml`
  - `configs/facenet/stage2.yaml`

- ViT:
  - `configs/vit/stage1.yaml`
  - `configs/vit/stage2.yaml`

### Inference config

- `configs/inference/ensemble.yaml`

This controls:
- checkpoint paths
- CSV paths
- threshold + fusion weight
- batch size / num workers
- output CSV path

---

## Inference (Late Fusion)

Main entrypoint:

```bash
python -m src.inference.predict --config configs/inference/ensemble.yaml
```

Notes:
- It writes predictions to the `output_csv` configured in the YAML (default: `outputs/inference/predictions_late_fusion.csv`).
- If `gt_csv` is provided and parseable, it prints an accuracy metric.

Common overrides:

```bash
python -m src.inference.predict --config configs/inference/ensemble.yaml --threshold 0.435 --facenet-weight 0.69
```

---

## Training

All training scripts support:
- `--model-config <path>` (defaults to the relevant file under `configs/`)
- `--data-config <path>` (defaults to `configs/data/retail_faces.yaml`)
- `--dry-run` (validates configs + CSV + paths, then exits **without training**)

### FaceNet Stage 1

Dry-run (recommended first):

```bash
python -m src.training.facenet_stage1 --dry-run
```

Start training:

```bash
python -m src.training.facenet_stage1
```

Config: `configs/facenet/stage1.yaml`

### FaceNet Stage 2 (Resume)

This stage continues training from a resume checkpoint.

Dry-run (checks resume checkpoint exists):

```bash
python -m src.training.facenet_stage2 --dry-run
```

Start training:

```bash
python -m src.training.facenet_stage2
```

Config: `configs/facenet/stage2.yaml`


### ViT Stage 1 (5-fold)

ViT training uses:
- Cross entropy (label smoothing)
- CurricularFace-style logits
- Metric-learning losses (triplet + contrastive)

Dependencies (already listed in `requirements.txt`):
- `torchmetrics`
- `pytorch-metric-learning`

Dry-run:

```bash
python -m src.training.vit_stage1 --dry-run
```

Start training:

```bash
python -m src.training.vit_stage1
```

Config: `configs/vit/stage1.yaml`

Outputs: one best checkpoint per fold under `outputs/vit/stage1_runs/`.

### ViT Stage 2 (Aggressive Finetune)

Dry-run (validates fold checkpoints exist):

```bash
python -m src.training.vit_stage2 --dry-run
```

Start training:

```bash
python -m src.training.vit_stage2
```

Config: `configs/vit/stage2.yaml`


---

## Outputs

- `outputs/facenet/stage2/` — legacy FaceNet checkpoint(s) used for inference
- `outputs/vit/stage2/` — legacy ViT fold checkpoints used for inference
- `outputs/*/*_runs/` — new training run outputs (safe, non-destructive)
- `outputs/inference/predictions_late_fusion.csv` — inference predictions

The training code uses “unique filename if exists” behavior when saving best checkpoints, so an existing file will never be overwritten.

---

