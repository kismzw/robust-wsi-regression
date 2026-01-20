#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/base.yaml}"

python train.py \
  --config "$CONFIG" \
  --max_train_batches 0 \
  --max_val_batches 0
