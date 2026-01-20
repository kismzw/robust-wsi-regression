#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/base.yaml}"
CKPT="${2:?Usage: $0 <config> <path/to/last.pt>}"

python train.py \
  --config "$CONFIG" \
  --resume "$CKPT"
