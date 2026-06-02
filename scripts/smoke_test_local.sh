#!/usr/bin/env bash
set -euo pipefail

python scripts/check_env.py
python scripts/make_debug_subset.py \
  --input data/raw/VL-Calibration-12K/train.jsonl \
  --output data/debug/train_32.jsonl \
  --n 32

python scripts/check_schema.py \
  --path data/debug/train_32.jsonl

python scripts/prepare_raw_for_pipeline.py \
  --input data/debug/train_32.jsonl \
  --output data/processed/train_32_pipeline.jsonl

python scripts/check_schema.py \
  --path data/processed/train_32_pipeline.jsonl

echo "Local smoke test passed."
