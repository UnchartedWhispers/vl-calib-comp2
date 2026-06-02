#!/usr/bin/env bash
set -euo pipefail

DATASET_ID="xiaowenyi/VL-Calibration-12K"
OUT_DIR="data/modelscope/VL-Calibration-12K"

mkdir -p "$OUT_DIR"

echo "Downloading $DATASET_ID to $OUT_DIR"
modelscope download --dataset "$DATASET_ID" --local_dir "$OUT_DIR"

echo ""
echo "Downloaded files:"
find "$OUT_DIR" -maxdepth 4 -type f | sort

echo ""
echo "Done."
