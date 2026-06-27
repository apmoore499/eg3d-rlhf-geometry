#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CREATE_DIR="$(cd "$SCRIPT_DIR/../data/create_train_data" && pwd)"
PYTHON_BIN="${PYTHON:-python}"

RUN_RGB="${RUN_RGB:-1}"
RUN_DMAP="${RUN_DMAP:-1}"
RUN_SIGMA256="${RUN_SIGMA256:-1}"
RUN_LANDMARKS="${RUN_LANDMARKS:-1}"

echo "Generating maintained reward-model training inputs."
echo "Directory: $CREATE_DIR"
echo "Python: $PYTHON_BIN"
echo
echo "Expected prerequisites:"
echo "- CUDA-capable environment with the EG3D stack installed"
echo "- external pretrained EG3D checkpoint available at the configured path"
echo "- writable output directory via E3D_RLHF_SAVE_DIR / E3D_RLHF_SIGMA_DATA_DIR"
echo
echo "This can take substantial time and storage. Existing outputs are reused"
echo "where the underlying synthesis scripts already support skipping."
echo

cd "$CREATE_DIR"

if [[ "$RUN_RGB" == "1" ]]; then
  echo "[1/4] triple RGB views"
  "$PYTHON_BIN" synthesize_triple_rgb.py
fi

if [[ "$RUN_DMAP" == "1" ]]; then
  echo "[2/4] triple depth maps"
  "$PYTHON_BIN" synthesize_triple_dmap.py
fi

if [[ "$RUN_SIGMA256" == "1" ]]; then
  echo "[3/4] sigma-field 256 slabs"
  "$PYTHON_BIN" synthesize_sigma_field_256_combined.py
fi

if [[ "$RUN_LANDMARKS" == "1" ]]; then
  echo "[4/4] AW98 landmarks"
  "$PYTHON_BIN" synthesize_landmarks.py --views 0 1 2
fi

echo
echo "Reward-model data generation complete."
