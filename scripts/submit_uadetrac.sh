#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-8}"
DATA_ROOT="${DATA_ROOT:-/home/zsj/data/datasets}"
OUTPUTS_DIR="${OUTPUTS_DIR:-./outputs/fdta_uadetrac}"
CONFIG_PATH="${CONFIG_PATH:-./configs/uadetrac.yaml}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a trained FDTA checkpoint}"
SPLIT="${SPLIT:-test}"

accelerate launch --num_processes="${NUM_GPUS}" submit_and_evaluate_uadetrac.py \
  --config-path "${CONFIG_PATH}" \
  --data-root "${DATA_ROOT}" \
  --outputs-dir "${OUTPUTS_DIR}" \
  --inference-mode submit \
  --inference-dataset UA-DETRAC \
  --inference-split "${SPLIT}" \
  --inference-model "${MODEL_PATH}"
