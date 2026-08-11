#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
DATA_ROOT="${DATA_ROOT:-/home/zsj/data/datasets}"
CONFIG_PATH="${CONFIG_PATH:-./configs/uadetrac.yaml}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a trained FDTA UA-DETRAC checkpoint}"
SPLIT="${SPLIT:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/efficiency/fdta_uadetrac}"
PRECISION="${PRECISION:-FP32}"
WARMUP_FRAMES="${WARMUP_FRAMES:-100}"
FLOPS_SAMPLES="${FLOPS_SAMPLES:-20}"
GPU_ONLY_TIMING="${GPU_ONLY_TIMING:-0}"
ALLOW_NON_L40S="${ALLOW_NON_L40S:-0}"

EXTRA_ARGS=()
if [[ "${GPU_ONLY_TIMING}" == "1" ]]; then
  EXTRA_ARGS+=(--gpu-only-timing)
fi
if [[ "${ALLOW_NON_L40S}" == "1" ]]; then
  EXTRA_ARGS+=(--allow-non-l40s)
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" python tools/analyze_efficiency.py \
  --dataset UA-DETRAC \
  --config-path "${CONFIG_PATH}" \
  --checkpoint "${MODEL_PATH}" \
  --data-root "${DATA_ROOT}" \
  --split "${SPLIT}" \
  --output-dir "${OUTPUT_DIR}" \
  --precision "${PRECISION}" \
  --warmup-frames "${WARMUP_FRAMES}" \
  --flops-samples "${FLOPS_SAMPLES}" \
  "${EXTRA_ARGS[@]}"
