#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-8}"
DATA_ROOT="${DATA_ROOT:-/home/zsj/data/datasets}"
OUTPUTS_DIR="${OUTPUTS_DIR:-./outputs/fdta_airmot}"
CONFIG_PATH="${CONFIG_PATH:-./configs/airmot.yaml}"
DETR_PRETRAIN="${DETR_PRETRAIN:-./pretrains/r50_deformable_detr_coco_airmot.pth}"

bash scripts/check_airmot_setup.sh

accelerate launch --num_processes="${NUM_GPUS}" train.py \
  --config-path "${CONFIG_PATH}" \
  --data-root "${DATA_ROOT}" \
  --outputs-dir "${OUTPUTS_DIR}" \
  --exp-name fdta_airmot \
  --detr-pretrain "${DETR_PRETRAIN}"
