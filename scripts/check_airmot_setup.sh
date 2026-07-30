#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/zsj/data/datasets}"
PRETRAIN="${PRETRAIN:-./pretrains/r50_deformable_detr_coco_airmot.pth}"

python tools/check_airmot_setup.py \
  --data-root "${DATA_ROOT}" \
  --dataset AirMot \
  --splits train val \
  --require-depth \
  --pretrain "${PRETRAIN}"
