#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/zsj/data/datasets}"
PRETRAIN="${PRETRAIN:-./pretrains/r50_deformable_detr_coco_uadetrac.pth}"
SPLITS="${SPLITS:-train test}"
GT_FORMAT="${GT_FORMAT:-auto}"
MIN_VISIBILITY="${MIN_VISIBILITY:-0.0}"

python tools/check_uadetrac_setup.py \
  --data-root "${DATA_ROOT}" \
  --splits ${SPLITS} \
  --gt-format "${GT_FORMAT}" \
  --min-visibility "${MIN_VISIBILITY}" \
  --require-depth \
  --pretrain "${PRETRAIN}"
