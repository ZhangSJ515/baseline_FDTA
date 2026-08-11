#!/usr/bin/env bash
set -euo pipefail

SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-./pretrains/r50_deformable_detr_coco.pth}"
OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-./pretrains/r50_deformable_detr_coco_uadetrac.pth}"

python tools/convert_uadetrac_detr_pretrain.py \
  --input "${SOURCE_CHECKPOINT}" \
  --output "${OUTPUT_CHECKPOINT}" \
  --overwrite
