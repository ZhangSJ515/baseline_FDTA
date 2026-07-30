#!/usr/bin/env bash
set -euo pipefail

SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-./pretrains/r50_deformable_detr_coco.pth}"
OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-./pretrains/r50_deformable_detr_coco_airmot.pth}"

python tools/convert_airmot_detr_pretrain.py \
  --input "${SOURCE_CHECKPOINT}" \
  --output "${OUTPUT_CHECKPOINT}" \
  --overwrite

echo "Prepared AirMOT DETR pretrain: ${OUTPUT_CHECKPOINT}"
