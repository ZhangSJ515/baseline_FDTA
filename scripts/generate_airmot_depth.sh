#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/zsj/data/datasets}"
ENCODER="${ENCODER:-vitl}"
INPUT_SIZE="${INPUT_SIZE:-518}"
FPS="${FPS:-30}"

for SPLIT in train val; do
  python tools/gen_depthmaps.py \
    --data-root "${DATA_ROOT}" \
    --dataset AirMot \
    --split "${SPLIT}" \
    --encoder "${ENCODER}" \
    --input-size "${INPUT_SIZE}" \
    --fps "${FPS}" \
    --grayscale
done
