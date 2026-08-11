#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/zsj/data/datasets}"
SPLITS="${SPLITS:-train test}"
ENCODER="${ENCODER:-vitl}"
INPUT_SIZE="${INPUT_SIZE:-518}"
FPS="${FPS:-25}"

for SPLIT in ${SPLITS}; do
  python tools/gen_depthmaps.py \
    --data-root "${DATA_ROOT}" \
    --dataset UA-DETRAC \
    --split "${SPLIT}" \
    --encoder "${ENCODER}" \
    --input-size "${INPUT_SIZE}" \
    --fps "${FPS}" \
    --grayscale
done
