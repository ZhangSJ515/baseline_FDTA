#!/usr/bin/env bash
set -euo pipefail

IMAGES_ROOT="${IMAGES_ROOT:?Set IMAGES_ROOT to the official UA-DETRAC image root}"
ANNOTATIONS_ROOT="${ANNOTATIONS_ROOT:?Set ANNOTATIONS_ROOT to the UA-DETRAC XML directory}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/zsj/data/datasets/UA-DETRAC}"
SPLIT="${SPLIT:-train}"
FPS="${FPS:-25}"
LINK_MODE="${LINK_MODE:-symlink}"

python tools/convert_uadetrac_xml_to_mot.py \
  --images-root "${IMAGES_ROOT}" \
  --annotations-root "${ANNOTATIONS_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --split "${SPLIT}" \
  --fps "${FPS}" \
  --link-mode "${LINK_MODE}"
