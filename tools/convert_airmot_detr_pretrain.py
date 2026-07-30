#!/usr/bin/env python3
"""Convert a COCO Deformable-DETR checkpoint to an AirMOT-compatible head.

FDTA's original loader only accepts a one-class head or a head whose class
count already equals the target class count. AirMOT has four classes. This
utility preserves all pretrained detector parameters, copies the semantically
matching COCO rows for ``airplane`` and ``person``, and initializes the two
airport-specific vehicle classes from the mean foreground classifier weight.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict

import torch


AIRMOT_CLASS_NAMES = [
    "airplane",
    "person",
    "baggage_tug",
    "follow_me_vehicle",
]

# Source row indices for common COCO classifier layouts.
# COCO 91-category ID layout: person=1, airplane=5.
COCO_91_TO_AIRMOT = [5, 1, -1, -1]
# Contiguous COCO-80 layout: person=0, airplane=4.
COCO_80_TO_AIRMOT = [4, 0, -1, -1]


def _target_rows(source_rows: int):
    if source_rows == 91:
        return COCO_91_TO_AIRMOT
    if source_rows == 80:
        return COCO_80_TO_AIRMOT
    if source_rows == 4:
        return list(range(4))
    raise ValueError(
        f"Unsupported classifier size {source_rows}. Expected 91, 80, or 4 rows."
    )


def _convert_weight(weight: torch.Tensor) -> torch.Tensor:
    mapping = _target_rows(weight.shape[0])
    if weight.shape[0] == 4:
        return weight.clone()

    # A mean foreground vector is a stable neutral initialization for classes
    # without a direct COCO counterpart. The mapped rows retain exact COCO
    # initialization for airplane and person.
    target = weight.mean(dim=0, keepdim=True).repeat(4, *([1] * (weight.ndim - 1)))
    for target_index, source_index in enumerate(mapping):
        if source_index >= 0:
            target[target_index] = weight[source_index]
    return target


def _convert_bias(bias: torch.Tensor, prior_prob: float) -> torch.Tensor:
    mapping = _target_rows(bias.shape[0])
    if bias.shape[0] == 4:
        return bias.clone()

    prior_bias = -math.log((1.0 - prior_prob) / prior_prob)
    target = torch.full(
        (4,),
        fill_value=prior_bias,
        dtype=bias.dtype,
        device=bias.device,
    )
    for target_index, source_index in enumerate(mapping):
        if source_index >= 0:
            target[target_index] = bias[source_index]
    return target


def convert_state_dict(state_dict: Dict[str, torch.Tensor], prior_prob: float):
    converted = dict(state_dict)
    converted_keys = []

    for key, value in state_dict.items():
        if "class_embed" not in key:
            continue
        if key.endswith(".weight"):
            converted[key] = _convert_weight(value)
            converted_keys.append(key)
        elif key.endswith(".bias"):
            converted[key] = _convert_bias(value, prior_prob=prior_prob)
            converted_keys.append(key)

    if not converted_keys:
        raise KeyError(
            "No class_embed weight or bias was found in the checkpoint model state."
        )
    return converted, converted_keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input COCO DETR checkpoint")
    parser.add_argument("--output", required=True, help="Output AirMOT checkpoint")
    parser.add_argument("--prior-prob", type=float, default=0.01)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output checkpoint.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )

    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        checkpoint = {"model": state_dict}
    else:
        raise TypeError(f"Unsupported checkpoint object type: {type(checkpoint)}")

    converted_state, converted_keys = convert_state_dict(
        state_dict,
        prior_prob=args.prior_prob,
    )
    checkpoint["model"] = converted_state
    checkpoint["airmot_conversion"] = {
        "classes": AIRMOT_CLASS_NAMES,
        "source_checkpoint": str(input_path),
        "converted_keys": converted_keys,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)

    print(f"Saved AirMOT-compatible DETR checkpoint to: {output_path}")
    print("Class mapping: airplane <- COCO airplane, person <- COCO person")
    print("Airport-specific classes use the mean foreground classifier initialization.")
    print(f"Converted {len(converted_keys)} classifier tensors.")


if __name__ == "__main__":
    main()
