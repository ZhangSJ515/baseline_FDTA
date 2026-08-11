#!/usr/bin/env python3
"""Convert a COCO Deformable-DETR checkpoint to a one-class vehicle head."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch


COCO_CAR_ROW = {
    91: 3,
    80: 2,
    1: 0,
}


def _convert_tensor(value: torch.Tensor) -> torch.Tensor:
    source_rows = value.shape[0]
    if source_rows not in COCO_CAR_ROW:
        raise ValueError(
            f"Unsupported classifier size {source_rows}. Expected 91, 80, or 1 rows."
        )
    source_index = COCO_CAR_ROW[source_rows]
    return value[source_index:source_index + 1].clone()


def convert_state_dict(state_dict: Dict[str, torch.Tensor]):
    converted = dict(state_dict)
    converted_keys = []
    for key, value in state_dict.items():
        if "class_embed" not in key:
            continue
        if key.endswith(".weight") or key.endswith(".bias"):
            converted[key] = _convert_tensor(value)
            converted_keys.append(key)

    if not converted_keys:
        raise KeyError(
            "No class_embed weight or bias was found in the checkpoint model state."
        )
    return converted, converted_keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input COCO DETR checkpoint")
    parser.add_argument("--output", required=True, help="Output UA-DETRAC checkpoint")
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

    converted_state, converted_keys = convert_state_dict(state_dict)
    checkpoint["model"] = converted_state
    checkpoint["uadetrac_conversion"] = {
        "classes": ["vehicle"],
        "source_checkpoint": str(input_path),
        "source_semantic_class": "COCO car",
        "converted_keys": converted_keys,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    print(f"Saved UA-DETRAC-compatible DETR checkpoint to: {output_path}")
    print(f"Converted {len(converted_keys)} classifier tensors from COCO car.")


if __name__ == "__main__":
    main()
