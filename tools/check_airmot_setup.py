#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_NAMES = {
    0: "airplane",
    1: "person",
    2: "baggage_tug",
    3: "follow_me_vehicle",
}


def split_fields(line):
    return [field for field in re.split(r"[,\s]+", line.strip()) if field]


def numeric_images(image_dir: Path):
    records = []
    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            try:
                records.append((int(path.stem), path))
            except ValueError:
                pass
    return sorted(records)


def resolve_depth(image_path: Path):
    base = image_path.parent.parent / "depth" / image_path.stem
    for extension in [".png", image_path.suffix.lower(), ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
        candidate = base.with_suffix(extension)
        if candidate.is_file():
            return candidate
    return None


def inspect_sequence(sequence_dir: Path, require_depth: bool, check_depth_size: bool):
    errors = []
    image_records = numeric_images(sequence_dir / "img1")
    if not image_records:
        return None, [f"No numeric images in {sequence_dir / 'img1'}"]

    frame_ids = {frame_id for frame_id, _ in image_records}
    if len(frame_ids) != len(image_records):
        errors.append("Duplicate numeric image frame IDs")

    missing_depth = 0
    size_mismatch = 0
    for _, image_path in image_records:
        depth_path = resolve_depth(image_path)
        if depth_path is None:
            missing_depth += 1
            continue
        if check_depth_size:
            with Image.open(image_path) as image, Image.open(depth_path) as depth:
                if image.size != depth.size:
                    size_mismatch += 1

    gt_path = sequence_dir / "gt" / "gt.txt"
    if not gt_path.is_file():
        return None, errors + [f"Missing GT file: {gt_path}"]

    class_counts = Counter()
    frame_object_counts = Counter()
    track_ids = set()
    invalid_rows = 0
    missing_image_annotations = 0
    with gt_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            fields = split_fields(line)
            if len(fields) < 8:
                invalid_rows += 1
                continue
            try:
                frame_id = int(float(fields[0]))
                track_id = int(float(fields[1]))
                class_id = int(float(fields[2]))
                x, y, width, height = map(float, fields[3:7])
                float(fields[7])
            except ValueError:
                invalid_rows += 1
                continue
            if frame_id not in frame_ids:
                missing_image_annotations += 1
            if class_id not in CLASS_NAMES or track_id < 0 or width <= 0 or height <= 0:
                invalid_rows += 1
                continue
            class_counts[class_id] += 1
            frame_object_counts[frame_id] += 1
            track_ids.add(track_id)

    if require_depth and missing_depth:
        errors.append(f"Missing {missing_depth} depth maps")
    if size_mismatch:
        errors.append(f"RGB/depth size mismatch in {size_mismatch} frames")
    if invalid_rows:
        errors.append(f"Invalid GT rows: {invalid_rows}")
    if missing_image_annotations:
        errors.append(
            f"GT rows referring to missing images: {missing_image_annotations}"
        )

    annotated_frames = set(frame_object_counts)
    empty_frames = len(frame_ids - annotated_frames)
    stats = {
        "frames": len(image_records),
        "boxes": sum(class_counts.values()),
        "tracks": len(track_ids),
        "empty_frames": empty_frames,
        "missing_depth": missing_depth,
        "max_objects": max(frame_object_counts.values(), default=0),
        "class_counts": class_counts,
    }
    return stats, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/zsj/data/datasets")
    parser.add_argument("--dataset", default="AirMot")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--require-depth", action="store_true")
    parser.add_argument("--check-depth-size", action="store_true")
    parser.add_argument("--pretrain", default=None)
    args = parser.parse_args()

    dataset_dir = Path(args.data_root) / args.dataset
    if not dataset_dir.is_dir() and args.dataset == "AirMot":
        alternate = Path(args.data_root) / "AirMOT"
        if alternate.is_dir():
            dataset_dir = alternate

    if not dataset_dir.is_dir():
        print(f"[ERROR] AirMOT dataset directory does not exist: {dataset_dir}")
        return 1

    all_errors = []
    for split in args.splits:
        split_dir = dataset_dir / split
        if not split_dir.is_dir():
            all_errors.append(f"Missing split directory: {split_dir}")
            continue

        sequences = sorted(
            path for path in split_dir.iterdir()
            if path.is_dir() and (path / "img1").is_dir()
        )
        if not sequences:
            all_errors.append(f"No sequences in {split_dir}")
            continue

        split_totals = Counter()
        split_class_counts = Counter()
        print(f"\n[{split}] {len(sequences)} sequences")
        for sequence_dir in sequences:
            stats, errors = inspect_sequence(
                sequence_dir,
                require_depth=args.require_depth,
                check_depth_size=args.check_depth_size,
            )
            if stats is not None:
                for key in ["frames", "boxes", "tracks", "empty_frames", "missing_depth"]:
                    split_totals[key] += stats[key]
                split_totals["max_objects"] = max(
                    split_totals["max_objects"], stats["max_objects"]
                )
                split_class_counts.update(stats["class_counts"])
            for error in errors:
                all_errors.append(f"{split}/{sequence_dir.name}: {error}")

        print(
            f"  frames={split_totals['frames']}, boxes={split_totals['boxes']}, "
            f"tracks(sum per sequence)={split_totals['tracks']}, "
            f"empty_frames={split_totals['empty_frames']}, "
            f"missing_depth={split_totals['missing_depth']}, "
            f"max_objects_per_frame={split_totals['max_objects']}"
        )
        for class_id, class_name in CLASS_NAMES.items():
            print(f"  class {class_id} ({class_name}): {split_class_counts[class_id]}")

    if args.pretrain is not None and not Path(args.pretrain).is_file():
        all_errors.append(f"AirMOT DETR pretrain is missing: {args.pretrain}")

    if all_errors:
        print("\nValidation errors:")
        for error in all_errors:
            print(f"  [ERROR] {error}")
        return 1

    print("\nAirMOT setup validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
