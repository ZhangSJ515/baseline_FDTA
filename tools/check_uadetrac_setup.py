#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DATASET_ALIASES = ("UA-DETRAC", "UADETRAC", "UA_DETRAC")


def split_fields(line):
    return [field for field in re.split(r"[,\s]+", line.strip()) if field]


def numeric_images(image_dir: Path):
    records = []
    if not image_dir.is_dir():
        return records
    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            try:
                records.append((int(path.stem), path))
            except ValueError:
                pass
    return sorted(records)


def resolve_depth(image_path: Path):
    base = image_path.parent.parent / "depth" / image_path.stem
    for extension in [
        ".png",
        image_path.suffix.lower(),
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
    ]:
        candidate = base.with_suffix(extension)
        if candidate.is_file():
            return candidate
    return None


def is_small_integer(value):
    try:
        number = float(value)
    except ValueError:
        return False
    return number.is_integer() and -1 <= int(number) <= 20


def parse_gt_row(fields, gt_format):
    layout = gt_format
    if layout == "auto":
        if len(fields) >= 9:
            layout = "mot"
        elif len(fields) == 8 and is_small_integer(fields[2]):
            layout = "native8"
        else:
            layout = "simple"

    if layout == "mot":
        if len(fields) < 8:
            raise ValueError
        frame_id = int(float(fields[0]))
        track_id = int(float(fields[1]))
        x, y, width, height = map(float, fields[2:6])
        mark = float(fields[6])
        visibility = float(fields[8]) if len(fields) >= 9 else 1.0
    elif layout == "native8":
        if len(fields) < 8:
            raise ValueError
        frame_id = int(float(fields[0]))
        track_id = int(float(fields[1]))
        x, y, width, height = map(float, fields[3:7])
        mark = float(fields[7])
        visibility = max(0.0, mark)
    elif layout == "simple":
        if len(fields) < 6:
            raise ValueError
        frame_id = int(float(fields[0]))
        track_id = int(float(fields[1]))
        x, y, width, height = map(float, fields[2:6])
        mark = float(fields[6]) if len(fields) >= 7 else 1.0
        visibility = 1.0
    else:
        raise ValueError(f"Unsupported GT format: {layout}")

    return frame_id, track_id, x, y, width, height, mark, visibility


def inspect_sequence(
        sequence_dir: Path,
        require_depth: bool,
        check_depth_size: bool,
        filter_mark: bool,
        min_visibility: float,
        gt_format: str,
):
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

    frame_object_counts = Counter()
    track_ids = set()
    invalid_rows = 0
    filtered_rows = 0
    missing_image_annotations = 0
    with gt_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            fields = split_fields(line)
            try:
                (
                    frame_id,
                    track_id,
                    _,
                    _,
                    width,
                    height,
                    mark,
                    visibility,
                ) = parse_gt_row(fields, gt_format)
            except (TypeError, ValueError):
                invalid_rows += 1
                continue

            if frame_id not in frame_ids:
                missing_image_annotations += 1
            if track_id < 0 or width <= 0 or height <= 0:
                invalid_rows += 1
                continue
            if filter_mark and mark <= 0:
                filtered_rows += 1
                continue
            if visibility < min_visibility:
                filtered_rows += 1
                continue
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
    stats = {
        "frames": len(image_records),
        "boxes": sum(frame_object_counts.values()),
        "tracks": len(track_ids),
        "empty_frames": len(frame_ids - annotated_frames),
        "missing_depth": missing_depth,
        "filtered_rows": filtered_rows,
        "max_objects": max(frame_object_counts.values(), default=0),
    }
    return stats, errors


def resolve_dataset_dir(data_root: Path, dataset: str):
    requested = data_root / dataset
    if requested.is_dir():
        return requested
    for alias in DATASET_ALIASES:
        candidate = data_root / alias
        if candidate.is_dir():
            return candidate
    return requested


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/zsj/data/datasets")
    parser.add_argument("--dataset", default="UA-DETRAC")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--require-depth", action="store_true")
    parser.add_argument("--check-depth-size", action="store_true")
    parser.add_argument("--filter-gt-by-mark", action="store_true", default=True)
    parser.add_argument("--no-filter-gt-by-mark", action="store_false", dest="filter_gt_by_mark")
    parser.add_argument("--min-visibility", type=float, default=0.0)
    parser.add_argument(
        "--gt-format",
        choices=["auto", "mot", "native8", "simple"],
        default="auto",
    )
    parser.add_argument("--pretrain", default=None)
    args = parser.parse_args()

    dataset_dir = resolve_dataset_dir(Path(args.data_root), args.dataset)
    if not dataset_dir.is_dir():
        print(f"[ERROR] UA-DETRAC dataset directory does not exist: {dataset_dir}")
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

        totals = Counter()
        print(f"\n[{split}] {len(sequences)} sequences")
        for sequence_dir in sequences:
            stats, errors = inspect_sequence(
                sequence_dir=sequence_dir,
                require_depth=args.require_depth,
                check_depth_size=args.check_depth_size,
                filter_mark=args.filter_gt_by_mark,
                min_visibility=args.min_visibility,
                gt_format=args.gt_format,
            )
            if stats is not None:
                for key in [
                    "frames",
                    "boxes",
                    "tracks",
                    "empty_frames",
                    "missing_depth",
                    "filtered_rows",
                ]:
                    totals[key] += stats[key]
                totals["max_objects"] = max(
                    totals["max_objects"], stats["max_objects"]
                )
            for error in errors:
                all_errors.append(f"{split}/{sequence_dir.name}: {error}")

        print(
            f"  frames={totals['frames']}, boxes={totals['boxes']}, "
            f"tracks(sum per sequence)={totals['tracks']}, "
            f"empty_frames={totals['empty_frames']}, "
            f"filtered_rows={totals['filtered_rows']}, "
            f"missing_depth={totals['missing_depth']}, "
            f"max_objects_per_frame={totals['max_objects']}"
        )

    if args.pretrain is not None and not Path(args.pretrain).is_file():
        all_errors.append(f"UA-DETRAC DETR pretrain is missing: {args.pretrain}")

    if all_errors:
        print("\nValidation errors:")
        for error in all_errors:
            print(f"  [ERROR] {error}")
        return 1

    print("\nUA-DETRAC setup validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
