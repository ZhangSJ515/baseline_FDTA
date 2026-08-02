from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image

from .one_dataset import OneDataset
from .util import append_annotation, is_legal


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_DATASET_ALIASES = ("UA-DETRAC", "UADETRAC", "UA_DETRAC")


def _split_fields(line: str) -> List[str]:
    return [field for field in re.split(r"[,\s]+", line.strip()) if field]


def _natural_key(name: str):
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", name)
    ]


def _is_small_integer(value: str) -> bool:
    try:
        number = float(value)
    except ValueError:
        return False
    return number.is_integer() and -1 <= int(number) <= 20


def _parse_gt_row(fields: List[str], gt_format: str):
    """Parse common MOTChallenge and converted UA-DETRAC rows.

    Supported layouts are:

    - ``mot``: frame,id,x,y,w,h,mark,class,visibility[,unused]
    - ``native8``: frame,id,class,x,y,w,h,flag
    - ``simple``: frame,id,x,y,w,h[,mark]

    ``auto`` selects ``mot`` for rows with at least nine columns. For an
    eight-column row it treats the third field as a class only when that field
    is a small integer; otherwise it treats the row as shortened MOT format.
    """
    if gt_format not in {"auto", "mot", "native8", "simple"}:
        raise ValueError(f"Unsupported UA-DETRAC GT format: {gt_format}")

    layout = gt_format
    if layout == "auto":
        if len(fields) >= 9:
            layout = "mot"
        elif len(fields) == 8 and _is_small_integer(fields[2]):
            layout = "native8"
        else:
            layout = "simple"

    if layout == "mot":
        if len(fields) < 8:
            raise ValueError("MOT-format rows require at least 8 columns")
        frame_id = int(float(fields[0]))
        obj_id = int(float(fields[1]))
        x, y, width, height = map(float, fields[2:6])
        mark = float(fields[6])
        visibility = float(fields[8]) if len(fields) >= 9 else 1.0
    elif layout == "native8":
        if len(fields) < 8:
            raise ValueError("native8 rows require 8 columns")
        frame_id = int(float(fields[0]))
        obj_id = int(float(fields[1]))
        x, y, width, height = map(float, fields[3:7])
        mark = float(fields[7])
        visibility = max(0.0, mark)
    else:
        if len(fields) < 6:
            raise ValueError("simple rows require at least 6 columns")
        frame_id = int(float(fields[0]))
        obj_id = int(float(fields[1]))
        x, y, width, height = map(float, fields[2:6])
        mark = float(fields[6]) if len(fields) >= 7 else 1.0
        visibility = 1.0

    return frame_id, obj_id, x, y, width, height, mark, visibility


class UADETRAC(OneDataset):
    """UA-DETRAC adapter for FDTA.

    Expected converted layout::

        <data_root>/UA-DETRAC/<split>/<sequence>/
            img1/<numeric frame image>
            depth/<numeric depth image>
            gt/gt.txt
            seqinfo.ini                 # optional

    The adapter is single-class and accepts both standard MOTChallenge rows
    and the compact formats documented in :func:`_parse_gt_row`.
    """

    class_names = {0: "vehicle"}

    def __init__(
            self,
            data_root: str = "./datasets/",
            sub_dir: str = "UA-DETRAC",
            split: str = "train",
            load_annotation: bool = True,
            filter_gt_by_mark: bool = True,
            min_visibility: float = 0.0,
            gt_format: str = "auto",
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported UA-DETRAC split: {split}")

        root = Path(data_root)
        requested = root / sub_dir
        if not requested.is_dir():
            for alias in _DATASET_ALIASES:
                candidate = root / alias
                if candidate.is_dir():
                    sub_dir = alias
                    break

        super().__init__(
            data_root=data_root,
            sub_dir=sub_dir,
            split=split,
            load_annotation=load_annotation,
        )
        self.filter_gt_by_mark = bool(filter_gt_by_mark)
        self.min_visibility = float(min_visibility)
        self.gt_format = gt_format

        split_dir = Path(self.data_dir) / self.split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"UA-DETRAC split directory does not exist: {split_dir}"
            )

        self._sequence_image_records: Dict[str, List[Tuple[int, str]]] = {}
        self._frame_id_to_index: Dict[str, Dict[int, int]] = {}

        self.sequence_infos = self._get_sequence_infos()
        self.image_paths = self._get_image_paths()
        if self.load_annotation:
            self.annotations = self._get_annotations()

    def _get_sequence_names(self) -> List[str]:
        split_dir = Path(self.data_dir) / self.split
        names = [
            path.name
            for path in split_dir.iterdir()
            if path.is_dir() and (path / "img1").is_dir()
        ]
        return sorted(names, key=_natural_key)

    @staticmethod
    def _scan_sequence_images(sequence_dir: Path) -> List[Tuple[int, str]]:
        records: List[Tuple[int, str]] = []
        for image_path in (sequence_dir / "img1").iterdir():
            if (
                not image_path.is_file()
                or image_path.suffix.lower() not in _IMAGE_EXTENSIONS
            ):
                continue
            try:
                frame_id = int(image_path.stem)
            except ValueError:
                continue
            records.append((frame_id, str(image_path)))
        records.sort(key=lambda item: item[0])
        return records

    def _get_sequence_infos(self):
        sequence_infos = {}
        for sequence_name in self._get_sequence_names():
            sequence_dir = Path(self.data_dir) / self.split / sequence_name
            image_records = self._scan_sequence_images(sequence_dir)
            if not image_records:
                raise RuntimeError(
                    f"No numeric frame images found in {sequence_dir / 'img1'}"
                )

            frame_ids = [frame_id for frame_id, _ in image_records]
            if len(frame_ids) != len(set(frame_ids)):
                raise ValueError(
                    f"Duplicate frame IDs found in sequence {sequence_name}"
                )

            with Image.open(image_records[0][1]) as image:
                width, height = image.size

            self._sequence_image_records[sequence_name] = image_records
            self._frame_id_to_index[sequence_name] = {
                frame_id: index for index, frame_id in enumerate(frame_ids)
            }
            sequence_infos[sequence_name] = {
                "width": width,
                "height": height,
                "length": len(image_records),
                "is_static": False,
                "frame_ids": frame_ids,
            }

        if not sequence_infos:
            raise RuntimeError(
                f"No valid UA-DETRAC sequences found under "
                f"{Path(self.data_dir) / self.split}"
            )
        return sequence_infos

    def _get_image_paths(self):
        image_paths = defaultdict(list)
        for sequence_name, records in self._sequence_image_records.items():
            image_paths[sequence_name] = [path for _, path in records]
        return image_paths

    def _init_annotations(self, sequence_names):
        annotations = {}
        for sequence_name in sequence_names:
            annotations[sequence_name] = []
            for _ in range(self.sequence_infos[sequence_name]["length"]):
                annotations[sequence_name].append({
                    "id": torch.zeros((0,), dtype=torch.int64),
                    "category": torch.zeros((0,), dtype=torch.int64),
                    "bbox": torch.zeros((0, 4), dtype=torch.float32),
                    "visibility": torch.zeros((0,), dtype=torch.float32),
                })
        return annotations

    def _get_annotations(self):
        sequence_names = self._get_sequence_names()
        annotations = self._init_annotations(sequence_names)

        for sequence_name in sequence_names:
            gt_path = (
                Path(self.data_dir)
                / self.split
                / sequence_name
                / "gt"
                / "gt.txt"
            )
            if not gt_path.is_file():
                raise FileNotFoundError(
                    f"UA-DETRAC annotation file does not exist: {gt_path}"
                )

            frame_to_index = self._frame_id_to_index[sequence_name]
            with gt_path.open("r", encoding="utf-8") as gt_file:
                for line_number, line in enumerate(gt_file, start=1):
                    if not line.strip():
                        continue
                    fields = _split_fields(line)
                    try:
                        (
                            frame_id,
                            obj_id,
                            x,
                            y,
                            width,
                            height,
                            mark,
                            visibility,
                        ) = _parse_gt_row(fields, self.gt_format)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Invalid UA-DETRAC annotation at {gt_path}:"
                            f"{line_number}: {line.strip()}"
                        ) from exc

                    if frame_id not in frame_to_index:
                        raise ValueError(
                            f"Annotation frame {frame_id} has no matching image "
                            f"in sequence {sequence_name}"
                        )
                    if self.filter_gt_by_mark and mark <= 0:
                        continue
                    if visibility < self.min_visibility:
                        continue
                    if obj_id < 0 or width <= 0 or height <= 0:
                        continue

                    ann_index = frame_to_index[frame_id]
                    annotations[sequence_name][ann_index] = append_annotation(
                        annotation=annotations[sequence_name][ann_index],
                        obj_id=obj_id,
                        category=0,
                        bbox=[x, y, width, height],
                        visibility=max(0.0, visibility),
                    )

        for sequence_name in sequence_names:
            for annotation in annotations[sequence_name]:
                annotation["is_legal"] = is_legal(annotation)

        return annotations
