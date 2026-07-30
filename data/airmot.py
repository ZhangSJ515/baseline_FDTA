from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image

from .one_dataset import OneDataset
from .util import append_annotation, is_legal


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_AIRMOT_CLASSES = {
    0: "airplane",
    1: "person",
    2: "baggage_tug",
    3: "follow_me_vehicle",
}


def _split_fields(line: str) -> List[str]:
    """Split either comma-separated or whitespace-separated MOT records."""
    return [field for field in re.split(r"[,\s]+", line.strip()) if field]


def _natural_key(name: str):
    return [int(token) if token.isdigit() else token.lower()
            for token in re.split(r"(\d+)", name)]


class AirMot(OneDataset):
    """AirMOT adapter used by FDTA.

    Expected directory layout::

        <data_root>/AirMot/<split>/<sequence>/
            img1/<frame image>
            depth/<depth image>
            gt/gt.txt

    The native annotation format is::

        frame_id track_id class_id x y width height flag

    where class IDs are 0=airplane, 1=person, 2=baggage_tug and
    3=follow_me_vehicle. Frame image names may use any numeric zero-padding.
    """

    class_names = _AIRMOT_CLASSES

    def __init__(
            self,
            data_root: str = "./datasets/",
            sub_dir: str = "AirMot",
            split: str = "train",
            load_annotation: bool = True,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported AirMOT split: {split}")

        # Be tolerant to the two common capitalizations used in local copies.
        requested = Path(data_root) / sub_dir
        if not requested.is_dir() and (Path(data_root) / "AirMOT").is_dir():
            sub_dir = "AirMOT"

        super().__init__(
            data_root=data_root,
            sub_dir=sub_dir,
            split=split,
            load_annotation=load_annotation,
        )

        split_dir = Path(self.data_dir) / self.split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"AirMOT split directory does not exist: {split_dir}")

        self._sequence_image_records: Dict[str, List[Tuple[int, str]]] = {}
        self._frame_id_to_index: Dict[str, Dict[int, int]] = {}

        self.sequence_infos = self._get_sequence_infos()
        self.image_paths = self._get_image_paths()
        if self.load_annotation:
            self.annotations = self._get_annotations()

    def _get_sequence_names(self) -> List[str]:
        split_dir = Path(self.data_dir) / self.split
        names = [
            path.name for path in split_dir.iterdir()
            if path.is_dir() and (path / "img1").is_dir()
        ]
        return sorted(names, key=_natural_key)

    @staticmethod
    def _scan_sequence_images(sequence_dir: Path) -> List[Tuple[int, str]]:
        records: List[Tuple[int, str]] = []
        for image_path in (sequence_dir / "img1").iterdir():
            if not image_path.is_file() or image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
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
                raise RuntimeError(f"No numeric frame images found in {sequence_dir / 'img1'}")

            frame_ids = [frame_id for frame_id, _ in image_records]
            if len(frame_ids) != len(set(frame_ids)):
                raise ValueError(f"Duplicate frame IDs found in sequence {sequence_name}")

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
                # Retained for native AirMOT result serialization.
                "frame_ids": frame_ids,
            }
        if not sequence_infos:
            raise RuntimeError(
                f"No valid AirMOT sequences found under {Path(self.data_dir) / self.split}"
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
            gt_path = Path(self.data_dir) / self.split / sequence_name / "gt" / "gt.txt"
            if not gt_path.is_file():
                raise FileNotFoundError(f"AirMOT annotation file does not exist: {gt_path}")

            frame_to_index = self._frame_id_to_index[sequence_name]
            with gt_path.open("r", encoding="utf-8") as gt_file:
                for line_number, line in enumerate(gt_file, start=1):
                    if not line.strip():
                        continue
                    fields = _split_fields(line)
                    if len(fields) < 8:
                        raise ValueError(
                            f"Invalid AirMOT annotation at {gt_path}:{line_number}: {line.strip()}"
                        )

                    frame_id = int(float(fields[0]))
                    obj_id = int(float(fields[1]))
                    category = int(float(fields[2]))
                    x, y, width, height = map(float, fields[3:7])
                    flag = float(fields[7])

                    if category not in _AIRMOT_CLASSES:
                        raise ValueError(
                            f"Unknown AirMOT class ID {category} at {gt_path}:{line_number}"
                        )
                    if frame_id not in frame_to_index:
                        raise ValueError(
                            f"Annotation frame {frame_id} has no matching image in sequence {sequence_name}"
                        )
                    if obj_id < 0 or width <= 0 or height <= 0:
                        continue

                    ann_index = frame_to_index[frame_id]
                    annotations[sequence_name][ann_index] = append_annotation(
                        annotation=annotations[sequence_name][ann_index],
                        obj_id=obj_id,
                        category=category,
                        bbox=[x, y, width, height],
                        visibility=max(0.0, flag),
                    )

        for sequence_name in sequence_names:
            for annotation in annotations[sequence_name]:
                annotation["is_legal"] = is_legal(annotation)

        return annotations
