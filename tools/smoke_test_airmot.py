#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "TrackEval"))

from data.airmot import AirMot
from data.joint_dataset import resolve_depth_path
import trackeval


def make_image(path: Path, size=(64, 48), mode="RGB"):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size).save(path)


def main():
    with tempfile.TemporaryDirectory(prefix="fdta_airmot_smoke_") as temporary:
        root = Path(temporary)
        sequence = root / "AirMot" / "val" / "seq01"
        for frame_id in [1, 2, 3]:
            make_image(sequence / "img1" / f"{frame_id}.jpg")
            make_image(sequence / "depth" / f"{frame_id}.png", mode="L")

        gt_path = sequence / "gt" / "gt.txt"
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        gt_path.write_text(
            "1 10 0 5 5 20 10 1\n"
            "2,10,0,6,5,20,10,1\n"
            "3 10 0 7 5 20 10 1\n"
            "1 20 1 30 10 8 12 1\n"
            "2 20 1 31 10 8 12 1\n"
            "3 20 1 32 10 8 12 1\n",
            encoding="utf-8",
        )

        dataset = AirMot(
            data_root=str(root),
            split="val",
            load_annotation=True,
        )
        assert dataset.sequence_infos["seq01"]["frame_ids"] == [1, 2, 3]
        assert len(dataset.image_paths["seq01"]) == 3
        assert dataset.annotations["seq01"][0]["category"].tolist() == [0, 1]
        assert Path(resolve_depth_path(dataset.image_paths["seq01"][0])).is_file()

        tracker_dir = root / "tracker"
        tracker_dir.mkdir()
        (tracker_dir / "seq01.txt").write_text(
            "1 0 0 5 5 20 10 0.9\n"
            "2 0 0 6 5 20 10 0.9\n"
            "3 0 0 7 5 20 10 0.9\n"
            "1 1 1 30 10 8 12 0.8\n"
            "2 1 1 31 10 8 12 0.8\n"
            "3 1 1 32 10 8 12 0.8\n",
            encoding="utf-8",
        )

        eval_dataset = trackeval.datasets.AirMOT({
            "GT_FOLDER": str(root / "AirMot" / "val"),
            "TRACKERS_FOLDER": str(tracker_dir),
            "OUTPUT_FOLDER": str(tracker_dir),
            "TRACKERS_TO_EVAL": [""],
            "CLASSES_TO_EVAL": [
                "airplane",
                "person",
                "baggage_tug",
                "follow_me_vehicle",
            ],
            "PRINT_CONFIG": False,
            "TRACKER_SUB_FOLDER": "",
            "OUTPUT_SUB_FOLDER": "",
            "FILTER_GT_BY_FLAG": False,
        })
        raw = eval_dataset.get_raw_seq_data("", "seq01")
        airplane = eval_dataset.get_preprocessed_seq_data(raw, "airplane")
        person = eval_dataset.get_preprocessed_seq_data(raw, "person")
        assert airplane["num_gt_dets"] == 3
        assert airplane["num_tracker_dets"] == 3
        assert person["num_gt_dets"] == 3
        assert person["num_tracker_dets"] == 3

    print("AirMOT adapter smoke test passed.")


if __name__ == "__main__":
    main()
