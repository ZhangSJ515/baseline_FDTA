#!/usr/bin/env python3

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "TrackEval"))

from data.joint_dataset import resolve_depth_path
from data.uadetrac import UADETRAC
import trackeval


def make_image(path: Path, size=(64, 48), mode="RGB"):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size).save(path)


def main():
    with tempfile.TemporaryDirectory(prefix="fdta_uadetrac_smoke_") as temporary:
        root = Path(temporary)
        sequence = root / "UA-DETRAC" / "test" / "MVI_00001"
        for frame_id in [1, 2, 3]:
            make_image(sequence / "img1" / f"img{frame_id:05d}.jpg")
            # The loader expects numeric stems; use canonical numeric copies.
            (sequence / "img1" / f"img{frame_id:05d}.jpg").unlink()
            make_image(sequence / "img1" / f"{frame_id:05d}.jpg")
            make_image(sequence / "depth" / f"{frame_id:05d}.png", mode="L")

        gt_path = sequence / "gt" / "gt.txt"
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        gt_path.write_text(
            "1,10,5,5,20,10,1,1,1\n"
            "2,10,6,5,20,10,1,1,1\n"
            "3,10,7,5,20,10,1,1,1\n",
            encoding="utf-8",
        )

        dataset = UADETRAC(
            data_root=str(root),
            split="test",
            load_annotation=True,
            gt_format="mot",
        )
        assert dataset.sequence_infos["MVI_00001"]["frame_ids"] == [1, 2, 3]
        assert len(dataset.image_paths["MVI_00001"]) == 3
        assert dataset.annotations["MVI_00001"][0]["category"].tolist() == [0]
        assert Path(resolve_depth_path(dataset.image_paths["MVI_00001"][0])).is_file()

        tracker_dir = root / "tracker"
        tracker_dir.mkdir()
        (tracker_dir / "MVI_00001.txt").write_text(
            "1,0,5,5,20,10,0.9,-1,-1,-1\n"
            "2,0,6,5,20,10,0.9,-1,-1,-1\n"
            "3,0,7,5,20,10,0.9,-1,-1,-1\n",
            encoding="utf-8",
        )

        eval_dataset = trackeval.datasets.UADETRAC({
            "GT_FOLDER": str(root / "UA-DETRAC" / "test"),
            "TRACKERS_FOLDER": str(tracker_dir),
            "OUTPUT_FOLDER": str(tracker_dir),
            "TRACKERS_TO_EVAL": [""],
            "CLASSES_TO_EVAL": ["vehicle"],
            "PRINT_CONFIG": False,
            "TRACKER_SUB_FOLDER": "",
            "OUTPUT_SUB_FOLDER": "",
            "FILTER_GT_BY_MARK": True,
            "MIN_VISIBILITY": 0.0,
            "GT_FORMAT": "mot",
        })
        raw = eval_dataset.get_raw_seq_data("", "MVI_00001")
        vehicle = eval_dataset.get_preprocessed_seq_data(raw, "vehicle")
        assert vehicle["num_gt_dets"] == 3
        assert vehicle["num_tracker_dets"] == 3

    print("UA-DETRAC adapter smoke test passed.")


if __name__ == "__main__":
    main()
