import copy
import os
from pathlib import Path

import torch
from collections import defaultdict
from torch.utils.data import Dataset
from PIL import Image

from .dancetrack import DanceTrack
from .sportsmot import SportsMOT
from .crowdhuman import CrowdHuman
from .bft import BFT
from .airmot import AirMot


dataset_classes = {
    "DanceTrack": DanceTrack,
    "SportsMOT": SportsMOT,
    "CrowdHuman": CrowdHuman,
    "BFT": BFT,
    "AirMot": AirMot,
    # Alias retained because existing AirMOT copies use both spellings.
    "AirMOT": AirMot,
}


_DEPTH_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def resolve_depth_path(image_path: str) -> str:
    """Resolve a depth image corresponding to an RGB frame.

    The standard layout is ``img1/<stem>.<rgb_ext>`` and
    ``depth/<stem>.png``. The resolver also accepts other common image
    extensions and therefore works with AirMOT sequences whose frame naming
    or RGB extension differs from DanceTrack.
    """
    image = Path(image_path)
    parts = list(image.parts)
    try:
        img1_index = len(parts) - 1 - parts[::-1].index("img1")
    except ValueError as exc:
        raise ValueError(f"Image path does not contain an 'img1' directory: {image_path}") from exc

    parts[img1_index] = "depth"
    depth_base = Path(*parts).with_suffix("")
    candidates = [depth_base.with_suffix(ext) for ext in _DEPTH_EXTENSIONS]
    # Prefer the same extension after the canonical PNG candidate.
    same_extension = depth_base.with_suffix(image.suffix.lower())
    if same_extension not in candidates:
        candidates.insert(1, same_extension)

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    candidate_text = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"No depth map found for RGB frame '{image_path}'. Checked: {candidate_text}. "
        "Generate depth maps before training FDTA."
    )


class JointDataset(Dataset):
    def __init__(
            self,
            data_root: str,
            datasets: list,
            splits: list,
            transforms=None,
            **kwargs,
    ):
        """
        Args:
            data_root: The root directory of datasets.
            datasets: The list of dataset names, e.g., ["DanceTrack", "AirMot"].
            splits: The list of dataset split names, e.g., ["train", "train"].
        """
        super().__init__()
        assert len(datasets) == len(splits), "The number of datasets and splits should be the same."
        self.transforms = transforms

        self.size_divisibility = kwargs.get("size_divisibility", 0)

        self.sequence_infos = defaultdict(lambda: defaultdict(dict))
        self.image_paths = defaultdict(lambda: defaultdict(dict))
        self.annotations = defaultdict(lambda: defaultdict(dict))
        for dataset, split in zip(datasets, splits):
            if dataset not in dataset_classes:
                raise AttributeError(f"Dataset {dataset} is not supported.")
            dataset_class = dataset_classes[dataset](
                data_root=data_root,
                split=split,
                load_annotation=True,
            )
            self.sequence_infos[dataset][split] = dataset_class.get_sequence_infos()
            self.image_paths[dataset][split] = dataset_class.get_image_paths()
            self.annotations[dataset][split] = dataset_class.get_annotations()

        # Decouple the 'is_legal' attribute from annotations so the sampler can
        # test complete clips without mutating the actual annotations.
        self.ann_is_legals = self._decouple_is_legal()

        self.sample_begins: list | None = None

    def _decouple_is_legal(self):
        decoupled_is_legal = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for dataset in self.annotations:
            for split in self.annotations[dataset]:
                for sequence_name in self.annotations[dataset][split]:
                    for annotation in self.annotations[dataset][split][sequence_name]:
                        decoupled_is_legal[dataset][split][sequence_name].append(annotation["is_legal"])

        decoupled_is_legal_in_tensor = defaultdict(
            lambda: defaultdict(lambda: defaultdict(torch.Tensor))
        )
        for dataset in decoupled_is_legal:
            for split in decoupled_is_legal[dataset]:
                for sequence_name in decoupled_is_legal[dataset][split]:
                    decoupled_is_legal_in_tensor[dataset][split][sequence_name] = torch.tensor(
                        decoupled_is_legal[dataset][split][sequence_name], dtype=torch.bool
                    )
        return decoupled_is_legal_in_tensor

    def set_sample_details(
            self,
            sample_length: int,
            sample_interval: int,
            sample_mode: str = "random_interval",
    ):
        assert sample_mode in ["random_interval"], f"Sample mode '{sample_mode}' is not supported."
        self.sample_begins = []
        for dataset in self.annotations:
            for split in self.annotations[dataset]:
                for sequence_name in self.annotations[dataset][split]:
                    sequence_length = self.sequence_infos[dataset][split][sequence_name]["length"]
                    for frame_id in range(sequence_length):
                        if self.sequence_infos[dataset][split][sequence_name]["is_static"]:
                            self.sample_begins.append((dataset, split, sequence_name, frame_id))
                        elif frame_id + sample_length <= sequence_length:
                            clip_legality = self.ann_is_legals[dataset][split][sequence_name][
                                frame_id:frame_id + sample_length
                            ]
                            if clip_legality.all():
                                self.sample_begins.append((dataset, split, sequence_name, frame_id))

    def __len__(self):
        assert self.sample_begins is not None, "Please use 'self.set_sample_details()' at the start of each epoch."
        return len(self.sample_begins)

    def __getitem__(self, info):
        dataset = info["dataset"]
        split = info["split"]
        sequence = info["sequence"]
        frame_idxs = info["frame_idxs"]

        image_paths = [
            self.image_paths[dataset][split][sequence][frame_idx]
            for frame_idx in frame_idxs
        ]
        depth_paths = [resolve_depth_path(image_path) for image_path in image_paths]

        images = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                images.append(image.convert("RGB"))

        depthmaps = []
        for depth_path in depth_paths:
            with Image.open(depth_path) as depthmap:
                depthmaps.append(depthmap.copy())

        annotations = [
            self.annotations[dataset][split][sequence][frame_idx]
            for frame_idx in frame_idxs
        ]
        metas = [
            {
                "dataset": dataset,
                "split": split,
                "sequence": sequence,
                "frame_idx": frame_idx,
                "is_static": self.sequence_infos[dataset][split][sequence]["is_static"],
                "is_begin": False,
                "size_divisibility": self.size_divisibility,
            }
            for frame_idx in frame_idxs
        ]
        metas[0]["is_begin"] = True

        annotations = [copy.deepcopy(annotation) for annotation in annotations]
        metas = [copy.deepcopy(meta) for meta in metas]

        if self.transforms is not None:
            images, depthmaps, annotations, metas = self.transforms(
                images, depthmaps, annotations, metas
            )
        return images, depthmaps, annotations, metas

    def statistics(self):
        statistics = []
        for dataset in self.sequence_infos:
            for split in self.sequence_infos[dataset]:
                num_sequences = len(self.sequence_infos[dataset][split])
                num_frames = sum(
                    info["length"]
                    for info in self.sequence_infos[dataset][split].values()
                )
                statistics.append(
                    f"{dataset}.{split}, {num_sequences} sequences, {num_frames} frames."
                )
        return statistics
