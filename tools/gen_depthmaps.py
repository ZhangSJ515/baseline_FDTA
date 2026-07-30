import argparse
import glob
import os
import re
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Video_depth_anything")))
from video_depth_anything.video_depth import VideoDepthAnything


DATASETS = ["DanceTrack", "SportsMOT", "BFT", "AirMot", "AirMOT"]
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def natural_key(path):
    name = os.path.basename(path)
    return [int(token) if token.isdigit() else token.lower()
            for token in re.split(r"(\d+)", name)]


def list_images(seq_path):
    image_paths = []
    for pattern in IMAGE_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(seq_path, "img1", pattern)))
    return sorted(set(image_paths), key=natural_key)


def process_sequence(model, seq_path, input_size, device, grayscale, fps):
    img_files = list_images(seq_path)
    if not img_files:
        print(f"  [warn] no images found in {os.path.join(seq_path, 'img1')}")
        return 0

    frames = []
    valid_files = []
    for path in img_files:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"  [warn] failed to read {path}, skipped")
            continue
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        valid_files.append(path)
    if not frames:
        return 0

    depths, _ = model.infer_video_depth(
        np.asarray(frames),
        fps,
        input_size=input_size,
        device=device,
    )
    if len(depths) != len(valid_files):
        raise RuntimeError(
            f"Depth output count {len(depths)} does not match input frame count {len(valid_files)} "
            f"for sequence {seq_path}."
        )

    depth_dir = os.path.join(seq_path, "depth")
    os.makedirs(depth_dir, exist_ok=True)

    global_max = float(np.max(depths))
    if global_max <= 0:
        raise RuntimeError(f"Non-positive depth maximum for sequence {seq_path}: {global_max}")

    for source_path, depth in zip(valid_files, depths):
        normalized = np.clip(depth / global_max * 255.0, 0, 255).astype(np.uint8)
        if not grayscale:
            normalized = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
        stem = os.path.splitext(os.path.basename(source_path))[0]
        output_path = os.path.join(depth_dir, stem + ".png")
        if not cv2.imwrite(output_path, normalized):
            raise IOError(f"Failed to write depth map: {output_path}")

    return len(valid_files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--encoder", type=str, default="vitl", choices=["vits", "vitl"])
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--grayscale", action="store_true")
    args = parser.parse_args()

    dataset_dir = os.path.join(args.data_root, args.dataset)
    if not os.path.isdir(dataset_dir) and args.dataset == "AirMot":
        alternate = os.path.join(args.data_root, "AirMOT")
        if os.path.isdir(alternate):
            dataset_dir = alternate
    split_dir = os.path.join(dataset_dir, args.split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(split_dir)

    checkpoint_path = os.path.join(
        os.path.dirname(__file__),
        "../../Video_depth_anything",
        "checkpoints",
        f"video_depth_anything_{args.encoder}.pth",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = VideoDepthAnything(**MODEL_CONFIGS[args.encoder])
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=False),
        strict=True,
    )
    model = model.to(device).eval()

    sequences = sorted(
        path for path in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, path))
    )
    total = 0
    for sequence in tqdm(sequences, desc=f"{args.dataset}/{args.split}"):
        total += process_sequence(
            model=model,
            seq_path=os.path.join(split_dir, sequence),
            input_size=args.input_size,
            device=device,
            grayscale=args.grayscale,
            fps=args.fps,
        )

    print(f"Done. {len(sequences)} sequences, {total} frames -> {split_dir}")


if __name__ == "__main__":
    main()
