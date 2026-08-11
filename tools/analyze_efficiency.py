#!/usr/bin/env python3
"""Unified FDTA efficiency benchmark for AirMOT and UA-DETRAC.

Protocol (kept consistent with the project's previous efficiency analysis):
  * batch size = 1;
  * exactly one visible CUDA GPU;
  * model.eval() inference;
  * 100 stateful warm-up frames by default;
  * latency is measured over the complete evaluation split;
  * data loading, image decoding/resizing, host-to-device transfer,
    visualization, result serialization, and TrackEval are excluded;
  * the timed region is RuntimeTracker.update(), i.e. detector forward,
    trajectory embedding/modeling, ID decoding/assignment, and online
    trajectory-state/postprocessing;
  * CUDA is synchronized immediately before and after every timed frame;
  * FPS = 1000 / mean latency (ms);
  * total model parameters are reported;
  * FLOPs are averaged over uniformly sampled, stateful online frames because
    the number of active trajectories varies with time.

PyTorch profiler FLOP counting can under-count custom CUDA operators (notably
custom deformable-attention kernels). The script therefore records profiler
FLOPs as an explicitly documented, reproducible estimate rather than silently
mixing incompatible FLOP tools.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from accelerate import Accelerator
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader

from configs.util import load_super_config
from data.airmot import AirMot
from data.seq_dataset import SeqDataset
from data.uadetrac import UADETRAC
from models.airmot_runtime_tracker import AirMOTRuntimeTracker
from models.fdta import build as build_fdta
from models.misc import get_model, load_checkpoint, load_previous_checkpoint
from models.runtime_tracker import RuntimeTracker
from utils.misc import yaml_to_dict


@dataclass
class FlopSample:
    global_frame_index: int
    sequence: str
    sequence_frame_index: int
    active_tracks_before: int
    input_height: int
    input_width: int
    flops: float


@dataclass
class EfficiencySummary:
    dataset: str
    split: str
    checkpoint: str
    gpu: str
    precision: str
    batch_size: int
    warmup_frames: int
    timed_frames: int
    max_shorter: int
    max_longer: int
    total_params: int
    params_m: float
    mean_latency_ms: float
    latency_std_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    fps: float
    mean_profiled_flops: float
    profiled_flops_g: float
    profiled_flops_std_g: float
    flops_samples: int
    gpu_only_mean_ms: float | None
    gpu_only_fps: float | None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure FDTA Params/FLOPs/Latency/FPS with a fixed online-MOT protocol."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["AirMot", "AirMOT", "UA-DETRAC", "UADETRAC", "UA_DETRAC"],
    )
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmup-frames", type=int, default=100)
    parser.add_argument(
        "--flops-samples",
        type=int,
        default=20,
        help="Number of uniformly sampled stateful online frames for profiler FLOPs.",
    )
    parser.add_argument("--image-max-shorter", type=int, default=800)
    parser.add_argument(
        "--image-max-longer",
        type=int,
        default=None,
        help="Defaults to INFERENCE_MAX_LONGER from the YAML config.",
    )
    parser.add_argument(
        "--precision",
        choices=["FP32", "FP16"],
        default=None,
        help="Defaults to INFERENCE_DTYPE from the YAML config.",
    )
    parser.add_argument(
        "--gpu-only-timing",
        action="store_true",
        help="Also report CUDA-event GPU elapsed time. Primary latency remains synchronized wall-clock latency.",
    )
    parser.add_argument(
        "--allow-non-l40s",
        action="store_true",
        help="Permit execution on a GPU other than NVIDIA L40S. Default is strict for protocol consistency.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    config = yaml_to_dict(path)
    super_path = config.get("SUPER_CONFIG_PATH")
    return load_super_config(config, super_path)


def canonical_dataset_name(name: str) -> str:
    if name in {"AirMot", "AirMOT"}:
        return "AirMOT"
    return "UA-DETRAC"


def build_dataset(name: str, data_root: str, split: str):
    canonical = canonical_dataset_name(name)
    if canonical == "AirMOT":
        return AirMot(data_root=data_root, split=split, load_annotation=False)
    return UADETRAC(data_root=data_root, split=split, load_annotation=False)


def to_tensor_dtype(precision: str) -> torch.dtype:
    if precision == "FP32":
        return torch.float32
    if precision == "FP16":
        return torch.float16
    raise ValueError(precision)


def move_nested_tensor_to_device(image, device: torch.device):
    # Transfer is deliberately completed before the timing boundary.
    image.tensors = image.tensors.to(device=device, non_blocking=False)
    image.mask = image.mask.to(device=device, non_blocking=False)
    return image


def make_loader(dataset, sequence_name: str, max_shorter: int, max_longer: int, dtype: torch.dtype):
    sequence_dataset = SeqDataset(
        seq_info=dataset.sequence_infos[sequence_name],
        image_paths=dataset.image_paths[sequence_name],
        max_shorter=max_shorter,
        max_longer=max_longer,
        size_divisibility=0,
        dtype=dtype,
    )
    # num_workers=0 makes the benchmark boundary unambiguous. Loading and
    # preprocessing still happen before each timed tracker.update().
    loader = DataLoader(
        dataset=sequence_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=lambda samples: samples[0],
    )
    return sequence_dataset, loader


def tracker_kwargs(config: dict, dtype: torch.dtype) -> dict:
    return {
        "use_sigmoid": config.get("USE_FOCAL_LOSS", False),
        "assignment_protocol": config.get("ASSIGNMENT_PROTOCOL", "object-max"),
        "miss_tolerance": config["MISS_TOLERANCE"],
        "det_thresh": config["DET_THRESH"],
        "newborn_thresh": config["NEWBORN_THRESH"],
        "id_thresh": config["ID_THRESH"],
        "area_thresh": config.get("AREA_THRESH", 0),
        "only_detr": (
            config["INFERENCE_ONLY_DETR"]
            if config.get("INFERENCE_ONLY_DETR") is not None
            else config["ONLY_DETR"]
        ),
        "dtype": dtype,
    }


def make_tracker(dataset_name: str, model, sequence_hw: tuple, config: dict, dtype: torch.dtype):
    kwargs = tracker_kwargs(config, dtype)
    if canonical_dataset_name(dataset_name) == "AirMOT":
        return AirMOTRuntimeTracker(
            model=model,
            sequence_hw=sequence_hw,
            class_aware=config.get("CLASS_AWARE_ASSOCIATION", True),
            **kwargs,
        )
    return RuntimeTracker(model=model, sequence_hw=sequence_hw, **kwargs)


def total_frames(dataset, sequence_names: list[str]) -> int:
    return int(sum(len(dataset.image_paths[name]) for name in sequence_names))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def uniformly_spaced_indices(n_frames: int, n_samples: int) -> set[int]:
    if n_frames <= 0 or n_samples <= 0:
        return set()
    n_samples = min(n_samples, n_frames)
    if n_samples == 1:
        return {n_frames // 2}
    return {
        int(round(i * (n_frames - 1) / (n_samples - 1)))
        for i in range(n_samples)
    }


def warm_up(
    dataset_name: str,
    dataset,
    sequence_names: list[str],
    model,
    config: dict,
    dtype: torch.dtype,
    device: torch.device,
    max_shorter: int,
    max_longer: int,
    warmup_frames: int,
):
    if warmup_frames <= 0:
        return

    processed = 0
    for sequence_name in sequence_names:
        seq_dataset, loader = make_loader(
            dataset, sequence_name, max_shorter, max_longer, dtype
        )
        tracker = make_tracker(
            dataset_name,
            model,
            seq_dataset.seq_hw(),
            config,
            dtype,
        )
        for image, _ in loader:
            image = move_nested_tensor_to_device(image, device)
            torch.cuda.synchronize(device)
            tracker.update(image=image)
            torch.cuda.synchronize(device)
            processed += 1
            if processed >= warmup_frames:
                return


def measure_latency(
    dataset_name: str,
    dataset,
    sequence_names: list[str],
    model,
    config: dict,
    dtype: torch.dtype,
    device: torch.device,
    max_shorter: int,
    max_longer: int,
    gpu_only_timing: bool,
):
    latencies_ms: list[float] = []
    gpu_ms: list[float] = []
    input_shapes: set[tuple[int, int]] = set()

    for sequence_name in sequence_names:
        seq_dataset, loader = make_loader(
            dataset, sequence_name, max_shorter, max_longer, dtype
        )
        tracker = make_tracker(
            dataset_name,
            model,
            seq_dataset.seq_hw(),
            config,
            dtype,
        )

        for image, _ in loader:
            input_shapes.add((int(image.tensors.shape[-2]), int(image.tensors.shape[-1])))
            image = move_nested_tensor_to_device(image, device)

            # Synchronization immediately before t0 guarantees that H2D transfer
            # and previous asynchronous kernels are outside the timed region.
            torch.cuda.synchronize(device)

            if gpu_only_timing:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

            start = time.perf_counter()
            tracker.update(image=image)
            torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies_ms.append(elapsed_ms)

            if gpu_only_timing:
                end_event.record()
                end_event.synchronize()
                gpu_ms.append(float(start_event.elapsed_time(end_event)))

    return latencies_ms, gpu_ms, input_shapes


def profile_flops(
    dataset_name: str,
    dataset,
    sequence_names: list[str],
    model,
    config: dict,
    dtype: torch.dtype,
    device: torch.device,
    max_shorter: int,
    max_longer: int,
    n_samples: int,
):
    n_total = total_frames(dataset, sequence_names)
    sample_indices = uniformly_spaced_indices(n_total, n_samples)
    if not sample_indices:
        return []

    samples: list[FlopSample] = []
    global_index = 0

    for sequence_name in sequence_names:
        seq_dataset, loader = make_loader(
            dataset, sequence_name, max_shorter, max_longer, dtype
        )
        tracker = make_tracker(
            dataset_name,
            model,
            seq_dataset.seq_hw(),
            config,
            dtype,
        )

        for sequence_frame_index, (image, _) in enumerate(loader):
            input_h = int(image.tensors.shape[-2])
            input_w = int(image.tensors.shape[-1])
            image = move_nested_tensor_to_device(image, device)
            torch.cuda.synchronize(device)

            if global_index in sample_indices:
                active_tracks = int(tracker.trajectory_features.shape[1])
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=False,
                    profile_memory=False,
                    with_flops=True,
                ) as prof:
                    tracker.update(image=image)
                torch.cuda.synchronize(device)

                frame_flops = float(
                    sum(float(event.flops or 0.0) for event in prof.key_averages())
                )
                samples.append(
                    FlopSample(
                        global_frame_index=global_index,
                        sequence=sequence_name,
                        sequence_frame_index=sequence_frame_index,
                        active_tracks_before=active_tracks,
                        input_height=input_h,
                        input_width=input_w,
                        flops=frame_flops,
                    )
                )
            else:
                # Stateful traversal is required so sampled FLOPs reflect the
                # real number/history of active trajectories at that frame.
                tracker.update(image=image)
                torch.cuda.synchronize(device)

            global_index += 1

    return samples


def save_outputs(
    output_dir: Path,
    summary: EfficiencySummary,
    flop_samples: list[FlopSample],
    input_shapes: set[tuple[int, int]],
):
    output_dir.mkdir(parents=True, exist_ok=True)

    json_payload = asdict(summary)
    json_payload["input_shapes_hw"] = [list(shape) for shape in sorted(input_shapes)]
    json_payload["timing_scope"] = (
        "RuntimeTracker.update only: detector forward + trajectory embedding/modeling + "
        "ID decoding/assignment + online trajectory-state/postprocessing"
    )
    json_payload["excluded_from_latency"] = [
        "data loading",
        "image decoding",
        "resize/normalization",
        "CPU-to-GPU transfer",
        "visualization",
        "result serialization/file I/O",
        "TrackEval",
    ]
    json_payload["flops_note"] = (
        "PyTorch profiler FLOPs averaged over uniformly sampled stateful online frames; "
        "custom CUDA operators such as deformable attention may be under-counted."
    )

    with (output_dir / "efficiency_summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)

    with (output_dir / "flops_samples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(FlopSample.__annotations__.keys()))
        writer.writeheader()
        for sample in flop_samples:
            writer.writerow(asdict(sample))

    lines = [
        "FDTA Efficiency Analysis",
        "========================",
        f"Dataset                 : {summary.dataset}",
        f"Split                   : {summary.split}",
        f"Checkpoint              : {summary.checkpoint}",
        f"GPU                     : {summary.gpu}",
        f"Precision               : {summary.precision}",
        f"Batch size              : {summary.batch_size}",
        f"Warm-up frames          : {summary.warmup_frames}",
        f"Timed frames            : {summary.timed_frames}",
        f"Resize rule             : shorter={summary.max_shorter}, longer<={summary.max_longer}",
        f"Observed input HxW      : {sorted(input_shapes)}",
        "",
        f"Params (M)              : {summary.params_m:.3f}",
        f"Profiled FLOPs (G)      : {summary.profiled_flops_g:.3f}",
        f"Mean Latency (ms)       : {summary.mean_latency_ms:.3f}",
        f"FPS                     : {summary.fps:.3f}",
        "",
        f"Latency std (ms)        : {summary.latency_std_ms:.3f}",
        f"Median Latency (ms)     : {summary.median_latency_ms:.3f}",
        f"P95 Latency (ms)        : {summary.p95_latency_ms:.3f}",
        f"FLOPs std (G)           : {summary.profiled_flops_std_g:.3f}",
        f"FLOPs samples           : {summary.flops_samples}",
    ]
    if summary.gpu_only_mean_ms is not None:
        lines.extend([
            f"GPU-only mean (ms)      : {summary.gpu_only_mean_ms:.3f}",
            f"GPU-only FPS            : {summary.gpu_only_fps:.3f}",
        ])
    lines.extend([
        "",
        "Primary timing scope:",
        "  RuntimeTracker.update only.",
        "  Includes detector, FDTA trajectory/ID modeling, assignment, and online state update.",
        "  Excludes loading/decoding/preprocessing, H2D transfer, visualization, file I/O, TrackEval.",
        "",
        "FLOPs note:",
        "  Mean profiler FLOPs across uniformly sampled stateful online frames.",
        "  Custom CUDA operators (e.g. deformable attention) may be under-counted by PyTorch profiler.",
    ])
    (output_dir / "efficiency_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()

    if args.warmup_frames < 0:
        raise ValueError("--warmup-frames must be >= 0")
    if args.flops_samples < 0:
        raise ValueError("--flops-samples must be >= 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the fixed efficiency protocol.")

    # The previous protocol is explicitly single-GPU. Enforce one visible GPU
    # rather than silently benchmarking with a different execution environment.
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Efficiency analysis requires exactly one visible CUDA GPU. "
            "Run with CUDA_VISIBLE_DEVICES=<gpu_id>."
        )

    accelerator = Accelerator()
    if accelerator.num_processes != 1:
        raise RuntimeError(
            "Do not use accelerate launch for efficiency analysis; run one Python process."
        )
    device = accelerator.device
    gpu_name = torch.cuda.get_device_name(device)
    if "L40S" not in gpu_name.upper() and not args.allow_non_l40s:
        raise RuntimeError(
            f"Protocol requires NVIDIA L40S, but visible GPU is '{gpu_name}'. "
            "Use --allow-non-l40s only for debugging/non-comparable measurements."
        )

    config = load_config(args.config_path)
    config["DATA_ROOT"] = args.data_root
    config["INFERENCE_MODEL"] = args.checkpoint
    config["INFERENCE_SPLIT"] = args.split

    precision = args.precision or config.get("INFERENCE_DTYPE", "FP32")
    config["INFERENCE_DTYPE"] = precision
    tensor_dtype = to_tensor_dtype(precision)
    max_longer = args.image_max_longer or int(config["INFERENCE_MAX_LONGER"])

    dataset = build_dataset(args.dataset, args.data_root, args.split)
    sequence_names = sorted(dataset.sequence_infos.keys())
    if not sequence_names:
        raise RuntimeError(f"No sequences found for {args.dataset}/{args.split}.")
    n_frames = total_frames(dataset, sequence_names)

    model, _ = build_fdta(config=config)
    if config.get("USE_PREVIOUS_CHECKPOINT", False):
        load_previous_checkpoint(model, path=args.checkpoint)
    else:
        load_checkpoint(model, path=args.checkpoint)
    model = accelerator.prepare(model)
    model.eval()

    bare_model = get_model(model)
    n_params = int(sum(parameter.numel() for parameter in bare_model.parameters()))

    print("\n[FDTA efficiency protocol]")
    print(f"  dataset/split : {canonical_dataset_name(args.dataset)}/{args.split}")
    print(f"  GPU           : {gpu_name}")
    print(f"  precision     : {precision}")
    print("  batch size    : 1")
    print(f"  warm-up       : {args.warmup_frames} frames")
    print(f"  timed frames  : {n_frames} (complete split)")
    print("  timed scope   : RuntimeTracker.update only")
    print("  transfer/I-O  : excluded")

    warm_up(
        args.dataset,
        dataset,
        sequence_names,
        model,
        config,
        tensor_dtype,
        device,
        args.image_max_shorter,
        max_longer,
        args.warmup_frames,
    )
    torch.cuda.synchronize(device)

    latencies_ms, gpu_ms, input_shapes = measure_latency(
        args.dataset,
        dataset,
        sequence_names,
        model,
        config,
        tensor_dtype,
        device,
        args.image_max_shorter,
        max_longer,
        args.gpu_only_timing,
    )
    if len(latencies_ms) != n_frames:
        raise RuntimeError(
            f"Expected {n_frames} timed frames, got {len(latencies_ms)}."
        )

    flop_samples = profile_flops(
        args.dataset,
        dataset,
        sequence_names,
        model,
        config,
        tensor_dtype,
        device,
        args.image_max_shorter,
        max_longer,
        args.flops_samples,
    )

    mean_latency = statistics.fmean(latencies_ms)
    latency_std = statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    mean_flops = statistics.fmean([s.flops for s in flop_samples]) if flop_samples else 0.0
    flops_std = (
        statistics.pstdev([s.flops for s in flop_samples])
        if len(flop_samples) > 1
        else 0.0
    )
    mean_gpu_ms = statistics.fmean(gpu_ms) if gpu_ms else None

    summary = EfficiencySummary(
        dataset=canonical_dataset_name(args.dataset),
        split=args.split,
        checkpoint=str(Path(args.checkpoint).resolve()),
        gpu=gpu_name,
        precision=precision,
        batch_size=1,
        warmup_frames=args.warmup_frames,
        timed_frames=n_frames,
        max_shorter=args.image_max_shorter,
        max_longer=max_longer,
        total_params=n_params,
        params_m=n_params / 1e6,
        mean_latency_ms=mean_latency,
        latency_std_ms=latency_std,
        median_latency_ms=statistics.median(latencies_ms),
        p95_latency_ms=percentile(latencies_ms, 0.95),
        fps=1000.0 / mean_latency,
        mean_profiled_flops=mean_flops,
        profiled_flops_g=mean_flops / 1e9,
        profiled_flops_std_g=flops_std / 1e9,
        flops_samples=len(flop_samples),
        gpu_only_mean_ms=mean_gpu_ms,
        gpu_only_fps=(1000.0 / mean_gpu_ms if mean_gpu_ms else None),
    )

    output_dir = Path(args.output_dir)
    save_outputs(output_dir, summary, flop_samples, input_shapes)

    print("\n[Efficiency summary]")
    print(f"  Params (M)         : {summary.params_m:.3f}")
    print(f"  Profiled FLOPs (G) : {summary.profiled_flops_g:.3f}")
    print(f"  Mean Latency (ms)  : {summary.mean_latency_ms:.3f}")
    print(f"  FPS                 : {summary.fps:.3f}")
    if summary.gpu_only_mean_ms is not None:
        print(f"  GPU-only mean (ms) : {summary.gpu_only_mean_ms:.3f}")
    print(f"  Results             : {output_dir.resolve()}")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    main()
