from __future__ import annotations

import os
import subprocess
import time

import torch
from accelerate import Accelerator
from accelerate.state import PartialState
from torch.utils.data import DataLoader

from configs.util import load_super_config, update_config
from data.seq_dataset import SeqDataset
from data.uadetrac import UADETRAC
from log.log import Metrics
from log.logger import Logger
from models.fdta import build as build_fdta
from models.misc import load_checkpoint, load_previous_checkpoint
from models.runtime_tracker import RuntimeTracker
from runtime_option import runtime_option
from submit_and_evaluate import get_eval_metrics_dict, get_results_of_one_sequence
from utils.misc import yaml_to_dict


def _resolve_dataset_dir(data_root: str, split: str) -> str:
    for name in ("UA-DETRAC", "UADETRAC", "UA_DETRAC"):
        candidate = os.path.join(data_root, name, split)
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(data_root, "UA-DETRAC", split)


def submit_and_evaluate_uadetrac(config: dict):
    accelerator = Accelerator()
    state = PartialState()

    mode = config["INFERENCE_MODE"]
    if mode not in {"submit", "evaluate"}:
        raise ValueError(f"Unsupported UA-DETRAC inference mode: {mode}")

    inference_dataset = config.get("INFERENCE_DATASET") or "UA-DETRAC"
    if inference_dataset not in {"UA-DETRAC", "UADETRAC", "UA_DETRAC"}:
        raise ValueError(
            "submit_and_evaluate_uadetrac.py only supports UA-DETRAC, "
            f"got {inference_dataset}."
        )

    inference_model = config["INFERENCE_MODEL"]
    if not inference_model:
        raise ValueError("INFERENCE_MODEL must point to a trained FDTA checkpoint.")
    if not config.get("OUTPUTS_DIR"):
        raise ValueError("OUTPUTS_DIR is not set.")

    model_name = os.path.splitext(os.path.basename(inference_model))[0]
    outputs_dir = os.path.join(
        config["OUTPUTS_DIR"],
        mode,
        config.get("INFERENCE_GROUP", "default"),
        "UA-DETRAC",
        config["INFERENCE_SPLIT"],
        model_name,
    )
    existed = os.path.exists(outputs_dir)
    accelerator.wait_for_everyone()
    os.makedirs(outputs_dir, exist_ok=True)

    logger = Logger(logdir=outputs_dir)
    logger.config(config=config)
    logger.info(
        f"{mode.capitalize()} UA-DETRAC model: {inference_model}, "
        f"split: {config['INFERENCE_SPLIT']}."
    )
    if existed:
        logger.warning(
            f"Output directory '{outputs_dir}' exists and may be overwritten."
        )
        time.sleep(2)

    model, _ = build_fdta(config=config)
    if config.get("USE_PREVIOUS_CHECKPOINT", False):
        load_previous_checkpoint(model, path=inference_model)
    else:
        load_checkpoint(model, path=inference_model)
    model = accelerator.prepare(model)

    metrics = submit_and_evaluate_one_uadetrac_model(
        is_evaluate=mode == "evaluate",
        accelerator=accelerator,
        state=state,
        logger=logger,
        model=model,
        data_root=config["DATA_ROOT"],
        data_split=config["INFERENCE_SPLIT"],
        outputs_dir=outputs_dir,
        image_max_longer=config["INFERENCE_MAX_LONGER"],
        size_divisibility=config.get("SIZE_DIVISIBILITY", 0),
        use_sigmoid=config.get("USE_FOCAL_LOSS", False),
        assignment_protocol=config.get("ASSIGNMENT_PROTOCOL", "object-max"),
        miss_tolerance=config["MISS_TOLERANCE"],
        det_thresh=config["DET_THRESH"],
        newborn_thresh=config["NEWBORN_THRESH"],
        id_thresh=config["ID_THRESH"],
        area_thresh=config.get("AREA_THRESH", 0),
        inference_only_detr=(
            config["INFERENCE_ONLY_DETR"]
            if config.get("INFERENCE_ONLY_DETR") is not None
            else config["ONLY_DETR"]
        ),
        dtype=config.get("INFERENCE_DTYPE", "FP32"),
        filter_gt_by_mark=config.get("UADETRAC_FILTER_GT_BY_MARK", True),
        min_visibility=config.get("UADETRAC_MIN_VISIBILITY", 0.0),
        gt_format=config.get("UADETRAC_GT_FORMAT", "auto"),
    )

    if metrics is not None:
        metrics.sync()
        logger.metrics(
            log=(
                f"Finish UA-DETRAC evaluation for '{inference_model}' on "
                f"split '{config['INFERENCE_SPLIT']}': "
            ),
            metrics=metrics,
            fmt="{global_average:.4f}",
        )


def submit_and_evaluate_one_uadetrac_model(
        is_evaluate: bool,
        accelerator: Accelerator,
        state: PartialState,
        logger: Logger,
        model,
        data_root: str,
        data_split: str,
        outputs_dir: str,
        image_max_shorter: int = 800,
        image_max_longer: int = 1536,
        size_divisibility: int = 0,
        use_sigmoid: bool = False,
        assignment_protocol: str = "object-max",
        miss_tolerance: int = 30,
        det_thresh: float = 0.5,
        newborn_thresh: float = 0.5,
        id_thresh: float = 0.1,
        area_thresh: int = 0,
        inference_only_detr: bool = False,
        dtype: str = "FP32",
        filter_gt_by_mark: bool = True,
        min_visibility: float = 0.0,
        gt_format: str = "auto",
):
    inference_dataset = UADETRAC(
        data_root=data_root,
        split=data_split,
        load_annotation=False,
    )

    match dtype:
        case "FP32":
            tensor_dtype = torch.float32
        case "FP16":
            tensor_dtype = torch.float16
        case _:
            raise ValueError(f"Unknown inference dtype '{dtype}'.")

    sequence_names = sorted(inference_dataset.sequence_infos.keys())
    if not sequence_names:
        raise RuntimeError("No UA-DETRAC sequences are available for inference.")

    if len(sequence_names) <= state.process_index:
        selected_sequence_names = [sequence_names[0]]
        is_fake_process = True
    else:
        selected_sequence_names = [
            sequence_name
            for index, sequence_name in enumerate(sequence_names)
            if index % state.num_processes == state.process_index
        ]
        is_fake_process = False

    tracker_dir = os.path.join(outputs_dir, "tracker")
    os.makedirs(tracker_dir, exist_ok=True)

    for sequence_name in selected_sequence_names:
        sequence_dataset = SeqDataset(
            seq_info=inference_dataset.sequence_infos[sequence_name],
            image_paths=inference_dataset.image_paths[sequence_name],
            max_shorter=image_max_shorter,
            max_longer=image_max_longer,
            size_divisibility=size_divisibility,
            dtype=tensor_dtype,
        )
        sequence_loader = DataLoader(
            dataset=sequence_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=lambda samples: samples[0],
        )
        runtime_tracker = RuntimeTracker(
            model=model,
            sequence_hw=sequence_dataset.seq_hw(),
            use_sigmoid=use_sigmoid,
            assignment_protocol=assignment_protocol,
            miss_tolerance=miss_tolerance,
            det_thresh=det_thresh,
            newborn_thresh=newborn_thresh,
            id_thresh=id_thresh,
            area_thresh=area_thresh,
            only_detr=inference_only_detr,
            dtype=tensor_dtype,
        )

        logger.info(
            f"Submitting UA-DETRAC sequence {sequence_name} with "
            f"{len(sequence_loader)} frames.",
            only_main=False,
        )
        sequence_results, sequence_fps = get_results_of_one_sequence(
            runtime_tracker=runtime_tracker,
            sequence_loader=sequence_loader,
            logger=logger,
        )

        if not is_fake_process:
            frame_ids = inference_dataset.sequence_infos[sequence_name]["frame_ids"]
            rows = []
            for frame_index, frame_result in enumerate(sequence_results):
                frame_id = frame_ids[frame_index]
                for obj_id, score, bbox in zip(
                        frame_result["id"],
                        frame_result["score"],
                        frame_result["bbox"],
                ):
                    rows.append(
                        f"{frame_id},{int(obj_id.item())},"
                        f"{float(bbox[0].item()):.6f},"
                        f"{float(bbox[1].item()):.6f},"
                        f"{float(bbox[2].item()):.6f},"
                        f"{float(bbox[3].item()):.6f},"
                        f"{float(score.item()):.6f},-1,-1,-1\n"
                    )

            result_path = os.path.join(tracker_dir, f"{sequence_name}.txt")
            with open(result_path, "w", encoding="utf-8") as result_file:
                result_file.writelines(rows)
            logger.success(
                f"UA-DETRAC sequence {sequence_name} saved to {result_path}; "
                f"FPS: {sequence_fps:.2f}.",
                only_main=False,
            )
        else:
            logger.success(
                f"Fake UA-DETRAC sequence {sequence_name} completed; "
                f"FPS: {sequence_fps:.2f}.",
                only_main=False,
            )

    accelerator.wait_for_everyone()
    if not is_evaluate:
        logger.success(f"UA-DETRAC submission files saved to {tracker_dir}.")
        return None

    if accelerator.is_main_process:
        gt_dir = _resolve_dataset_dir(data_root, data_split)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        trackeval_script = os.path.join(
            current_dir, "TrackEval", "scripts", "run_uadetrac.py"
        )
        command = [
            "python",
            trackeval_script,
            "--GT_FOLDER", gt_dir,
            "--TRACKERS_FOLDER", tracker_dir,
            "--OUTPUT_FOLDER", tracker_dir,
            "--TRACKERS_TO_EVAL", "",
            "--TRACKER_SUB_FOLDER", "",
            "--OUTPUT_SUB_FOLDER", "",
            "--CLASSES_TO_EVAL", "vehicle",
            "--METRICS", "HOTA", "CLEAR", "Identity",
            "--USE_PARALLEL", "True",
            "--NUM_PARALLEL_CORES", "8",
            "--PLOT_CURVES", "False",
            "--FILTER_GT_BY_MARK", str(bool(filter_gt_by_mark)),
            "--MIN_VISIBILITY", str(float(min_visibility)),
            "--GT_FORMAT", gt_format,
        ]
        logger.info("Start native UA-DETRAC evaluation.")
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"UA-DETRAC TrackEval failed with code {completed.returncode}."
            )
        logger.success("UA-DETRAC TrackEval completed.")

    accelerator.wait_for_everyone()
    metric_path = os.path.join(tracker_dir, "vehicle_summary.txt")
    metric_values = get_eval_metrics_dict(metric_path=metric_path)
    metrics = Metrics()
    for metric_name in [
        "HOTA",
        "DetA",
        "AssA",
        "DetPr",
        "DetRe",
        "AssPr",
        "AssRe",
        "MOTA",
        "IDF1",
    ]:
        metrics[metric_name].update(metric_values[metric_name])
    logger.success(f"Loaded UA-DETRAC metrics from {metric_path}.")
    return metrics


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    option = runtime_option()
    config = yaml_to_dict(option.config_path)
    if option.super_config_path is not None:
        config = load_super_config(config, option.super_config_path)
    else:
        config = load_super_config(config, config["SUPER_CONFIG_PATH"])
    config = update_config(config=config, option=option)
    submit_and_evaluate_uadetrac(config=config)
