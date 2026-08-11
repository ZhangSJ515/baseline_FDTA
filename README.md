# [CVPR 2026] From Detection to Association: Learning Discriminative Object Embeddings for Multi-Object Tracking

[![Arxiv](https://img.shields.io/badge/ArXiv-2512.02392-B31B1B.svg)](https://arxiv.org/abs/2512.02392)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-FDTA-yellow)](https://huggingface.co/Spongebobbbbbbbb/FDTA)

> **TL;DR.** We reveal that DETR-based end-to-end MOT suffers from overly similar object embeddings. FDTA explicitly enhances discriminativeness in this paradigm.

![Teaser](./assets/teaser.png)

## Adaptation status

This branch provides complete FDTA adapters for:

- **AirMOT**: four-class airport-surface MOT with class-aware association;
- **UA-DETRAC**: single-class vehicle MOT with robust MOT-format parsing;
- dataset loading, depth-map generation, checkpoint conversion, training, inference, native result serialization, TrackEval evaluation, setup validation, smoke tests, and static CI checks.

## Environment

```shell
conda create -n FDTA python=3.12
conda activate FDTA
pip install -r requirements.txt
cd models/ops/
sh make.sh
cd ../..
```

FDTA requires depth maps aligned with the RGB frames. The depth-generation scripts expect Video-Depth-Anything at `../Video_depth_anything` relative to this repository.

## Dataset layout

```text
/home/zsj/data/datasets/
├── AirMot/                       # AirMOT or AirMot are both accepted
│   ├── train/<sequence>/
│   │   ├── img1/
│   │   ├── depth/
│   │   └── gt/gt.txt
│   └── val/<sequence>/...
└── UA-DETRAC/                    # UADETRAC and UA_DETRAC are also accepted
    ├── train/<sequence>/
    │   ├── img1/
    │   ├── depth/
    │   ├── gt/gt.txt
    │   └── seqinfo.ini           # optional
    └── test/<sequence>/...
```

Frame images must have numeric stems, with arbitrary zero-padding and common image extensions. Depth files use the same stem as the corresponding RGB frame.

### AirMOT annotations

AirMOT ground-truth rows use:

```text
frame_id track_id class_id x y width height flag
```

Classes:

| ID | Class |
|---:|---|
| 0 | airplane |
| 1 | person |
| 2 | baggage_tug |
| 3 | follow_me_vehicle |

AirMOT tracker rows are serialized as:

```text
frame_id track_id class_id x y width height score
```

### UA-DETRAC annotations

The UA-DETRAC loader and evaluator accept comma- or whitespace-separated rows in these layouts:

```text
# Standard MOTChallenge
frame,id,x,y,width,height,mark,class,visibility[,unused]

# Compact single-class format
frame id x y width height [mark]

# Class-first eight-column format
frame id class x y width height flag
```

Set `UADETRAC_GT_FORMAT` to `auto`, `mot`, `native8`, or `simple` when explicit parsing is required. The model treats all valid UA-DETRAC targets as the single `vehicle` class. Tracker files are written in standard MOTChallenge format:

```text
frame,id,x,y,width,height,score,-1,-1,-1
```

## Depth maps

Generate AirMOT depth maps:

```shell
DATA_ROOT=/home/zsj/data/datasets \
SPLITS="train val" \
bash scripts/generate_airmot_depth.sh
```

Generate UA-DETRAC depth maps:

```shell
DATA_ROOT=/home/zsj/data/datasets \
SPLITS="train test" \
bash scripts/generate_uadetrac_depth.sh
```

Validate the resulting datasets:

```shell
DATA_ROOT=/home/zsj/data/datasets bash scripts/check_airmot_setup.sh
DATA_ROOT=/home/zsj/data/datasets bash scripts/check_uadetrac_setup.sh
```

## Detector pretraining conversion

FDTA uses a Deformable-DETR detector checkpoint. Convert a COCO checkpoint to the target classifier head before training.

AirMOT copies the COCO airplane and person rows and initializes the two airport-specific vehicle classes from the mean foreground representation:

```shell
SOURCE_CHECKPOINT=./pretrains/r50_deformable_detr_coco.pth \
OUTPUT_CHECKPOINT=./pretrains/r50_deformable_detr_coco_airmot.pth \
bash scripts/prepare_airmot_pretrain.sh
```

UA-DETRAC initializes its one-class detector from the COCO car classifier row:

```shell
SOURCE_CHECKPOINT=./pretrains/r50_deformable_detr_coco.pth \
OUTPUT_CHECKPOINT=./pretrains/r50_deformable_detr_coco_uadetrac.pth \
bash scripts/prepare_uadetrac_pretrain.sh
```

## AirMOT

Configuration: `configs/airmot.yaml`.

Train:

```shell
NUM_GPUS=8 \
DATA_ROOT=/home/zsj/data/datasets \
OUTPUTS_DIR=./outputs/fdta_airmot \
bash scripts/train_airmot.sh
```

Evaluate:

```shell
MODEL_PATH=./outputs/fdta_airmot/checkpoint_12.pth \
NUM_GPUS=8 \
DATA_ROOT=/home/zsj/data/datasets \
bash scripts/evaluate_airmot.sh
```

Generate tracker files only:

```shell
MODEL_PATH=./outputs/fdta_airmot/checkpoint_12.pth \
SPLIT=val \
NUM_GPUS=8 \
DATA_ROOT=/home/zsj/data/datasets \
bash scripts/submit_airmot.sh
```

AirMOT inference uses `AirMOTRuntimeTracker`, which masks cross-class identity candidates and stabilizes the class assigned to an established trajectory. TrackEval evaluates the four classes independently and writes the detection-weighted combined summary to:

```text
tracker/cls_comb_det_av_summary.txt
```

## UA-DETRAC

Configuration: `configs/uadetrac.yaml`.

Train:

```shell
NUM_GPUS=8 \
DATA_ROOT=/home/zsj/data/datasets \
OUTPUTS_DIR=./outputs/fdta_uadetrac \
bash scripts/train_uadetrac.sh
```

Evaluate:

```shell
MODEL_PATH=./outputs/fdta_uadetrac/checkpoint_12.pth \
NUM_GPUS=8 \
DATA_ROOT=/home/zsj/data/datasets \
SPLIT=test \
bash scripts/evaluate_uadetrac.sh
```

Generate standard MOTChallenge tracker files only:

```shell
MODEL_PATH=./outputs/fdta_uadetrac/checkpoint_12.pth \
NUM_GPUS=8 \
DATA_ROOT=/home/zsj/data/datasets \
SPLIT=test \
bash scripts/submit_uadetrac.sh
```

The UA-DETRAC TrackEval adapter reports HOTA, DetA, AssA, MOTA, IDF1, and the remaining standard fields in:

```text
tracker/vehicle_summary.txt
```

## Smoke tests

The smoke tests build temporary toy datasets and validate the loader, depth resolver, tracker-file parser, and TrackEval preprocessing without requiring the real datasets:

```shell
bash scripts/smoke_test_airmot.sh
bash scripts/smoke_test_uadetrac.sh
```

## Original FDTA workflows

### DanceTrack training

```shell
accelerate launch --num_processes=4 train.py \
  --data-root /path/to/your/datasets/ \
  --exp-name fdta_dancetrack \
  --config-path ./configs/dancetrack.yaml \
  --detr-pretrain ./pretrains/r50_deformable_detr_coco_dancetrack.pth
```

Replace `dancetrack` with `sportsmot` or `bft` for the other original benchmarks.

### Original inference

```shell
accelerate launch --num_processes=4 submit_and_evaluate.py \
  --data-root /path/to/your/datasets/ \
  --inference-mode evaluate \
  --config-path ./configs/dancetrack.yaml \
  --inference-model ./checkpoints/dancetrack.pth \
  --outputs-dir ./outputs/ \
  --inference-dataset DanceTrack \
  --inference-split val
```

## Main results reported by FDTA

### DanceTrack

| Training Data | HOTA | IDF1 | AssA | MOTA | DetA |
|---|---:|---:|---:|---:|---:|
| train | 71.7 | 77.2 | 63.5 | 91.3 | 81.0 |
| train+val | 74.4 | 80.0 | 67.0 | 92.2 | 82.7 |

### SportsMOT

| Training Data | HOTA | IDF1 | AssA | MOTA | DetA |
|---|---:|---:|---:|---:|---:|
| train | 74.2 | 78.5 | 65.5 | 93.0 | 84.1 |

### BFT

| Training Data | HOTA | IDF1 | AssA | MOTA | DetA |
|---|---:|---:|---:|---:|---:|
| train | 72.2 | 84.2 | 74.5 | 78.2 | 70.1 |

No AirMOT or UA-DETRAC metric is claimed in this repository until the adapted model has been trained and evaluated on the corresponding local split.

## Acknowledgements

The code is built on top of:

- [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR)
- [MOTR](https://github.com/megvii-research/MOTR)
- [MOTIP](https://github.com/MCG-NJU/MOTIP)
- [MonoDETR](https://github.com/ZrrSkywalker/MonoDETR)
- [TrackEval](https://github.com/JonathonLuiten/TrackEval)

## Citation

```bibtex
@article{shao2025fdta,
  title={From Detection to Association: Learning Discriminative Object Embeddings for Multi-Object Tracking},
  author={Shao, Yuqing and Yang, Yuchen and Yu, Rui and Li, Weilong and Guo, Xu and Yan, Huaicheng and Wang, Wei and Sun, Xiao},
  journal={arXiv preprint arXiv:2512.02392},
  year={2025}
}
```
