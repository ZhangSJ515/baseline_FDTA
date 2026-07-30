# [CVPR 2026] From Detection to Association: Learning Discriminative Object Embeddings for Multi-Object Tracking

[![Arxiv](https://img.shields.io/badge/ArXiv-2512.02392-B31B1B.svg)](https://arxiv.org/abs/2512.02392)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-FDTA-yellow)](https://huggingface.co/Spongebobbbbbbbb/FDTA)

> **TL;DR.** We reveal that DETR-based end-to-end MOT suffers from overly similar object embeddings. FDTA explicitly enhances discriminativeness in this paradigm.

![Teaser](./assets/teaser.png)

## 📢 News

- **[2026]** Our paper has been accepted by **CVPR 2026**.
- **[AirMOT adaptation]** This branch adds native four-class AirMOT training, class-aware inference, native result serialization, and TrackEval evaluation.

## 🚀 Getting Started

### 1. Environment Setup

```shell
conda create -n FDTA python=3.12
conda activate FDTA
pip install -r requirements.txt
cd models/ops/
sh make.sh
cd ../..
```

### 2. Data Preparation

FDTA requires depth maps alongside RGB images:

```text
datasets/
├── DanceTrack/
├── SportsMOT/
├── BFT/
└── AirMot/
    ├── train/
    │   └── <sequence>/
    │       ├── img1/
    │       ├── depth/
    │       └── gt/gt.txt
    ├── val/
    └── test/
```

AirMOT ground-truth rows use the native format:

```text
frame_id track_id class_id x y width height flag
```

AirMOT classes are:

| ID | Class |
|---:|---|
| 0 | airplane |
| 1 | person |
| 2 | baggage_tug |
| 3 | follow_me_vehicle |

The loader accepts comma- or whitespace-separated annotations and numeric frame names with arbitrary zero-padding. Both `AirMot` and `AirMOT` directory capitalization are recognized.

Generate AirMOT depth maps with Video-Depth-Anything:

```shell
DATA_ROOT=/home/zsj/data/datasets bash scripts/generate_airmot_depth.sh
```

Validate dataset annotations, depth maps, and pretrained weights before training:

```shell
DATA_ROOT=/home/zsj/data/datasets bash scripts/check_airmot_setup.sh
```

### 3. Pre-trained Weights

The original FDTA experiments use COCO-pretrained Deformable DETR weights from MOTIP. AirMOT has four classes, whereas the original loader assumes a one-class benchmark. Convert the COCO classifier head once:

```shell
SOURCE_CHECKPOINT=./pretrains/r50_deformable_detr_coco.pth \
OUTPUT_CHECKPOINT=./pretrains/r50_deformable_detr_coco_airmot.pth \
bash scripts/prepare_airmot_pretrain.sh
```

The converter copies the COCO `airplane` and `person` classifier rows and initializes `baggage_tug` and `follow_me_vehicle` from the mean foreground classifier representation. All other detector parameters are preserved.

### 4. AirMOT Training

The AirMOT configuration is `configs/airmot.yaml`. The default dataset root is `/home/zsj/data/datasets` and can be overridden from the command line.

```shell
NUM_GPUS=8 \
DATA_ROOT=/home/zsj/data/datasets \
OUTPUTS_DIR=./outputs/fdta_airmot \
bash scripts/train_airmot.sh
```

The training configuration uses:

- four detection classes;
- 300 Deformable-DETR queries;
- FDTA identity modeling with a 100-entry ID vocabulary;
- the original FDTA depth and contrastive objectives;
- no evaluation inside `train.py`, because AirMOT uses the dedicated native multi-class evaluation entry point.

### 5. AirMOT Evaluation and Submission

Evaluate a checkpoint on the AirMOT validation split:

```shell
MODEL_PATH=./outputs/fdta_airmot/checkpoint_12.pth \
NUM_GPUS=8 \
DATA_ROOT=/home/zsj/data/datasets \
bash scripts/evaluate_airmot.sh
```

Generate native AirMOT tracker files for a split:

```shell
MODEL_PATH=./outputs/fdta_airmot/checkpoint_12.pth \
SPLIT=test \
NUM_GPUS=8 \
DATA_ROOT=/home/zsj/data/datasets \
bash scripts/submit_airmot.sh
```

Tracker rows are written as:

```text
frame_id track_id class_id x y width height score
```

AirMOT inference uses `AirMOTRuntimeTracker`, which applies class-consistent ID assignment and stabilizes the class of an established trajectory. This prevents an embedding match from associating objects across heterogeneous AirMOT classes.

The native TrackEval adapter evaluates each class independently and also writes detection-weighted combined results:

```text
tracker/cls_comb_det_av_summary.txt
```

The combined summary contains HOTA, DetA, AssA, MOTA, IDF1, and the other standard TrackEval fields used in the AirMOT experiments.

## Original FDTA Workflows

### DanceTrack Training

```shell
accelerate launch --num_processes=4 train.py \
  --data-root /path/to/your/datasets/ \
  --exp-name fdta_dancetrack \
  --config-path ./configs/dancetrack.yaml \
  --detr-pretrain ./pretrains/r50_deformable_detr_coco_dancetrack.pth
```

Replace `dancetrack` with `sportsmot` or `bft` for the other original benchmarks.

### Original Inference

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

## Main Results Reported by FDTA

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

No AirMOT metric is claimed in this repository until the adapted model has been trained and evaluated on the local AirMOT split.

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
