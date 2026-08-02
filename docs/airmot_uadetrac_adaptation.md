# AirMOT and UA-DETRAC adaptation guide

This guide documents the complete FDTA adaptation added by the
`feature/airmot-uadetrac-support` branch.

## Included components

| Component | AirMOT | UA-DETRAC |
|---|---|---|
| Dataset loader | `data/airmot.py` | `data/uadetrac.py` |
| Training config | `configs/airmot.yaml` | `configs/uadetrac.yaml` |
| Depth generation | `scripts/generate_airmot_depth.sh` | `scripts/generate_uadetrac_depth.sh` |
| Pretrain conversion | `tools/convert_airmot_detr_pretrain.py` | `tools/convert_uadetrac_detr_pretrain.py` |
| Setup validation | `tools/check_airmot_setup.py` | `tools/check_uadetrac_setup.py` |
| Training launcher | `scripts/train_airmot.sh` | `scripts/train_uadetrac.sh` |
| Inference/evaluation | `submit_and_evaluate_airmot.py` | `submit_and_evaluate_uadetrac.py` |
| TrackEval dataset | `TrackEval/trackeval/datasets/airmot.py` | `TrackEval/trackeval/datasets/uadetrac.py` |
| Smoke test | `tools/smoke_test_airmot.py` | `tools/smoke_test_uadetrac.py` |

## 1. AirMOT

Expected directory layout:

```text
AirMot/
├── train/<sequence>/
│   ├── img1/<numeric frame image>
│   ├── depth/<matching depth image>
│   └── gt/gt.txt
└── val/<sequence>/...
```

Ground truth:

```text
frame_id track_id class_id x y width height flag
```

Class IDs are `0=airplane`, `1=person`, `2=baggage_tug`, and
`3=follow_me_vehicle`.

The AirMOT runtime tracker performs class-consistent association and keeps the
semantic class of an established trajectory stable. This logic is required
because the original FDTA benchmarks are single-class.

### Commands

```shell
DATA_ROOT=/home/zsj/data/datasets bash scripts/generate_airmot_depth.sh
SOURCE_CHECKPOINT=./pretrains/r50_deformable_detr_coco.pth \
OUTPUT_CHECKPOINT=./pretrains/r50_deformable_detr_coco_airmot.pth \
bash scripts/prepare_airmot_pretrain.sh
DATA_ROOT=/home/zsj/data/datasets bash scripts/check_airmot_setup.sh
NUM_GPUS=8 DATA_ROOT=/home/zsj/data/datasets bash scripts/train_airmot.sh
MODEL_PATH=/path/to/checkpoint.pth NUM_GPUS=8 bash scripts/evaluate_airmot.sh
```

## 2. UA-DETRAC conversion

If the local dataset is already stored in MOT layout, skip this section. To
convert the official XML annotations and image folders:

```shell
IMAGES_ROOT=/path/to/official/images \
ANNOTATIONS_ROOT=/path/to/official/xml \
OUTPUT_ROOT=/home/zsj/data/datasets/UA-DETRAC \
SPLIT=train \
LINK_MODE=symlink \
bash scripts/convert_uadetrac.sh
```

The converter:

- parses the official `<frame>`, `<target_list>`, `<target>`, `<box>`, and
  `<attribute>` elements;
- writes standard MOTChallenge ground truth;
- converts `truncation_ratio` to a bounded visibility value;
- creates numeric frame names and `seqinfo.ini`;
- supports `copy`, `symlink`, and `hardlink` image installation.

Run it separately for each split.

## 3. UA-DETRAC layout and formats

Expected converted layout:

```text
UA-DETRAC/
├── train/<sequence>/
│   ├── img1/
│   ├── depth/
│   ├── gt/gt.txt
│   └── seqinfo.ini
└── test/<sequence>/...
```

The loader supports:

```text
# MOTChallenge
frame,id,x,y,width,height,mark,class,visibility[,unused]

# Compact single-class
frame id x y width height [mark]

# Class-first eight-column
frame id class x y width height flag
```

Use one of the following values when automatic detection is ambiguous:

```yaml
UADETRAC_GT_FORMAT: mot       # standard MOTChallenge
UADETRAC_GT_FORMAT: native8   # class-first eight-column
UADETRAC_GT_FORMAT: simple    # single-class compact rows
```

The following options control GT filtering:

```yaml
UADETRAC_FILTER_GT_BY_MARK: True
UADETRAC_MIN_VISIBILITY: 0.0
```

## 4. UA-DETRAC commands

```shell
DATA_ROOT=/home/zsj/data/datasets \
SPLITS="train test" \
bash scripts/generate_uadetrac_depth.sh

SOURCE_CHECKPOINT=./pretrains/r50_deformable_detr_coco.pth \
OUTPUT_CHECKPOINT=./pretrains/r50_deformable_detr_coco_uadetrac.pth \
bash scripts/prepare_uadetrac_pretrain.sh

DATA_ROOT=/home/zsj/data/datasets \
PRETRAIN=./pretrains/r50_deformable_detr_coco_uadetrac.pth \
bash scripts/check_uadetrac_setup.sh

NUM_GPUS=8 \
DATA_ROOT=/home/zsj/data/datasets \
OUTPUTS_DIR=./outputs/fdta_uadetrac \
bash scripts/train_uadetrac.sh

MODEL_PATH=/path/to/checkpoint.pth \
NUM_GPUS=8 \
DATA_ROOT=/home/zsj/data/datasets \
SPLIT=test \
bash scripts/evaluate_uadetrac.sh
```

UA-DETRAC predictions are written as:

```text
frame,id,x,y,width,height,score,-1,-1,-1
```

The TrackEval summary is written to:

```text
<output>/tracker/vehicle_summary.txt
```

## 5. Static and smoke checks

```shell
bash scripts/smoke_test_airmot.sh
bash scripts/smoke_test_uadetrac.sh
python -m py_compile data/airmot.py data/uadetrac.py
```

The GitHub workflow also compiles all adapted Python files and validates all
AirMOT/UA-DETRAC shell scripts.

## 6. Notes

- FDTA requires depth maps during training.
- The UA-DETRAC adaptation is class-agnostic at the vehicle level.
- AirMOT remains class-aware during both inference and evaluation.
- No benchmark score is embedded in the adaptation. Report metrics only after
  training and evaluating on the local dataset split.
