# FDTA Efficiency Analysis

The FDTA efficiency benchmark is intentionally separated from normal submission/evaluation. It follows the same fixed protocol used for the other MOT baselines in this project.

## Fixed protocol

- Batch size: **1**.
- Device: **single NVIDIA L40S GPU** by default.
- Mode: `eval()` / inference.
- Warm-up: **100 frames**.
- Latency: synchronized end-to-end online latency over the **complete evaluation split**.
- Timed scope: `RuntimeTracker.update()` only.
- Included: detector forward, FDTA trajectory embedding/modeling, ID decoder/assignment, and online trajectory-state/postprocessing.
- Excluded: image loading/decoding, resize/normalization, CPU-to-GPU transfer, visualization, result serialization/file I/O, and TrackEval.
- CUDA synchronization is performed immediately before and after each timed frame.
- `FPS = 1000 / mean_latency_ms`.
- Params: total model parameters.
- FLOPs: average PyTorch-profiler FLOPs over uniformly sampled, stateful online frames. Stateful sampling is required because the number of active track hypotheses changes over time.
- Precision and input resize policy must remain identical across compared models.

The primary efficiency table should report:

| Method | Params (M) | FLOPs (G) | Latency (ms) | FPS |
|---|---:|---:|---:|---:|

`Latency (ms)` is the synchronized wall-clock latency and is the primary speed metric. Optional GPU-event timing is diagnostic only.

> Note: PyTorch profiler may under-count custom CUDA operators such as deformable-attention kernels. Therefore the FLOP value is stored together with its sampling protocol and should only be compared with results produced using the same profiler rule.

## AirMOT

```bash
cd /data/zsj/workspace/multi-object_tracking/baseline_FDTA

MODEL_PATH=./outputs/fdta_airmot/checkpoint_12.pth \
GPU_ID=0 \
DATA_ROOT=/home/zsj/data/datasets \
SPLIT=val \
PRECISION=FP32 \
WARMUP_FRAMES=100 \
FLOPS_SAMPLES=20 \
bash scripts/analyze_efficiency_airmot.sh
```

Outputs:

```text
outputs/efficiency/fdta_airmot/
├── efficiency_summary.txt
├── efficiency_summary.json
└── flops_samples.csv
```

## UA-DETRAC

```bash
MODEL_PATH=./outputs/fdta_uadetrac/checkpoint_12.pth \
GPU_ID=0 \
DATA_ROOT=/home/zsj/data/datasets \
SPLIT=test \
PRECISION=FP32 \
WARMUP_FRAMES=100 \
FLOPS_SAMPLES=20 \
bash scripts/analyze_efficiency_uadetrac.sh
```

## Non-L40S debugging

The benchmark rejects a non-L40S GPU by default so an accidentally incomparable number cannot enter the paper table. For code debugging only:

```bash
ALLOW_NON_L40S=1 \
MODEL_PATH=/path/to/checkpoint.pth \
bash scripts/analyze_efficiency_airmot.sh
```

Results obtained with `ALLOW_NON_L40S=1` should not be mixed with the fixed L40S comparison table.
