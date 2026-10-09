# HS6-L5 ck29279 official-Gram trajectory comparison

Status: `LOCKED_BEFORE_MAIN_RUN`.

## Question

Does the published within-image Gram anchoring structure preserve the early
segmentation geometry without constraining task families that peak later in
the HS6-L 5TB trajectory?

## Matched training arms

Both arms resume the same full optimizer/model/EMA state at `ck29279` and
consume the same deterministic WebDataset stream.

| Arm | Student start | Patch Gram anchor | Updates | Endpoint |
|---|---:|---:|---:|---:|
| official Gram | ck29279 | ck7807 | 2440 | ck31719 |
| matched control | ck29279 | none | 2440 | ck31719 |

Locked common settings:

- Four deepcad A100 GPUs, per-GPU batch 64, real global batch 256.
- Gradient accumulation 4, effective global batch 1024.
- Student crops 256, clean Gram-teacher crop 512.
- Normalized within-image patch Gram, weight 2.0.
- Gram representative refresh first at update 29480, every 200 updates, at
  most 13 refreshes.
- Full activation checkpointing, one deterministic data worker per rank.
- Teacher and full optimizer checkpoints every 488 updates.
- Endpoints: ck29767, ck30255, ck30743, ck31231, and ck31719.

The control uses the same crop/data/batch/checkpoint protocol. The only active
method difference is the official Gram loss and its frozen anchor.

## Evaluation

Run the frozen full registry on every endpoint. Report classification,
regression, retrieval, clustering, segmentation, and detection separately.
Clustering is derived from the retrieval lane, so retaining it adds no SSL
training or label-selection path. OOD is outside this question.

The baseline `ck29279` result comes from the already running plain-trajectory
evaluator. Every curve comparison must use a common complete checkpoint set.

## Fail-closed gates

1. Require 2440 finite optimizer-update records per arm.
2. Require 2440/2440 identical per-update sample-key digests.
3. Require identical real/effective batch, data, augmentation, optimizer
   schedule, and checkpoint cadence.
4. Require all five endpoint teacher snapshots for both arms.
5. Do not form a cross-task aggregate or use downstream labels to choose the
   anchor, loss weight, or stopping point.
6. A negative result establishes insufficiency only for this published Gram
   structure and anchor protocol, not for all possible Gram objectives.

