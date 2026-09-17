# Shared frozen stage 1

APPROVED by the user on 2026-09-17: independently run the ready standard
classification/regression/retrieval components now; complete CTC and other
components separately. This is not a full-v3 campaign and must not produce
any full-v3 aggregate. No change to `protocol_v3.json` or its full-suite gates.

Grid: all actual retained original HS6-L 5TB no-Gram teacher checkpoints and
ck12687 official-Gram teacher checkpoints, directly read on Huawei shared
storage. The approved asset CSV is under
`/mnt/huawei_deepcad/benchmark_model/fair_plot_20260915/`.
Initially prioritize matched ck18543 plus no-Gram ck26351/29279 and the common
ck12687 anchor; then cover all retained points. Update budget matching is
required for a Gram/no-Gram ablation; unequal points remain separate curve data.

Dataset grid: all 24 classification and 2 regression datasets from Tier A/B.
The four currently generic-CLI-supported retrieval datasets are nct-crc-he-1k,
crc-val-he-7k, hpa-subcellular, rxrx1-cross. RxRx3-core is explicitly pending its
dedicated-worker assignment, not omitted from the overall remaining-work table.
Segmentation/CTC/OOD are separate later components; BBBC038 stays observation.
Each dataset must independently pass its full sample-identity/count/split/label
preflight before that dataset's component jobs are admitted. A blocked dataset
does not invalidate the independent preflights of already-ready datasets.

Protocol: teacher, final CLS concatenated with final patch mean, L2 normalized,
n_last_blocks=1, avgpool enabled, BF16, frozen batch explicitly 64, seed 0,
dataset-best crop/resize (retrieval 224/256), channel policy auto, workers 2,
BLAS 1. Classification StandardScaler/balanced LogisticRegression max_iter10000;
regression StandardScaler/Ridge alpha1; BBBC013 compound/log1p OOF special case.
Official train/test are not merged with validation. Group splits are the
committed hashes from Rules/02, with group leakage checked. Retrieval uses
fixed within-set or independent query/gallery/core manifests.

No old result/cache reuse. New output:
`/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/shared_frozen_stage1_20260917`.
Keep small manifests/JSON/CSV/logs. Do not persist rebuildable feature banks;
do not delete old material files or transfer weights/data between machines.
One shared missing-cell queue with atomic claims across all shared hosts.
Skipped results require an independent validation report and exact fingerprints.
Do not mark complete merely because a process exited zero. Independently checked
cell metadata/counts/finite metrics are required; component completion never
means full-v3 completion or authorizes a global ranking.

Placement: shared local/deepcad/3090-qi/cpu nodes only, clean same GitHub commit.
Every admitted job records code/dataset/split/checkpoint/config hashes and
size/mtime, host/GPU, versions, commands/environment and GPU resource snapshots.
CPU nodes must pass a real modern evaluator/CUDA/BF16/import check, not only a
CPU unit test. Torch<2 nodes wait for an evaluation environment rather than
having their existing environment changed in place.

Scheduling: target 3 low-memory tests/card; hard max5 across all our evaluator
roots (not controllers/loader children). >=60% occupied cards first add1;
only measured stable peaks permit up to3. Existing stage evaluators count
towards admission. Workers2 and BLAS1; host RAM/disk/NFS gates are mandatory.
The new shared component queue is capped at 24 concurrent jobs across hosts;
this is an initial NFS admission limit, not a claim that all GPU slots are full.
High-resolution batch64 jobs require a conservative memory reserve and do not
silently become batch32/16. Deepcad is restricted to GPUs0-3 while Gram trains;
on any card insufficient resources mean wait, not an extra project card.
Resource and project-PID snapshots at startup and every30min. Stop new dispatch
on protocol error, retain the offending cell, no automatic retry (max1 attempt).

All modified evaluator/queue code is tested, committed, pushed to origin, then
fetched/checked out at the same SHA on each execution host before starting.
Each machine's startup manifest proves its clean commit and dependency versions.
Unreachable/unauthenticated/busy/incompatible nodes remain explicit in the
node-readiness report. Existing training and other users' GPU jobs are preserved.
