# HS6-L 5TB early/middle/late coexistence diagnostic (v4, single-3090 fleet)

Status: **PLAN FOR USER REVIEW — NO FORMAL DISPATCH BEFORE APPROVAL**.
Requested 2026-09-21. The purpose is to decide whether useful early/middle
features can coexist with late features, not to establish an unbiased optimal
checkpoint or prove that one network can simultaneously store both.

## Locked inputs and comparisons

- Three **no-Gram, same-trajectory EMA teacher** checkpoints under
  `outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_5tb_mix1m03_5tv107_8x5090zxr_20260907`:
  `eval/training_12687/teacher_checkpoint.pth` (E),
  `eval/training_20007/teacher_checkpoint.pth` (M),
  `eval/training_29279/teacher_checkpoint.pth` (L). Shared `config.yaml`.
- E and M were chosen from previous downstream observations; they are
  **retrospective exploratory anchors**. L is the terminal checkpoint, chosen
  without downstream-test maximization. This selection cannot be reused to
  claim a label-free checkpoint-selection method or confirmatory test.
- Same input samples and geometry for E/M/L; compare `E`, `M`, `L`,
  `[E;L]`, `[M;L]`, and `PCA_train([E;L]) -> d`, with no per-task arm selection.
  Each global half uses the protocol's L2-normalized final CLS concatenated
  with final patch mean; concatenate equally weighted halves and L2-normalize.
  For segmentation/detection/tracking use matching spatial maps and their
  unchanged downstream head, never substitute global vectors for patch maps.
  Train the PCA on *training-side* features only. Global d = 2048 for this
  ViT-L readout; dense native-final d = 1024. If fewer than d train examples
  are available, declare `PCA_UNAVAILABLE_INSUFFICIENT_TRAIN_RANK`; do not fit
  on validation/test. For retrieval-only/unsupervised datasets a separate
  fixed pretraining-unlabeled calibration inventory must be identity-locked
  before PCA; no evaluation query/gallery embeddings may fit PCA. PCA coverage
  gaps remain visible; no silent dimension reduction.

## Entire v4 task inventory and protocol

The source of truth is `../protocol_v4.json`: classification 25, regression 4,
retrieval and clustering 7 datasets each (same seven extraction banks), v3-only
segmentation 7 (PanNuke three rotations), matched detection **proxy** 3, native
CTC 1, OOD 2. All 49 unique task-dataset pairs per representation are listed
  in the generated campaign manifest as 56 task-dataset metrics; retrieval and
  clustering share seven extraction/evaluation jobs, yielding 49 distinct
  per-arm dataset jobs, and PanNuke adds two execution cells. Thus the
  six requested representations imply 306 expected dataset-arm execution cells *before*
separate E20/E50 and segmentation seed/fold expansions. Never call an
incomplete component grid a full v4 result.

- Non-dense: BF16, batch 64, workers 2, BLAS threads 1, seed 0,
  auto/TTA8 channels, dataset-best classification/regression resolution,
  retrieval 256 resize/224 crop; locked v3 split/manifests, logistic/ridge,
  cosine retrieval, and KMeans clustering. BBBC013 stays compound-row OOF.
- Dense segmentation: **only v3** native-final/last1 primary, dataset-specific
  geometry and split, feature/probe B32, independent E20 and E50 schedules,
  seeds 0/1/2, every-epoch validation and test-once best-validation readout.
  PanNuke runs all three disjoint train/val/test rotations. Never use the
  historical 8-dataset old segmentation score as a v4 fusion baseline.
- Detection proxy: independent observation lane, BF16 B8, 224 stretch,
  frozen final-patch-map center-to-patch linear head, AdamW 1e-3/1e-4,
  five epochs, seed 0, test patch F1; no native-detection claim.
- OOD: use frozen v3 matched ID/OOD manifests. CTC: native TRA/SEG and fixed
  training folds/head/linker only, not count proxies. Neither is silently
  dropped from the inventory when its evaluator is not ready.

## Readiness and preflight (fail closed per cell)

The existing `retest_20260918` v3 campaign and the separate
`hs6_5tb_protocol_union_nonseg_20260921` and
`hs6_5tb_union_detection_observation_20260921` are *evidence of input/split
readiness*, not reusable fused features or v4 results. The union companion's
LC25000 20000/5000 stratified manifest is frozen but **not source-disjoint
verified**: label any LC25000 classification result `PROVISIONAL_LEGACY_ONLY`,
exclude it from claims of strict full-v4 completion. NCT100 is LOW_N (audit
support/uncertainty). MoNuSeg v3 official identity remains
`BLOCKED_NOT_TESTED`; CTC is `APPROVED_PENDING_FIXED_HEAD_AND_LINKER` in
protocol_v4.json. RxRx3 must use its approved full eligible-gene plate-disjoint
manifest (not 128-gene quick screen). OOD may run only on fixed matched
manifests. Independent ready cells may execute with a visible full inventory,
but no v4 overall aggregate until all gates and PCA eligibility are resolved.

Before GPU dispatch: hash all three teacher files/config, freeze the full v4
rules+source snapshot and numerical environment, and store the base Git SHA
`c95377800b3289f1e85e6565344d2eb158e98656` plus complete dirty-source
SHA256 inventory. Capture dataset/split identities, ordered sample paths and
labels; a fused cell is valid only if E/M/L arrays have identical ordered
identities, labels, dimensions and extraction settings. No historical cache
is accepted without this identity proof. Train/val/test fitting boundaries
are checked separately for probe, scaler and PCA. Save every per-cell command,
host/GPU, environment, feature/config hash, error and independent validation.

## Cluster, resource limits, outputs

3090-qi currently has eight cards around 9.2/24 GiB resident and high
utilization; do **not** preempt its workloads. Read-only inspection showed
potentially idle single-3090 hosts `cpu1`, `cpu2`, `cpu5`, `cpu8`, `cpu9`,
`cpu10`, `cpu11`, `cpu12`, `cpu15` (each must pass a fresh admission check).
`cpu7`, `cpu18`, `cpu20` currently have substantial resident GPU memory;
do not schedule there without fresh proof of enough headroom. All these hosts
read checkpoint and datasets directly on Huawei shared storage, with no
checkpoint, dataset archive or feature-bank transfer. Prefer one process/card
for first measured B64 BF16 and high-resolution dense peaks, then add
co-resident processes only within Rule03 GPU, RAM and NFS caps. Do not lower
batch, geometry or layers for a 24GiB card. Capture start and 30-minute
GPU/PID/RAM/NFS snapshots, yield capacity to other users, and stop scheduling
on protocol error. Existing training remains untouched.

Proposed new output:
`outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921`.
Keep immutable input/source manifests, per-checkpoint feature banks with
sample identities, fused bank provenance, per-dataset/per-arm metrics,
validation reports and paired test predictions where feasible. No deletion,
overwriting of historical results or background process launch before plan
review and preflight PASS. Resource failures are explicit `FAILED_RESOURCE`,
not retries with a changed protocol. Publish all arm-by-dataset scores and
paired confidence intervals before task-family summaries.
