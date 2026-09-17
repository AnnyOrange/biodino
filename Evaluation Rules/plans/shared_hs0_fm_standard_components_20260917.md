# Shared HS0 / FM standard components

Status: APPROVED_WORKFLOW_BY_USER_20260917; per-cell preflights remain mandatory.
The user explicitly requested concurrent FM and HS0 S+--H+ supplementary tests
on shared machines, including repaired single-GPU cpui nodes, targeting five
tests per low-occupancy card. hxw/lyx/H100 original-directory sync is deferred.

## Scientific scope

HS0 S+/B/L/H+ use the predeclared common ck8199, teacher branch. All four
weights/configs remain in their original shared training outputs. H+ uses its
complete eight-shard DCP directly; no consolidated copy or weight transfer.
FM14 published weights remain under benchmark_model. Retrospective per-task
Test maximization is forbidden. This is a separate fixed-SHA component campaign,
not an update to the active shared 5TB campaign or permission for a full-v3 mean.

Audit all historical 25/4/6/8/3 plus RxRx3/CTC/OOD. Current ready dispatch covers
24 classification, two regression and four generic retrieval datasets, subject
to individual preflight. Formal dataset eligibility is not a claim of a full
suite. RxRx3 dedicated integration, seven formal segmentation datasets,
independent BBBC038 observation, native CTC and OOD remain explicit pending rows.
Their existing approved protocols are unchanged, including segmentation
E20/E50 independent horizons and per-epoch best-Val selection, seeds0/1/2.

FM first ready token adapters: dinov2, mae, phikon2, uni, virchow2, gigapath,
hoptimus0. They use actual final CLS concatenated with final patch mean,
excluding registers/prefix tokens, L2 normalized. MAE masking is disabled.
Native published normalization is retained; the requested dataset crop/resize
is applied before the nearest legal patch grid. Requested and actual encoder
dimensions are both recorded; positional embeddings must interpolate, never
silently revert to a native 224 crop. The other seven FMs remain blocked pending
strict readout/input adapters: no-CLS/convolutional models need an explicit
architecture-equivalent protocol; native CLS-only results cannot be renamed.
Failures pause dispatch and are retained, not hidden by reducing batch.

## Fixed component parameters and proof

Feature batch64, BF16, last block, CLS||patch-mean L2, seed0, channel policy auto,
channel_tta_samples8, channel_policy_seed0, workers2, BLAS/OMP1. Dataset-best
crop/resize from Rules; ordinary224/256, blood/MIDOG384/439, BBBC048/chest512/585.
Canonical StandardScaler/balanced LogisticRegression max_iter10000,n_jobs1;
Ridge alpha1; BBBC013 compound/log1p eight-fold OOF. No caps/random fallbacks.
Validate official/group/source identities, leakage, labels and fixed hashes.
No unverified reuse. Old results remain; reconstructible feature banks are not
persisted. Exact old companion-evidence matches can later be certified separately.
Record SHA, clean status, all asset/config hashes/stats, dataset/source/split
fingerprints, dependency versions, commands/env, actual geometry and finite
metrics. Use independent cell validation, not exit0 alone. External shared
loader sources outside Git are pinned by SHA and rejected if modified.

## Placement and resource amendment

Output: outputs/02_eval_runs/shared_hs0_fm_standard_20260917.
Use a new clean published checkout, never hot-update active 5TB code.
Eligible: local/3090-qi GPU0--7; repaired/ready cpu1,2,5,7,8,9,10,11,12,15,18,20
GPU0. No new deepcad jobs: its four allowed cards already host training with
very little headroom. cpu19 stays occupied/excluded; other nodes require fresh
SSH/CUDA/BF16/import proof before admission. All jobs read shared assets directly.

Target five TOTAL evaluator roots/card, including existing campaigns. On cards
>=60% occupied, add one initially, at most three after stable peaks; no claim
that occupied cards can have five. Include pending encoder memory reservations,
host RAM>=32GiB, output disk>=64GiB, conservative4--18GiB+ model reserves.
GPU assignment favors less loaded cards, not static experiment/card stripes.
New campaign cap40 concurrent jobs; existing 5TB cap24 stays unchanged. This
is an upper admission bound, not a demand to start64 jobs regardless of IO.
Resource snapshots at least every30min; failures pause new admission. Do not
kill other jobs, change protocol batch/size/layers, transfer assets or delete
material outputs. Actual per-card counts, not controller counts, are reported.
