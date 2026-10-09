# Two 20TB raw-intensity online comparisons

Status: APPROVED by explicit user instruction on 2026-09-18.

Compare default projection and cosine_v2 from identical official ViT-L
initialization. Three input channels; uint16/65535 then batch-specific first-three
Mean/Std from 20TB_projection_two_batches_mean_std.md. Training B32 x4 GPUs
x accumulation8 =1024; LR1e-4; schedule10 reference epochs, LR warmup0.5,
teacher temperature warmup1. Keep all 488-update teacher/optimizer checkpoints.
Both use reference epoch length35374, equal optimizer-update budgets, not equal
physical corpus traversals. Fresh output roots; preserve old experiments.

Online evaluation reads EMA teacher checkpoints ONLY after two stable observations
and age120seconds, records checkpoint/config SHA256,size,mtime and teacher key.
Complete Tier A/B inventory from protocol_v3.json must remain in manifest.
Only independently READY components may run; native CTC, RxRx3, OOD and MoNuSeg
remain BLOCKED until their own provenance/acceptance validators pass.
full_v3_aggregate_allowed=false. No old scores/completion markers/features reused.

Frozen tasks B64, BF16, final CLS||patch mean then L2; seed0, channelauto/TTA8,
dataset-best resolutions; StandardScaler+balanced LR(max_iter10000,n_jobs1),
Ridgealpha1; BBBC013 compound/log1p OOF. Splits/hashes/counts locked by reused
immutable preflight identities, reverified before each execution. Dense primary
last1; B32 feature/probe, E20/E50 independent horizons, seeds0/1/2, AdamW
LR1e-3/WD1e-4, validation every epoch, earliest best val mIoU, test once.
PanNuke expands three disjoint folds; CoNIC source-grouped; LIVECell official
annotation hashes. Dataset-best/even4 is supplementary, not mixed into primary.

Temporary explicit machine exception: deepcad may occupy at most SIX distinct
project GPUs, indices0-5. GPUs6-7 excluded. This overrides Rule03 four-card limit
ONLY for this user-approved campaign, including concurrent project workloads.
Shared single-3090 nodes are authorized; discover available nodes independently.
Target total3 actual test processes per card, not three additional processes.
At>=60% memory add one unmeasured job at a time; dense initially exclusive1.
Memory/CPU/NFS gates override concurrency target. Never lower evaluation batch,
resolution/layers, or stop unrelated work. Workers2, BLAS1. Resource snapshots
recorded at launch and at least every30minutes. No checkpoint/data transfers.

2026-09-19 user-approved local amendment: on the local eight RTX 5090 cards,
frozen-task concurrency may target FOUR total test processes per card. Dense
segmentation remains exclusive1 because its recorded 8/16GiB reservation and the
resident training process override this ceiling. Evaluation batch sizes and all
numerical/protocol settings remain unchanged.

Create independent immutable source snapshot from admitted retest source, with
the deepcad authorization gate recorded in ALL-source SHA256 inventory. Exact
Python/torch/numpy/sklearn etc must match numerical environment manifest.
Per-cell command/environment/input/result provenance and independent validator
required before VALID_COMPLETE. Failures isolated and marked FAILED; no silent
zero scores. Invalid cells cannot appear in curves. Do not retry changed protocol
in-place. Daily curves per dataset/metric with raw CSV; no incomplete overall
aggregate. Training loss and eval results stored separately; refresh60seconds.

Output: outputs/02_eval_runs/20tb_raw_online_20260918.
Log/curve: output/logs and output/curves. Config/checkpoint roots in training_roots.json.
Execution commit/source/rules/dependencies, counts, geometry and expanded probe
settings recorded in campaign_manifest.json and cells/*/invocation_manifest.json.

2026-09-20 user-approved observation amendment: retain the v3 formal detection
count at zero, and add a separate three-dataset detection observation companion
for BBBC038, CoNIC and LIVECell. Report coverage as formal
`24/2/4/4/6/0 + Det-Obs3` (or `24/2/4/4/6/3*`, with the asterisk defined as
observation-only). Never include these proxy scores in Tier A, Tier B, overall
means or main ranking. Run both the retained 5TB no-GRAM checkpoints and all
four 20TB arms under one matched observation protocol: 224 stretch, final
spatial patch map, batch8, BF16, workers2, seed0, fixed center-patch linear
head, AdamW lr1e-3/wd1e-4, five epochs, and test patch F1. The existing 5TB
batch4 results remain historical only and cannot be joined to the matched B8
curves. Use a new immutable observation campaign/output root; do not append
these cells to the formal 20TB campaign manifest.
