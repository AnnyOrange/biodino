# HS6 H+ frozen-v3 result artifact recovery

Status: APPROVED by the user's explicit H100 GPUs 0–3 rerun instruction.

The original 2026-09-18 frozen-v3 campaign has 570 H+ VALID_COMPLETE markers
but no corresponding `cells/` directory or recoverable result JSON on H100.
Markers alone are not scores. Rerun the exact 15 resident teacher checkpoints ×
38 originally admitted READY task/split components (450 nondense, 120 dense),
without relabeling legacy results or expanding blocked tasks. This is a
recovery of frozen v3 components, not a new v4 aggregate; Rule09 explicitly
does not retroactively alter previously approved v3 task protocols. The
unchanged 2026-09-18 `source_snapshot.json`, v3 registry, dataset manifests,
checkpoint/config identities, numerical environment, and executable must
pass their original recorded SHA/version gates before rerunning. Also retain
the current Evaluation Rules including Rule09/protocol_v4 as a separately
hashed policy copy; do not change the pinned v3 evaluator mid-campaign.

Source evidence: H100 original campaign
`/data_2/suxin/output_checkpoints/evaluations/retest_20260918` (read-only);
snapshot `/data_2/suxin/biodino/worktrees/biodino_retest_20260918`;
15 teacher checkpoints under resident `/data_2/suxin/runs/HS6_Hplus_...`.
The missing numerical files are not restored from SHA-only done markers.
Output root `/data_2/suxin/output_checkpoints/evaluations/hs6_hplus_v3_artifact_recovery`.
Copy only the original manifest and dataset-preflight evidence into this new
root; keep original output and all 570 original markers unchanged. New jobs
write only to the new root, with independent invocation/validation reports.

The H100 data relocation moved source data from `/data_2/suxin/{Classification,
Regression,Retrieval_Clustering,segmentation,ood}` into the identically named
directories under `/data_2/suxin/test-dataset/`. Restore missing legacy paths
using symlinks only after confirming the old links do not exist, then verify
every original source SHA, split identity and count. No datasets, checkpoints,
heads, or feature caches are copied across machines.

GPU allocation after initial memory measurements: GPU0 target 3 tests (other
user reserves about 27 GiB), GPU1 starts with 1 test (other user reserves
about 54 GiB; never force 3–5 on an occupied card), GPUs2/3 target 4 tests
each. A dedicated lane per GPU shares one atomic claim queue, with a hard
maximum of 5 tests/card. Reassess measured peaks and other users' processes
before any increase. Dense segmentation starts at one per GPU, holds frozen
feature/probe batch32, E20+E50, every-epoch validation, seeds0/1/2 and all
PanNuke rotations. Nondense feature batch64, BF16, final CLS+patch mean,
seed0 and workers2 are unchanged. Never lower batch, geometry or layer counts
for concurrency. Preserve heads and numerical JSON; only validated result
files enter completed status. Record per-card PIDs, errors and OOM separately.
