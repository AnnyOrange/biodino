# HS6 resident frozen ready-component proposal

Status: WORKFLOW_APPROVED; launch preflights and per-cell reuse audit pending.
Approval: user requested source-local code synchronization followed by resident
HS6 bulk supplementary tests, and clarified that the audit covers ALL tasks,
not segmentation alone. This document is not evidence of a running campaign.
This ready-component plan does not replace the separately approved
segmentation E20/E50 plan or waive any launch gate.
Existing approved shared 5TB tests and environment repairs continue separately.

## Scope

Use clean published commit 1d2330accebbb91e539b50924db3440fac598e7e.
The prerequisite audit covers all HS0/HS6/FM evaluations, including the
historical 25 classification / 4 regression / 6 retrieval+clustering /
8 segmentation / 3 detection inventory, plus RxRx3, native CTC and OOD.
Audit checkpoint/config/teacher, split and ordered sample identities,
resolution/resize/channels, embedding/layers/dtype, feature versus probe batch,
probe hyperparameters and selection, seeds, code and provenance separately.
Segmentation is an example of a potential mismatch, not the audit boundary.
Matching counts or dataset names alone cannot prove fairness. Missing feature
caches are normal and are not grounds to mark a result invalid.
Keep explicit differences distinct from evidence still awaiting companion
manifest/log verification. Batch-only differences are separately listed under
the user's tolerance; they do not excuse other scientific protocol differences.
Run only genuinely missing or independently invalid frozen components after
the legacy reuse audit, on the machine already holding checkpoint and data.
Do not move checkpoints, datasets, archives or feature banks.

| Machine | Resident HS6 1TB assets | Dataset preflight PASS | Upper-bound cells before reuse audit |
|---|---|---:|---:|
| hxw | B, all 15 retained candidates | 30 | 450 |
| lyx-xr | S+, 9 candidates; L, 12 candidates | 29 | 609 |
| H100 | H+, all 15 retained candidates | 30 | 450 |

The remaining S+ six and L three candidates stay on Huawei shared storage.
Schedule those there under a separately pinned matching campaign; do not
copy them to lyx-xr. Fifteen-candidate trajectories are exploratory, not
permission to choose the largest Test score for confirmatory figures.

Each full resident preflight covers 24 classification, two regression and
five retrieval entries. hxw/H100 admit four retrieval entries; lyx admits
three. RxRx3 is pending its dedicated worker on all three machines.
lyx RxRx1 is pending discovery of a resident official archive; no transfer
of an archive is authorized. The fixed NCT release has 999 samples, not 1000.

## Immutable protocol

Use the dataset registry and official/group splits from Evaluation Rules.
Fixed batch64, BF16, final CLS concatenated with patch mean, L2 normalization,
n_last_blocks1, seed0, workers2, BLAS/OMP1. No sample cap or split fallback.
Use the approved sklearn frozen classifiers/regressors and native retrieval
definitions; do not introduce E20/E50 into sklearn classification probes.
E20/E50 best-validation training belongs to the separate segmentation plan.

Dataset preflight PASS is not full launch readiness. Before dispatch, verify
teacher branch, checkpoint/config hashes and stable stat, sample identities
and group disjointness, source/split hashes, environment, legacy-result reuse
eligibility and resource/storage estimates. Write immutable campaign and
per-cell invocation manifests and independently validated results.
All outputs are independent components, not full-v3 completion or aggregates.

## Resources

Use one machine-wide work-stealing queue, not static task/card stripes.
Synchronize the original lyx `/data/xuzijing/biodino` and H100
`/data_2/suxin/biodino` project directories only after preserving existing code
changes and confirming no process uses the directory. Do not move their outputs
or checkpoints. hxw `/home/xzj/biodino` synchronization and new resident tests
wait until its current training has finished; do not hot-update training code.
hxw and lyx GPUs0--7 are eligible, without interrupting training.
H100 GPUs0--7 are eligible under the approved all-GPU amendment; start with
the cards having sufficient measured memory headroom. Occupied cards wait;
permission does not authorize killing other users' jobs.
Below 60% used, target five total project tests per card after measured admission;
H100 GPU1 targets three. At or above 60%, initially add one and increase only
after stable fixed-batch peak admission, with maximum three. Hard maximum five.
Count evaluator roots, not controllers or DataLoader children.
Measure resident model peaks before stacking, especially H+.
Honor RAM, disk and I/O caps. OOM reduces concurrency, never protocol batch.

## Segmentation exclusion from this proposal

The already approved segmentation budgets remain E20/E50 independent cosine
horizons, validation each epoch, best validation mIoU with earliest tie,
one Test evaluation, seeds0/1/2, probe batch32 and primary last1 features.
The approved extraction batches8/2 are not yet numerically certified against
the Rules reference across the full scope. B ck8199 on 32 official CoNIC
Train samples passed batch8 versus batch32 with max absolute difference0
under commits aa06247 and 1d2330a. This single diagnostic is not a certificate
for other checkpoints/models/datasets.
Full ordered split-identity and legacy reuse audits are
also pending. Do not falsify the required PASS report to start that queue.
BBBC038 stays a separate observation; PanNuke needs all three experiments.

Evidence lives in benchmark_model/fair_plot_20260915/
latest_resident_code_sync_20260917.json and
resident_frozen_dataset_preflight_20260917.json.
