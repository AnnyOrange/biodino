# Full fleet retest execution amendment

Status: APPROVED by explicit user launch instruction on 2026-09-18.

Local /mnt/huawei_deepcad/dinov3 and /mnt/huawei_deepcad/benchmark_model are
the development sources. Copy code and the complete Evaluation Rules directory
into new independent execution snapshots, retaining the base Git commit and
SHA256 of every execution source. Do not change active training repositories.
Every host verifies these hashes and records the actual dependency fingerprint
before claiming work. No checkpoint, image archive or feature bank is copied.

Models: FM14 published encoders; HS0 S+/B/L/H+ ck8199; all retained registered
5TB no-GRAM teacher candidates; all HS6 S+/B/L/H+ resident candidates; existing
and subsequently stable GRAM and 20TB consolidated teacher candidates.
Exact checkpoint/config paths and expected task specs are in each immutable
campaign manifest. Online additions have an append-only checkpoint admission
journal; the rules/source fingerprint cannot change within a campaign.

Expected full inventory is protocol_v3 Tier A AND B (41 task-dataset pairs per
checkpoint). READY classification/regression/retrieval and segmentation cells
may start as independent components. Missing official provenance/adapter
acceptance is NOT TESTED, not zero or completed. No overall/ranking aggregate
is emitted before every expected task is admitted and validated.

Fixed frozen tasks: B64, BF16, final CLS concatenated with patch mean, L2,
dataset-best classification geometry, retrieval resize256/crop224, seed0,
workers2, BLAS1, teacher/EMA. Dense primary: last1/native-final map, locked
dataset geometry/split/class weights, feature B32/probe B32; independent
E20/E50 cosine schedules, seeds0/1/2, AdamW lr1e-3/wd1e-4/dropout0.1,
validation EVERY epoch, earliest best validation mIoU, test exactly once.
Dataset-best/even4 is supplementary, never merged with primary-last.

Placement: shared checkpoints stay on Huawei; use qi GPUs0-7 and available
single-3090 nodes with target5/card, local GPUs0-7 target2/card. HS6 B stays
hxw, S+/L stay lyx, H+ stays H100. All eight cards are authorized by the user
for this campaign, but other users' processes and active training are never
terminated. Target5/card is a maximum/target, not permission to cause OOM.
Memory reservation and measured peak override target; dense initially1/card.
deepcad remains restricted to GPUs0-3 if used, never more than four distinct
project GPUs. Do not reduce protocol batch/geometry/layers to fill a card.

MoNuSeg is blocked until official train30/test14 sample IDs and original
image/XML fingerprints are proven. The new official-manifest loader uses a
separate ID manifest and seed42 train24/val6 split; it does NOT mutate the old
37-pool index file or delete any original image. The challenged 37-pool scores
remain STALE. Official challenge URL returns403 at review; paper confirms
30-image training pool: https://pmc.ncbi.nlm.nih.gov/articles/PMC10439521/ .
RxRx3 FM integration, CTC native decoder/linker/all-five-fold acceptance,
OOD common input/config freeze, and BBBC038 detection remain NOT TESTED
until their validators pass. BBBC038 is only a separate observation.

Use fresh output roots suffixed retest_20260918; no old marker/cache reuse.
Record checkpoint/config hashes, source/rules/dependencies, command, GPU,
split identities/counts and result SHA256. Only post-validated files enter
VALID_COMPLETE. OOM is FAILED_RESOURCE, never a secretly lower-batch rerun.
The five-hour deadline is recorded for progress/ETA, not a reason to cap
samples, drop folds, omit seeds or label pending tasks complete. Training is
ongoing, so future checkpoints cannot all finish in a finite five-hour window.
