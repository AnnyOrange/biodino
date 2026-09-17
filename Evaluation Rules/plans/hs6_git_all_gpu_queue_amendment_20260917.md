# Git-pinned resident segmentation queue correction

Status: code synchronization and GPU eligibility APPROVED by the user on
2026-09-17; scheduling implementation is prepared, launch remains gated by
Evaluation Rules preflight. The latest instruction is to synchronize first,
then apply the rules without rushing into additional tests. The user explicitly
permits every GPU on lyx-xr and H100.

## Scope and immutable protocol

Continue the already approved HS6 1TB S+/B/L/H+ fifteen-checkpoint trajectory
and HS0-L/L5 pilot completion. This correction does not claim completion of
the full classification/regression/retrieval/detection/FM matrix.
Keep resident checkpoints and dataset roots from the placement catalog.
No checkpoint, dataset or feature-bank transfers. Do not change splits,
resolution, last1 features, AdamW settings, E20/E50 independent horizons,
validation-every-epoch, best-validation selection, seeds 0/1/2, or probe batch32.
Extraction batch remains the previously recorded S+/B/H+=8 and L=2.
The pending strict extraction-batch certificate and sample-identity audit
remain pending: do not launch further evaluations until the required gates
pass. BBBC038 remains a separately reported observation.

## Synchronization and provenance

Stop the old detached queue trees, preserving outputs and unrelated jobs.
Commit only the relevant evaluator, runners, tests and approved plans;
preserve unrelated dirty user changes. Push to GitHub origin. Create clean
independent checkouts on all four execution machines, fetch origin and checkout
one identical SHA. Configs live outside the checkout. Preflight checks Git SHA,
tracked cleanliness, frozen code and split hashes, CUDA and resident paths.

New output roots: existing run parent plus `git_campaign_20260917/results`.
New cache roots: existing run parent plus `git_campaign_20260917/cache`.
Old outputs remain at the previous roots, recorded as legacy and not silently
reused or attributed to the new commit. Audit reuse eligibility first; do not
schedule a blanket re-run merely because Git provenance was missing. The
proposed fresh output root is reserved for genuinely missing/invalid cells.
Resumes reuse only validated results
from the identical commit/checkpoint/config/result hashes in the new root.

Each new output root gets campaign_manifest.json with commit/status, hostname,
GPU set, Python/torch/sklearn versions, checkpoint/config paths and hashes,
dataset/split hashes, explicit commands and environment overrides. Independent
per-dataset validation reports check all six fits (PanNuke eighteen); report
compute-schema validity separately from formal reportability.

## Machine-wide resource scheduling

lyx-xr and H100: eligible GPUs0--7; hxw: eligible GPUs0--7, without disturbing
its training; 3090-qi: existing shared GPUs0/1/2/4 only.
All resident jobs enter one machine-wide pool, not static per-GPU stripes.
On low-memory cards admit up to three tests; a >=60%-occupied card gets only
one additional test after the memory headroom check. Max three is below the
global rule's hard maximum five. MoNuSeg768 remains single-job peak probing;
never stack speculative encoder extractions on one card. Additional tasks
may stack once an existing task reaches a cached-probe phase.

All processes use num_workers2 and BLAS/OMP threads1. Re-sample resources every
five seconds before each admission. Respect host job caps, RAM and free-disk
reserves; a free GPU does not override RAM or disk checks. Record resources
at start and at least every thirty minutes. Occupied H100 GPUs4--7 may wait
for sufficient encoder memory; permission does not authorize killing training.
Failures pause the machine-wide queue and mark the failed cell invalid;
no batch/resolution/layer reductions and no output deletion on OOM.

Deploy writes per-host configs with concrete reserves, fixed commit and paths;
launch records and live snapshots are kept with the inventory artifacts.
