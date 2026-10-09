# Old ∪ v3 archive: isolated remaining READY cells

## Question and scope

Complete **only missing, runnable v3 components** for 5TB no-GRAM, FM14 and
HS0 S+/B/L/H+; do not rerun cells already validated in the earlier campaign.
The historical protocol is preserved separately under
`outputs/02_eval_runs/old_v3_protocol_union`; its split/probe results never
substitute for a different v3 cell. `full_v3_aggregate_allowed=false` while
RxRx3, native CTC, and common OOD remain blocked.

5TB + GRAM is intentionally **not admitted**: 26 registered checkpoints
reference a training config whose SHA changed from
`ac0363fed5310d9f23a52526d07cb5a5c395f9c4a127aa9701843ae6a5d6e648`
to `2edaa329e0492b3d22133c7865de5324aee86b4fa17ecb0c20711c927fbe372d`.
Restore the exact earlier config as a read-only immutable copy and investigate
the change before a separate GRAM admission. Do not modify the training config
or registered hashes. HS6 H+ 570 v3 done markers lack their corresponding
cell artifacts in the current H100 location; locate those before aggregation,
not by silently treating their completion markers as result JSON.

## Immutable protocol and selection

For each missing cell the source is the approved `retest_20260918` manifest,
including its exact asset path, config path and registered SHA, teacher branch,
dataset inventory/split SHA, Python/torch/sklearn environment, evaluator source
snapshot, command settings, seed, and feature choice. The isolated campaign
copies only missing `tasks` from that manifest, and records this plan's SHA.
Tier-A and Tier-B inventory, blocked entries, PanNuke three rotations and
CoNIC/LIVECell source-specific splits remain in the parent manifest. Frozen
classification/regression/retrieval: batch64, final CLS+mean patch, seed0;
segmentation: feature/probe batch32, final native map, E20/E50 independently,
every-epoch best validation, seeds0/1/2; dataset-specific resolution, resize,
split and class weights are those expanded in each source task descriptor.
Observation/detection entries are not admitted to the formal campaign.

The source root is `/mnt/huawei_deepcad/dinov3_retest_snapshot_20260918_fm_bound`.
The output and logs are
`/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/old_v3_protocol_union/new_runs`.
Every selected checkpoint/data asset already resides on shared storage;
no checkpoint or data transfer. Link small validated JSON/CSV into the archive
upon completion; do not store duplicate large feature caches outside the
campaign. Skip a cell if and only if its source campaign already contains
VALID_COMPLETE + a readable original validation report/results. Running
source cells are excluded at selection time to avoid simultaneous duplicates.
Retries are at most one per FAILED_RESOURCE cell after resource adjustment;
never change batch, resolution, feature layer, seed or split.

## Machines and safety

Local and 3090-qi each see the same shared checkpoint/data. Start with local
GPU0-3 at most one **new** high-resolution dense task per card and 3090-qi
GPU0-3 at most one new task per card. Read GPU free/used before admission;
their ongoing jobs/training retain priority. The queue's reserve checks limit
memory and total concurrent processes; no fifth GPU on deepcad is requested.
High-resolution segmentation is single-job initially; expand only after a
measured stable peak as in Rule03. Any new provenance/preflight failure pauses
this isolated campaign. Do not unpause or alter the original campaign's
`_state/PAUSED.json` while its GRAM config mismatch remains unresolved.
