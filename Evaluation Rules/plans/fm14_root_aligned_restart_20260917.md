# FM14 root-aligned restart

Authorization: user requested direct modification of benchmark_model, audit and
restart on 2026-09-17. This supplements the approved shared HS0/FM campaign.

Implementation is in `/mnt/huawei_deepcad/benchmark_model`: `run_fm_rules.py`,
`run_fm_dense_rules.py`, `run_fm_rules_campaign.py`, and `benchmark_eval/rules_features.py`.
Use a clean fixed DINOv3 Git checkout for shared dataset/probe code; track SHA256
of every external evaluator file. Commit/push locally before starting workers.
Do not change code used by active old training/HS0/5TB campaigns.

All 14 published FM assets are registered: dinov2, mae, siglip2, pe, bioclip,
cytoself, jump_cp, cytoimagenet, uni, conch, phikon2, virchow2, gigapath, hoptimus0.
Use their existing Huawei directories and published weights; no transfers.

Frozen component matrix: all 24 official/grouped classification datasets, BBBC005
and compound-OOF BBBC013, and the four generic retrieval/clustering datasets from
Tier A+B. Feature batch64, BF16, workers2, BLAS1, seed0, final-layer readout,
dataset-best resolution and canonical probe. True multichannel tensors stay
float and use the same bilinear antialiased geometry as HS0. Models without CLS
declare native pooling instead of fabricating a CLS token.

Dense matrix: Cellpose, CoNIC, LIVECell, MoNuSeg, TissueNet, Multimodal and all
three official PanNuke rotations. Shared probe, feature/probe batch32, BF16,
workers2, independent E20/E50 horizons, seeds0/1/2, validation every epoch,
earliest best validation mIoU, test once. Primary all-last and dataset-best are
separate comparison views; dataset-best uses even4 where supported, else last.
Reuse primary-last results for best-view datasets that already request last
without extracting or testing twice, with explicit provenance.

Do not claim this independent-component campaign completes full-v3. RxRx3,
native CTC, locked OOD and BBBC038 observation are separately inventoried and
must not be replaced by historical proxy scores. BBBC038 is not in this formal
component manifest.

Batch-only differences do not trigger reruns. Reuse old VALID_COMPLETE cells
only after checking dataset/split, metric finiteness, checkpoint identity and
readout parity. Old tensor->uint8 PIL features differ from current native float
input and require targeted supplementation. Missing companion evidence remains
review, not automatic rerun. Preserve every old result and failed log.

Execution pool: shared 3090-qi and environment-ready single-card cpu nodes.
Local CUDA preflight currently fails; record and skip it without GPU reset.
Do not use deepcad beyond GPU0-3; its active training remains untouched.
Independent hxw/lyx/H100 training/sync is outside this restart.

Target at most five actual formal tests per GPU. For >=60% occupied cards add
one first, maximum three; reserve measured peak plus margin, not lower batch.
Dense extraction initially requires an otherwise evaluation-idle GPU; MoNuSeg
768 remains single-test peak probing. Low-memory frozen jobs fill other cards.
Global cap80 new jobs limits NFS/RAM; minimum available RAM32 GiB and storage64
GiB, with higher dense reservations. Capture GPU/RAM snapshots at startup and
every30minutes. Jobs get the next gap from one shared claim queue.

Output: benchmark_model/benchmark_runs/fm14_rules_aligned_20260917.
Per-cell invocation, checkpoint/config hashes, frozen split identities or
dense split/hash audit, actual commands/environment and independent validation
reports are mandatory. Never overwrite an existing result to resume.
OOM marks FAILED_RESOURCE, keeps logs and blocks that cell from automatic retry;
do not lower protocol batch or silently claim completion. Other errors pause
the queue for correction. Maximum one attempt per fixed code fingerprint.
