# Shared 5TB standard retest and resident-model deployment

Scope requested by the user on 2026-09-17: synchronize GitHub code first, then
supplement the original 5TB no-Gram trajectory and the ck12687 official-Gram
trajectory on local/deepcad/3090-qi/single-card cpu hosts. Other resident HS6
checkpoints stay on hxw/lyx/H100. No checkpoint/data/feature-bank transfers.
This execution plan does not authorize bypassing any preflight failure.

## Comparison grid and assets

- No-Gram run: `outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_5tb_mix1m03_5tv107_8x5090zxr_20260907`.
- Gram run: `outputs/01_training_runs/HS6_L5_ck12687_official_gram_a12687_b32_gb1024_noac_4xdeepcad_u2440_contract_v2_20260915`.
- Use each run's `config.yaml` and actual retained teacher checkpoints under
  `eval/training_<update>/teacher_checkpoint.pth`; a `ckpt/<update>/checkpoint.pth`
  fallback is eligible only after explicitly validating its teacher branch.
- Inventory all retained files, not just the eight recent full no-Gram saves.
  Compare shared update IDs first, include the common ck12687 anchor, and retain
  the remaining trajectory points. Do not compare mismatched training budgets
  as a matched Gram/no-Gram ablation.
- Authoritative asset table:
  `/mnt/huawei_deepcad/benchmark_model/fair_plot_20260915/5tb_main_gram_checkpoint_assets_20260917.csv`.
- Resident 1TB asset queues and HS0/FM inventories remain in
  `/mnt/huawei_deepcad/benchmark_model/fair_plot_20260915/`.
  Their legacy values are not evidence that formal-v3 is complete.

## Protocol

Read the entire `Evaluation Rules` directory before launching. Use the complete
Tier-A and Tier-B matrices from `protocol_v3.json`: 24 classification datasets,
2 regression datasets, 5 retrieval datasets, 7 segmentation datasets, native CTC,
and 2 OOD tasks. Retrieval/clustering use the same locked dataset manifests.
Historical 25/4/6/8/3 coverage is reported separately, not silently discarded or
counted as formal completion. BBBC038 needs its own observation plan/root.

- Frozen batch 64, BF16, teacher final CLS concatenated with final patch mean,
  L2 normalization, dataset-best resolution, channel policy auto, seed 0.
- Segmentation feature/probe batch 32/32, BF16 encoder, dataset-best resolution,
  resize, layers and weights exactly as specified in `protocol_v3.json`.
- CoNIC 3469/494/1018 source-disjoint; LIVECell original official COCO records
  3253/570/1564, including the documented official duplicates; PanNuke all three
  locked train/validation/test rotations. Never reuse legacy PanNuke val=test.
- Standard segmentation remains E50/eval-every-50 under the current rules.
  The separately approved E20/E50, eval-every-epoch, best-validation, 3-seed
  embedding-budget experiment is a distinct protocol/output. Do not silently
  merge its scores or change the standard rules to claim a stronger comparison.
- CTC uses native TRA/SEG and all 20 domains/five folds, not count proxies.
- Workers 2, BLAS threads 1. No sample caps, smoke results, or automatic
  batch/resolution/layer reductions in formal output.

## Placement and admission

| Host | Weights | Planned admission |
|---|---|---|
| local | Shared no-Gram/Gram and resident HS0/HS6/FM | All eligible cards; when training occupies >=60%, add at most one low-memory test initially |
| deepcad | Same shared weights | Only GPUs 0-3 while the four-rank Gram training is active; never GPU 4-7 |
| 3090-qi | Same shared weights | On eligible low-memory cards target 3 tests/card, scale to 5 only after measured peaks and host RAM/NFS checks |
| single-card cpu hosts | Same shared weights | Same 3-5 low-memory tests/card rule; count other project test processes, not just this queue's children |
| hxw/lyx/H100 | Already-resident HS6 checkpoints/data only | Clean same-SHA checkout, all eligible cards, same memory-based admission |

Use one missing-cell queue across the shared group. Claim/checkpoint readiness
must be locked across hosts; a completed task immediately frees a slot for the
next missing cell. Controller count is not test concurrency. Count actual tests
and their GPU bindings. Respect all other users' workloads.
MoNuSeg 768 and other dense jobs first measure batch-32 peak with one process;
add co-residents only after the measured GPU/RAM peak supports them. If the
protocol does not fit, wait for a larger eligible card; do not lower batch.
Record startup and 30-minute GPU/PID/host RAM/NFS/storage snapshots; shrink when
other users arrive. Failures pause dispatch immediately; maximum one formal
attempt before diagnosis, not repeated automatic retries of protocol errors.

## Code, outputs, completion

- Origin is the source of truth. Test and push the evaluator fixes, then fetch
  and detach every execution host at that single SHA in isolated checkouts.
- Shared checkout: `/mnt/huawei_deepcad/dinov3_eval_git_20260917`.
  Independent machine checkout paths are in `code_sync_verified_20260917.json`.
- Proposed NEW output:
  `/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/shared_5tb_standard_retest_20260917`.
  Never overwrite or weak-skip legacy result files.
- Before launch, record checkpoint/config SHA256, size/mtime, teacher key,
  all dataset/split hashes/identity/count/leakage checks, command matrix,
  fixed commit, clean status, Python/torch/sklearn versions and environment.
  A BF16 batch-invariance certificate and per-resolution resource peak are
  required for cache/result reuse and scheduling.
- The approved preflight must PASS including CTC readiness. A dedicated RxRx3
  evaluator must be assigned explicitly; it is not present in the generic
  retrieval CLI. A launch cannot silently omit either task.
- Only independent per-cell `validation_report.json` plus complete matching
  provenance/counts/finite metrics permits `VALID_COMPLETE`. Old results and
  incomplete provenance remain separately visible, never included in formal
  aggregate. No automatic output/cache/checkpoint deletion.

## Current blockers (audit, not approval to bypass)

Before formal launch, the batch-invariance diagnostic may run on a clean pinned
checkout using the first 64 official-train PathMNIST samples, seed 0, BF16,
224/256, final CLS concatenated with patch mean, comparing batches 1 and 64.
Prespecified screening tolerances: minimum cosine 0.9999 and maximum relative
L2 0.02, all features finite. Results are GPU_PREFLIGHT_ONLY, never formal task
scores. A pass covers only that checkpoint/input path, not dense/high-resolution
or multichannel paths. Start up to 3 diagnostics/card only while free memory
and measured peak support them; never run over an occupied card's memory budget.

CTC native code exists in the dirty working tree but is not yet in the pinned
Git commit or marked READY by `protocol_v3.json`. The generic retrieval CLI does
not dispatch RxRx3-core; dedicated scripts exist and require explicit wiring.
The complete fixed-commit command matrix and GPU batch-invariance/resource-peak
certificates have not yet passed. New formal dispatch remains gated until these
checks are completed. Git sync alone does not imply tests have restarted.
