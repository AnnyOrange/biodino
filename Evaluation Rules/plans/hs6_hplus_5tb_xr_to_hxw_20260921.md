# HS6 H+ 5TB no-Gram trajectory: hxw evaluation (2026-09-21)

## Scope and source

The user explicitly requested the 5090-lyx-xr H+ 5TB trajectory, step 0 through
6831, be transferred to 5090-hxw-xzj for testing, historical protocol first,
then only the remaining v3 regression and segmentation cells. The user also
explicitly requested removal of this campaign's hxw cache and transferred
checkpoints after tests and result validation. Do not delete source weights,
any other campaign's files, or the results/manifest/logs of this campaign.

Source training run:
`/data/xuzijing/biodino/outputs/01_training_runs/HS6_Hplus_robust_biosafe256_gb1024_lr5e5_wu3_tw30_nosig_e15_5tb_mix1m03_5tv107_8x5090lyxxr_20260920`.
Step 0: `/data/xuzijing/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth`.
Teacher snapshots: `eval/training_{487,975,1463,1951,2439,2927,3415,3903,4391,4879,5367,5855,6343,6831}/teacher_checkpoint.pth`.
Architecture `vit_huge2`, seed 0, RGB, batch 1024, 4098 updates per epoch,
15 planned epochs, 30% 1TB + 70% 4TB. The run ended after 6832 updates due
to a host-memory OOM. The independent teacher export at 6831 was verified on
the destination. Separately, the source's 17.8-GB consolidated checkpoint
6831 was verified via `torch.load(..., mmap=True)` to contain iteration 6831,
1700 model keys, and optimizer state; it is used only to resume source training.

Destination exclusively owned by this campaign:
`/data/hs6_hplus_5tb_eval_20260921` on hxw. Copy the 14 teacher exports,
the step-0 initialization, and the run config, approximately 50 GiB total;
do not copy any full training checkpoint or dataset. hxw had 577 GiB free at
admission. Verify transferred sizes and hashes against lyx before accepting
each checkpoint. Retain source hashes/provenance in small metadata.

## Evaluation order and isolation

1. Historical 25-classification, 4-regression, 6-retrieval/clustering,
   8-segmentation, 3-detection dataset matrix for each of 15 points. Keep
   observations/legacy proxies labelled, not misreported as v3. Use the
   historical evaluation's fixed, recorded best-resolution/split/layer/seed
   choices. Frozen feature batch 64, bf16, final CLS + mean patch; do not
   lower batch/resolution/layer to fit hardware. Cache is per-campaign only.
2. After all historical cells have validated results, run only *missing*
   formal-v3 regression and segmentation cells. Formal regression excludes
   CoNIC/LIVECell cell-count proxies; formal segmentation excludes BBBC038
   observation, includes CoNIC source-grouped, PanNuke all three folds, and
   LIVECell official split. Use approved v3 best dataset resolution/layers;
   frozen batch 64, segmentation feature/probe batch 32/32, independent E20
   and E50 with every-epoch validation and seeds 0/1/2 when the corresponding
   approved formal descriptor requires them. Read existing historical cell
   result and validation provenance before deciding it can be reused. Separate
   historical and v3 output roots; never substitute a differing split/probe.
3. Pin code snapshot, dataset inventory/split hashes, checkpoint/config hashes,
   Python/torch/sklearn versions, expanded command, GPU, seed and output paths
   in campaign manifests. Validate individual results for readable finite
   metrics, expected counts and protocol provenance; don't treat exit code 0
   alone as success. Mark blocked/missing cells explicitly, not complete.

On hxw GPUs 2-7 are occupied by another user's IRFlow training (~30 GiB per
GPU), not available for this campaign. Use only GPU 0 and GPU 1, rechecking
memory and other jobs before each admission; 3 historical lanes per available
GPU were admitted at 12:17 UTC (~21 GiB on GPU 0). CPU workers 2 and BLAS
threads 1; never preempt other work.
Output and logs must survive on hxw and a small-result mirror on the shared
Huawei workspace before cleanup. Resource/protocol errors suspend dispatch
until diagnosed; no silent parameter changes or repeated blind retries.

## Cleanup gate

As each individual historical dataset result validates, remove its feature
cache immediately; as each v3 cell validates, remove its cell cache immediately.
The campaign-only sweeper checks for active processes every 30 seconds and
logs exactly what was removed and how it can be regenerated. Preserve all
unfinished caches and results. Only after the *entire requested grid* is
terminal and every claimed complete cell has independently validated output
on hxw plus verified shared small-file mirror: inventory explicit realpaths
under this campaign's target, ensure no process has them open, remove the
transferred 15 checkpoint files (including step 0), and record exact removed
paths/bytes. Do not delete the original lyx weights, the local/shared copies,
or hxw's pre-existing checkpoint/cache trees. If any task is still running,
blocked, invalid, or unmirrored, defer checkpoint deletion and report work.

## Execution record (2026-09-21 08:57 UTC)

- Destination was absent before staging; source had exactly 14 independent
  teacher exports of 3,556,425,127 bytes each plus the step-0 initialization.
- The step-0 file SHA256 matched source and destination:
  `7c1da9a54b3bdb333f5ebc42e404b7f19b1b5bed504877623c9dc87397f41488`.
  The copied config SHA256 matched too:
  `df3e0ecdd32adb7d85d5453d0c210cebccc44873cb4bee2ec3ffa38100897933`.
- Direct source-versus-target `rsync -acn --itemize-changes` over the entire
  14-snapshot eval tree completed with exit 0 and **no differences**.
  A memory-mapped read of the copied step-6831 export found the explicit
  `teacher` dictionary. The 17.8-GB training checkpoints were not copied.
- Historical step 0 GPU 0 and step 1463 GPU 1 started under the identical
  pre-existing remote benchmark shell (SHA256
  `a29c8ccf64bb9182f40e0b9a435e2dcc79b108bb0bf23b4b6224f6ff121e8e4c`)
  and the isolated local worker script copied to the campaign's `bin/`
  (SHA256 `58defeeeb8a5b4ae2096565ec2e6a942a2e0567a46cd59338b1faf8f2ec7e55a`).
  All other 13 verified teacher snapshot directories are admitted via symlinks;
  no old-result directory was overwritten.
- The existing hxw v3 dataset preflight marks official MoNuSeg **BLOCKED**
  (`official30/test14 sample identity evidence pending; 37-pool forbidden`).
  Do not represent that cell as formally tested, or satisfy the final cleanup
  gate, until a valid approved split can be established.
- A campaign-only wait script starts a second historical worker on GPU 0 after
  all 11 historical step-0 lanes have completed without a terminal error;
  it will not use GPU 2–7 or admit an unverified checkpoint. The 14-point
  GPU-1 worker and step-0 GPU-0 worker each have at most one child per GPU and
  one attempt per lane. Historical outputs are under the hxw campaign's `old/`
  directory; small results are mirrored to
  `outputs/02_eval_runs/hs6_hplus_5tb_xr_hxw_20260921` in the shared workspace.

## Added work (2026-09-21 12:17 UTC)

- Started two additional historical workers per GPU 0 and GPU 1 using shared
  atomic lane claims and one child per worker; six workers total, three per
  available GPU. No jobs were admitted to GPUs 2–7 because of IRFlow.
- Started `scripts/sweep_hplus_5tb_eval_cache_20260921.py` from the isolated
  destination `bin/`, polling every 30 seconds; its initial verified sweep
  removed 31 completed historical feature caches (~8.2 GB), leaving metrics,
  checkpoints, and unfinished features intact. Recorded individual paths and
  result SHA256 hashes in `logs/cache_cleanup.jsonl`.
- The separate 2-GPU H+ continuation launcher is
  `scripts/resume_hs6_hplus_5tb_2x5090lyxxr_20260921.sh`. GPU 4/5 were idle
  before admission. A hardlink in a new run directory points to the validated
  6831 full optimizer checkpoint without copying or editing the source. With
  2 x 64 x 8 accum the effective batch remains 1024. Launcher log is
  `outputs/01_training_runs/hplus_5tb_resume6831_2gpu_20260921.nohup.log`
  on lyx-xr. At 12:19 UTC model load reported zero missing/unexpected keys,
  optimizer was restored, and the first resumed optimizer update 6832 wrote a
  finite total_loss=13.501849 to `raw_loss_metrics.jsonl`. GPUs 4/5 each used
  ~27.2 GiB out of 32 GiB during the first update; preserve the ~4.8 GiB
  headroom rather than increasing the local microbatch to a risky 128.

## Additional GPU admission (2026-09-22 02:31 UTC)

- On hxw there were 84/165 completed historical lanes, 345 individual
  dataset result JSON files, and 345 immediately removed validated feature
  caches. Source lyx-xr training had advanced to optimizer update 8159.
- Added two bounded dense-lane workers on GPU 0 and one on GPU 1 using
  `scripts/run_hplus_5tb_eval_dense_20260922.sh`; the GPUs then had exactly
  three active benchmark children each. At admission GPU 0 occupied ~14.5 GiB,
  GPU 1 ~18.1 GiB. Shared atomic lane claims prevent duplicated tests.
- Another user's IRFlow still occupies ~25.6 GiB on each GPU 6 and 7; the
  available ~6.5 GiB is less than the observed H+ high-resolution test peaks
  and risks terminating that unrelated training. A separate one-child-per-GPU
  gated dispatcher `scripts/wait_hplus_5tb_eval_gpu67_20260922.sh` now waits
  for at least 14 GiB free on each card before admitting its next historical
  lane. It leaves IRFlow, frozen evaluation batch 64, and results unchanged.

## GPU 6/7 direct dispatch (2026-09-22 02:43 UTC)

- The user explicitly requested direct H+ tests on GPUs 6 and 7 despite the
  ~6.5 GiB free per card. Stopped only this campaign's two *idle* gate shells
  (PIDs 842485 and 842486); IRFlow was left untouched.
- Launched one old-protocol worker per card with fixed frozen batch 64,
  `--jobs-cap 1` and shared claims, restricted to the observed lower-memory
  regression/retrieval lanes. GPU 6 claimed step-6343 regression and GPU 7
  claimed step-6343 retrieval. Each H+ process reserved ~5.2 GiB, leaving
  ~1.3 GiB spare on those cards while the unrelated IRFlow training stayed
  running. Monitor for resource errors and do not misreport successful launch
  as completed results.

## Per-point cleanup override (2026-09-22 09:20 UTC)

The latest user instruction **supersedes the earlier global checkpoint
cleanup gate**: after *each checkpoint's historical 11 lanes* all finish,
validate 48 dataset/fold result JSONs, confirm zero active references and
delete only that checkpoint's **hxw copy** plus its two symlinks. Keep the
old-protocol results and the lyx-xr originals. Formal-v3 did not yet run; its
missing regression/segmentation cells must re-transfer the required copy from
lyx-xr first. Create `v3/checkpoint_holds/point_<id>.hold` **before** any
v3 re-transfer, keep it until formal-v3 result validation, and only then
remove the hold so the cleanup watcher deletes that temporary copy again.
Never misreport historical completion as full v3 completion.

GPU 0 and GPU 1 were restored to three old-protocol children each; GPUs 6 and
7 were restored to one each using the `--once` historical worker wrapped by
`scripts/keep_hplus_5tb_eval_lane_busy_20260922.sh`. Restrict each to its
assigned GPU and rely on atomic claims to avoid duplicate lane runs. Both
campaigns now have 30-second validated per-dataset feature-cache sweepers;
L's first sweep removed 105 finished feature caches, reducing its results
directory from ~8.6 GiB to a few MiB. H+ completed-lane bulk caches are
removed by the worker on success.

The first eight eligible H+ points were 0, 487, 975, 1463, 1951, 2439,
2927, 3415. Each had 11 successful lane markers and 48 readable finite
result JSONs. Their complete old-protocol result directories were mirrored
to the shared workspace and verified by `rsync -acn` before deletion. The
scoped `scripts/cleanup_hplus_5tb_completed_old_checkpoints_20260922.py`
removed all eight **hxw** copies, in total 28,258,208,456 bytes, and recorded
pre-deletion checkpoint SHA256/result digest plus deletion receipts in
`/data/hs6_hplus_5tb_eval_20260921/logs/old_complete_checkpoint_deletions.jsonl`.
The 30-second watcher remains active for later historically complete points.
L checkpoints have not yet passed a full historical 11-lane gate; none was
deleted.

## Parallel old + v3 override (2026-09-22 14:12 UTC)

The user superseded the sequential-order and per-old-point checkpoint-deletion
instructions: run old and formal-v3 concurrently, reusing only results whose
protocol, split, model, data identity, runtime, budget and seed are verified
identical. **Do not delete any further H+ or L hxw checkpoint copy merely on
old completion.** The H+ and L per-point checkpoint-deletion watchers were
stopped (PIDs 1179731/1183089); keep the validated 30-second old/v3 cache
sweepers active. Each v3 cache can be removed only after all six independent
E20/E50 x seed0/1/2 fits validate, because they share extracted features.

Re-staging the 12 prematurely removed H+ exports plus step0 from unchanged
lyx-xr originals is in progress through
`scripts/restage_hplus_5tb_for_parallel_v3_20260922.sh` under hxw
`logs/restage_for_v3_20260922.log`; exact archive/source checksum comparison
must finish before calling restoration complete. The L source-to-hxw watcher
continues to transfer training teacher exports as they stabilize; it must not
be stopped or subjected to per-old-point deletion.

The pinned September 18 formal-v3 evaluator snapshot was copied to hxw's
L campaign `bin/v3_source_snapshot` (source code hashes matched the established
old evaluation entry points). `scripts/run_hplus_l_5tb_v3_dense_hxw_20260922.py`
validates explicit canonical splits, uses batch32/32 and independent E20/E50
seed0/1/2 with every-epoch best validation, records config/checkpoint/code
hashes, then emits a cell-level VALID_COMPLETE report. Initial L 29767 and
H+ 5855 CoNIC probes encountered a broken PyTorch 2.10 torch_shm_manager
dynamic-library search. The temporary torch2.8 diagnostic probes were
archived as `results.diagnostic_env28.json`, *not counted as formal*. The
existing feature cache was originally extracted under torch2.10 and reused.
A smoke test confirmed PyTorch 2.10 multiprocessing works with explicit
CUDA runtime/CUPTI library paths; both formal-v3 cells were restarted with
that pinned numerical environment.

GPU0 and GPU1 currently each carry three combined H+/L old+v3 tests. GPU2
and GPU3 were verified free of unrelated user compute and now carry five old
L tests apiece, with extra slots restricted to segmentation. Claims are
atomic across existing and added old workers. Monitor free disk (new five-way
segmentation cache can grow quickly) and GPU peaks; pause admission before
storage exhaustion without deleting active or unvalidated feature caches.

Additional user direction: aim for >50% of each GPU0–3's VRAM and high GPU
utilization where the real evaluator can achieve it, provided host RAM and
storage remain safe. GPU1 received two light old-L slots, GPU2 four H+ v3
CoNIC cells, and GPU3 seven L v3 CoNIC cells alongside five old-L slots per
GPU2/3. At 14:19:33 UTC the observed GPU0–3 VRAM was 23.7/21.3/18.7/17.3
GiB and GPU utilization 99/61/81/99%; occupancy fluctuates as each v3 cell
switches from GPU feature extraction to mostly CPU probe fitting. Do not
allocate fake tensors solely to inflate VRAM accounting. Stop admitting
new tasks when either free `/data` is <230 GiB or MemAvailable <60 GiB;
`scripts/guard_hxw_l_5tb_eval_disk_ram_20260922.py` pauses only the exact
12 additional old-L launch shells and resumes above 300/110 GiB, leaving
their already running test children untouched. The two 30-second sweepers
directly delete validated old result features/fold caches and validated v3
cell caches; never migrate deletable completed cache to the system disk.
System disk is authorized only as an emergency location for genuinely
unfinished per-campaign scratch if data storage cannot hold it safely.

## Five-real-tests-per-GPU recovery (2026-09-23 02:50 UTC)

The apparent worker count had diverged from real test concurrency: by the
morning, H+ old was 165/165, L old was 159/165 including the terminal
29767/detection resource failure, all original one-shot v3 CoNIC processes
had exited after 15 VALID_COMPLETE cells, and only six old benchmark children
were still active. Idle loop shells are not tests and must not be counted.

Admitted the remaining 15 formal-v3 CoNIC cells so the live process inventory
became exactly GPU0=5 v3, GPU1=3 v3+2 old, GPU2=3 v3+2 old, GPU3=4 v3+1 old.
As old lanes completed, a 20-slot atomic v3 supervisor fleet
(`scripts/run_hplus_l_5tb_v3_queue_hxw_20260923.py`) took over; a direct
`/proc` ancestry audit at 02:50 UTC again found exactly five real tests on
each GPU0–3. Each slot waits for the test it replaces, refreshes newly staged
L checkpoints after every cell, atomically claims only a missing formal-v3
segmentation cell, and refuses new admission below 180 GiB free data space or
60 GiB MemAvailable. It records rather than blindly retries runtime failures.

The L cache sweeper had exited because another cleanup removed the same
validated fold between discovery and `rmtree`. The sweeper now treats this
FileNotFound race as the already-desired state and was restarted as PID
2191690; the H+ sweeper remains PID 1190069. Completed caches are deleted
directly, never moved to system disk.

Five early supervisor LIVECell admissions failed before extraction because
the frozen preflight could not locate `Evaluation Rules/protocol_v3.json`:
the v3 launcher had not exported `DINOV3_CODE_ROOT`. The launcher now pins it
to the frozen snapshot. Those failure receipts were retained under
`failures_recovered_missing_code_root`, their empty atomic claims were
released, and post-fix LIVECell 6343/3415 passed the locked annotation audit
and entered formal extraction. At handoff there were zero active queue
failure receipts, all 20 supervisors alive, ~428 GiB free on `/data`, and
~230 GiB MemAvailable.
