# HS6 all-checkpoint expanded evaluation (2026-09-09)

## Scope

- Tasks: regression, retrieval/clustering, detection.
- Regression: `bbbc013`, `bbbc005`, `conic-cell-count`, `livecell-cell-count`.
- Retrieval/clustering: `lc25000`, `nct-crc-he-100`, `nct-crc-he-1k`,
  `crc-val-he-7k`, `hpa-subcellular`, `rxrx1-cross`.
- Detection: `livecell`, `bbbc038`, `conic`.
- Models: the standard HS6 S+, B, L, and H+ e15 runs.
- Checkpoints: every non-empty, readable `ckpt/<id>/checkpoint.pth` currently
  available under the four run directories.

## Execution

- Output:
  `outputs/02_eval_runs/hs6_all_checkpoints_reg4_ret6_cluster6_det3_3090fleet_20260909`.
- Queue: shared atomic checkpoint claims; one full checkpoint per worker.
- Hosts: `cpu1 cpu2 cpu5 cpu8 cpu9 cpu10 cpu11 cpu12 cpu15 cpu19`, GPU 0 only.
- One benchmark child per GPU (`JOBS_PER_GPU=1`, `MAX_CONCURRENT_JOBS=1`,
  `MAX_CPU_JOBS=1`, `CONCURRENT_TASK_GROUPS=0`).
- Frozen batch: S+ 64, B 32, L 16, H+ 4.
- Detection batch: S+ 8, B 8, L 4, H+ 2.
- Common protocol: bf16, current split, best regression resolution, seed 0,
  RxRx1 balanced core (`RXRX1_FULL=0`).
- A checkpoint is skipped only when all 4 regression results, all 6
  retrieval/clustering results, and all 3 detection JSON files validate. The
  validator accepts both the canonical `last_result.json` and legacy
  `summary.csv` formats.

## Checkpoint audit

- Readable for the final queue: S+ 15, B 4, L 5, H+ 1 (25 total).
- Corrupt NFS copies left excluded: B 4099, L 2049, H+ 1024.
- H+ 15374 was explicitly approved for transfer from `suxin-8H100-1`. The
  complete 17,780,596,943-byte checkpoint passed the archive check and replaced
  the truncated NFS copy; the old 81 MB file is retained as
  `checkpoint.pth.corrupt_20260909`.
- The exact `cpu19` dinov3 environment was archived at
  `outputs/02_eval_inputs/env_bootstrap/dinov3_cpu19_py311_20260910.tar` and
  installed on both `cpu2` and `cpu18`. Both nodes pass the PyTorch,
  torchvision, CUDA, source-import, and regression-registry checks.
- `cpu2` was manually mounted read-write to `172.16.1.102:/deepcad` using NFSv3
  and joined the queue. This is a live mount, not an `/etc/fstab` entry.
- `cpu18` already mounts `/mnt/huawei_deepcad`; it was not added to this run
  because its 3090 was occupied by another user's `RUSH3D_neuron` process.

## Completion audit (2026-09-10)

- All 25 readable checkpoints validate: S+ 15/15, B 4/4, L 5/5, H+ 1/1.
- All requested regression, retrieval/clustering, and detection results are
  present; validation reports 25 valid and 0 invalid checkpoints.
- The first regression launch used a stale registry copy. Its outputs were
  archived and rerun after confirming all four regression datasets on every
  worker.
- Twenty-three failure records caused solely by the old CSV-only validator are
  retained under `_state/failures_csv_validator_false_positive_20260910`; the
  active `_state/failures` directory is empty.
