# External-4 H+ / FM fixed-budget screen — 3090-qi — 2026-09-10

Status: `OBSERVATIONAL` / `EXPERIMENTAL_NOT_REPORTABLE` (not part of the formal-v2/v3 aggregate).

Post-screen decision (2026-09-11): CTC and RxRx3-core are admitted to formal
v3 only under their new native/full fixed-manifest protocols.  The results in
this plan remain quick-screen evidence and are not promoted retroactively.
MIDOG++ is excluded from the formal matrix; its result remains diagnostic.

## Question and matrix

Screen the frozen representation quality of HS6 H+ and the available external
foundation models on CTC, HEST, RxRx3-core, and MIDOG++.  The comparison grid is
15 models × 4 tests.  This is a one-hour, fixed-sample screen requested for rapid
triage; it must not be relabelled as canonical CTC tracking/segmentation or
MIDOG++ whole-slide detection.

Models: `hs6_hplus`, `dinov2`, `mae`, `siglip2`, `uni`, `conch`, `virchow2`,
`gigapath`, `hoptimus0`, `phikon2`, `pe`, `bioclip`, `cytoself`, `jump_cp`,
`cytoimagenet`.

HS6 H+ checkpoint:

- checkpoint: `/mnt/huawei_deepcad/dinov3/outputs/01_training_runs/HS6_Hplus_robust_biosafe256_gb1024_lr5e5_wu3_tw30_nosig_e15_seed0_4xH100_20260818/ckpt/15374/checkpoint.pth`
- config: `/mnt/huawei_deepcad/dinov3/outputs/01_training_runs/HS6_Hplus_robust_biosafe256_gb1024_lr5e5_wu3_tw30_nosig_e15_seed0_4xH100_20260818/config.yaml`
- teacher readout: final CLS + final patch mean, L2-normalized (`n_last_blocks=1`, `use_avgpool=true`)
- iteration: 15374; branch: teacher

## Fixed-budget protocols

- `ctc_2d_count_proxy_v1`: ten 2-D CTC domains; sequence 01 train and sequence
  02 test; four deterministic matched raw/TRA frames per sequence and domain.
  Ridge predicts instance count. Metrics: R2, MAE, Spearman and per-domain MAE.
- `hest_fold0_64_v1`: official fold-0 task CSVs; the lexically first sample in
  each official train/test partition and at most 64 spots from it per
  organ/cancer task; train/test therefore remain sample-disjoint.
  Multi-output ridge on the official 50-gene panel. Metrics: gene-wise Pearson,
  gene-wise Spearman, R2 and MAE, macro-averaged across tasks.
- `rxrx3_plate_disjoint_128_v1`: 128 genes with one gallery and one query well
  per gene, disjoint plates, CRISPR query guides only; six channels mapped to
  RGB by averaging normalized channel pairs. Metrics: R@1/5/10, MRR@10, NMI.
- `midogpp_candidate_256_v1`: official train/test slide assignment, balanced
  mitosis/hard-negative annotated candidate crops (up to 256 per split).
  Logistic regression. Metrics: F1, AP, recall and tumor-domain F1. This is a
  candidate classification/detection proxy, not whole-slide detection AP.

All images are cached as deterministic 224-pixel uint8 RGB. Probe seed is 0.
Encoder batch is 4 to permit the requested multi-process packing. Feature and
result files record exact sample counts and protocol IDs.

## Fleet allocation

Host `3090-qi`, physical GPUs 2, 5, 6, 7.  Preflight on 2026-09-10 found each
card at 1 MiB / 24576 MiB and 0% utilization.  One process owns one model and
runs all four tests sequentially.  Initial packing (at least three tests/model
workers per card as requested):

- GPU 2: `hs6_hplus`, `cytoself`, `cytoimagenet`
- GPU 5: `hoptimus0`, `gigapath`, `uni`
- GPU 6: `dinov2`, `mae`, `siglip2`, `bioclip`
- GPU 7: `pe`, `conch`, `phikon2`, `virchow2`, `jump_cp`

If a process fails from memory pressure, it is retried once with batch 2 on the
same target fleet; protocol and samples remain unchanged.

## Inputs, outputs, and reuse

- data roots are the four shared paths under
  `/mnt/huawei_deepcad/benchmark/external_benchmarks_20260901`.
- external FM loader is `/mnt/huawei_deepcad/benchmark_model/benchmark_eval/encoders.py`.
- output/log root:
  `outputs/02_eval_runs/external4_hplus_fm_fixedbudget_3090qi_20260910`.
- skip only when a model result parses, contains all four protocol IDs, has no
  error/non-finite primary metric, and its manifest fingerprint matches.
- no checkpoint or dataset is transferred; all inputs are read directly from
  the shared Huawei mount.

The existing RxRx3 HS6-L screen may be used only as diagnostic context and is
not reused as an H+ or external-FM result.

## Code and environment

The repository is intentionally dirty from already documented unrelated work.
This campaign uses only the newly named external4 scripts and this plan; their
SHA256 hashes are written to `campaign_manifest.json` at launch.  Remote Python:
`/home/bbnc/anaconda3/envs/siglip2_env/bin/python`, with only that environment's
`libstdc++.so.6` preloaded and the Transformers TensorFlow backend disabled.
H+ uses the local DINOv3 source; external
models use the shared benchmark-model registry.  Dependency versions and Git
commit are captured during preflight.
