# FM14 BBBC038 observation companion

Authorization: user explicitly requested BBBC038 execution on 2026-09-17.
Models: all14 published FM checkpoint assets; no checkpoint/data transfers.
Dedicated output: `benchmark_model/benchmark_runs/fm14_bbbc038_observation_20260917`.
Never include BBBC038 in formal TierA/TierB/overall means or main ranking.

Use the existing locked stage1_train masked seed42 split, train470/val100/test100,
split hash `4eb72dc1e58453126893261fedff661b7eb58dc9863ae04d991e7a5282f57ac4`.
Confirm actual counts automatically before launch; official unlabelled test
is not evaluated. No split edits and no old results deletion.

Segmentation:512/pad, primary final spatial map, additional dataset-best
even4 if supported; CNN last fallback aliases primary. Native channels and
published normalization use the audited FM14 adapter. Feature/probe batch32,
BF16/workers2, independent E20/E50 with seed0/1/2, validate every epoch,
earliest best-validation head then one test. Record all expanded recipes.

Detection observation:224/stretch, final spatial map aligned to common
stride16 labels, batch8/BF16/workers2/seed0, fixed center-patch linear head,
AdamW lr1e-3/wd1e-4,5epochs. Save test patch F1, explicitly a proxy not COCO mAP.

Shared machines follow maximum5 total project tests/GPU, BLAS1 and Rules'
memory/RAM/NFS gates; high-resolution segmentation starts single. Frozen and
small observation tasks can fill spare slots alongside the unchanged main
campaign. One attempt per cell, fixed code/version and independent validators.
