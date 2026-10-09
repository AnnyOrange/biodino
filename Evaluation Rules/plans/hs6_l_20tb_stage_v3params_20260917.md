# 20TB online checkpoint comparison

User approved stage comparison on 2026-09-17. Status: EXPERIMENTAL_NOT_REPORTABLE.
Not a full formal-v3 campaign; no formal aggregate/ranking and no reuse of legacy results.

Training: HS6 ViT-L 20TB default36m keepall 20260915, config in training output.
Teacher snapshots: eval/training_<id>/teacher_checkpoint.pth, every 488 optimizer
updates (499,712 image visits), starting at 487. Backfill all available snapshots,
then poll every 30 seconds. Ready age 90 seconds; maximum 3 attempts per lane.
Training schedule 35374 updates/epoch, 15 epochs; effective training batch 1024.

Task/dataset matrix derives from protocol_v3.json Tier A/B classification,
regression, retrieval, segmentation and OOD. CTC is missing because the approved
fixed head/linker is not ready. RxRx3-core is missing because the dedicated
evaluator is absent in the frozen runtime. These missing tasks are not zero scores.
No forbidden dataset, detection proxy or count proxy is scheduled.

Frozen classification/regression/retrieval/OOD batch 64, bf16, final block CLS +
patch mean, channel auto, split current, seed 0. Classification/regression use
dataset-best resolution; retrieval/OOD use 256 resize and 224 crop.
Segmentation uses dataset-best resolution/layers/resize, feature/probe batch 32,
50 probe epochs and evaluation every 50. Complete commands are generated in each
point/lane output; historical split/evaluator limitations remain non-reportable.
No sample cap. Results are only comparable between checkpoints in this campaign.

Hosts: 3090-qi GPU 2 (approximately idle at preflight), cpu1 GPU 0 (12MiB/24GiB).
One job per GPU initially, 2 data workers, BLAS threads 1. Existing jobs are not
stopped. No checkpoint, dataset or feature bank transfer; shared paths only.

Runtime: existing frozen full-registry runtime
outputs/02_eval_runtime/dinov3_a029eef0_full_registry.GOjODo. Exact commit and
hashes are recorded in campaign_manifest.json; fleet module copied once into the
campaign runtime. Launcher verifies recorded code hashes before importing it.
This staged local adapter is not a published formal evaluator commit.

Output: outputs/02_eval_runs/hs6_l_20tb_stage_v3params_20260917.
Inputs: outputs/02_eval_inputs/hs6_l_20tb_stage_v3params_20260917.
Fleet exit/done markers are execution status, not VALID_COMPLETE. Results must
be audited before downstream comparisons; no formal validator pass is claimed.

Authorized training adjustment: after checkpoint 16103 is complete and readable,
resume batch-per-GPU 128, accumulation 1, effective batch unchanged at 1024.
Keep activation checkpointing, optimizer/EMA and schedules; if CUDA OOM occurs,
recover batch 64/accumulation 2 from the latest complete checkpoint.
