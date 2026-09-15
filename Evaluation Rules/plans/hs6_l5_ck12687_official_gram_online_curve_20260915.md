# HS6-L5 ck12687 official-Gram online curve evaluation

- Weight source: `HS6_L5_ck12687_official_gram_a12687_b32_gb1024_noac_4xdeepcad_u2440_contract_v2_20260915/eval/training_<ck>/teacher_checkpoint.pth`.
- Reuse `point_12687` from the audited original 5TB full-registry run because it is the exact anchor file (SHA256 `c14adbb...456a`). Do not rerun it.
- New checkpoint grid starts at 13175 and continues every 488 updates through 60999.
- Families: classification, regression, retrieval, clustering, segmentation, detection. OOD is excluded from these task-family curves.
- Frozen protocol: existing full-registry evaluator, image size 224, resolution `best`, channel policy `auto`, split protocol `current`, seed 0. Frozen batch is 64 on every host.
- Workers share the output `_state/claims` directory. Completed lanes are skipped; a lane may never be computed twice concurrently.
- Dispatch to free 3090-qi GPUs and accessible single-card RTX 3090 hosts. Stack jobs using the existing per-lane worker concurrency; never reduce protocol batch by host.
- Preserve result JSON/CSV. The worker deletes only transient per-lane `cache/` after successful completion. It never deletes or copies training checkpoints.
