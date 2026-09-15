# HS6-L5 dual-anchor spatial diagnostic dataset correction

Status: `LOCKED_CORRECTION_BEFORE_ACCEPTED_DUAL_SPATIAL_RESULT`.

The first dual invocation inherited `train.dataset_path` from the 5-TB
continuation config.  The pre-existing control and official-a7807 spatial
records instead used the fixed 1-TB diagnostic stream:

`packwds_robust:/mnt/huawei_deepcad/webds_micro_100k_by_channel_patched_shuffle/filtered_mixed_train_w*.tar::pct=1,99`

That first dual output is protocol-invalid because none of its 128 ordered
keys matched the paired controls.  It is retained under
`outputs/00_reports/hs6_l5_dual_anchor_gram_screen_20260912/invalid_spatial_5tb_mix/`.

The accepted rerun changes only the explicit `--dataset` argument to the
fixed stream above.  Checkpoint, code, seed, crop geometry, augmentation,
layers, batch count, and the preregistered decision thresholds remain
unchanged.  Acceptance requires exact equality of all 128 ordered keys and
all 128 local crop areas with both frozen control arms.
