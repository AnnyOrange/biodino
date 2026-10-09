# Experiment source snapshots

On 2026-10-08, 28 `dinov3*` sibling directories under
`/mnt/huawei_deepcad` were archived in
`outputs/03_source_archives/dinov3_siblings_20261008/`. The SHA-256 index and
original absolute paths are in
`outputs/00_reports/source_cleanup_20261008/archive_manifest.json`.
Twenty-five inactive directories were removed after the archives passed
integrity checks. The main `dinov3` repository was not moved or replaced.

The current Adaptive-v2 training implementation already lives in this
repository (`dinov3/train/selective_recoverability.py`, the recovery hooks in
`ssl_meta_arch.py` and `train.py`, and the `launch_v2_recovery_*` scripts).
Historical WDR, calibrated/metric recovery and multistage prototypes remain
in their named archives. They were not copied over the current training path
because they are not the running Adaptive-v2 implementation.

Three directories remain because evaluation processes still reference their
absolute paths:

- `dinov3_20tb_online_snapshot_20260918` (3090-qi)
- `dinov3_method_full_v4_snapshot_20260928` (local and deepcad)
- `dinov3_retest_snapshot_20260918_fm_bound` (local and 3090-qi)

Their archives have also been created. Remove these last three only after
their running evaluation controllers and workers have stopped or migrated.

To replay an older run, extract its named archive back under
`/mnt/huawei_deepcad` and verify its SHA-256 against the index. Archives omit
Python caches, `pymp-*` scratch directories, and two generated WebDataset test
shards. The omitted shards in seven large copies were byte-for-byte identical
to the files retained in the main repository at
`dinov3/dataset_webdataset/test/dataset_webdataset/`.
