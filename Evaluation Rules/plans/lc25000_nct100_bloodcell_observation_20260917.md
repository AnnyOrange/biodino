# LC25000 / NCT100 / BloodCell Observation Protocol - 2026-09-17

Status: OBSERVATIONAL_TEST_PROTOCOL_DRAFT

Owner intent:
- LC25000 is allowed back into testing as a formal-candidate retrieval/classification dataset.
- NCT-CRC-HE-100 should be tested, but remains observation-only until we decide whether its small sample size is acceptable for reporting.
- BloodCell can be tested after fixing a deterministic split; it is not yet a formal-table dataset.

## Datasets

### LC25000
- Local source: `/mnt/huawei_deepcad/benchmark/Retrieval_Clustering/LC25000`
- HXW copy: `/home/xzj/chadavit_wds/benchmark/Retrieval_Clustering/LC25000`
- lyx-xr copy: `/data/xuzijing/chadavit_wds/benchmark/Retrieval_Clustering/LC25000`
- Task: retrieval / clustering observation first; formal-candidate after result sanity check.
- Protocol: use existing `run_retrieval_clustering` dataset id `lc25000`.
- Split: within-dataset feature extraction over all images; retrieval evaluated by label match. No train/test fitting.
- Metrics: retrieval top-k / mAP and clustering metrics emitted by the existing runner.
- Formal gate: result file must include sample count 25000 and 5 class labels.

### NCT-CRC-HE-100
- Local source: `/mnt/huawei_deepcad/benchmark/Retrieval_Clustering/NCT-CRC-HE/owkin_hf_parquet/data/nct_crc_he_100-00000-of-00001-25a54abad9e9e379.parquet`
- HXW copy: `/home/xzj/chadavit_wds/benchmark/Retrieval_Clustering/NCT-CRC-HE/owkin_hf_parquet/data/nct_crc_he_100-00000-of-00001-25a54abad9e9e379.parquet`
- Task: retrieval / clustering observation only.
- Reason for non-formal status: only 99 usable samples across 9 tissue classes, so metric variance is high and one or two examples can move class-level behavior.
- Protocol: use existing `run_retrieval_clustering` dataset id `nct-crc-he-100`.
- Split: full-dataset feature extraction; no train/test fitting.
- Metrics: retrieval top-k / mAP and clustering metrics emitted by the existing runner.
- Formal gate: do not aggregate into the official table unless the review explicitly accepts low-N observation datasets.

### BloodCell / PBC
- Local intended source: `Classification/BloodCell/PBC_dataset_normal_DIB`
- HXW copy: `/home/xzj/chadavit_wds/benchmark/Classification/BloodCell`
- lyx-xr copy: `/data/xuzijing/chadavit_wds/benchmark/Classification/BloodCell`
- Task: classification observation.
- Current protocol gap: the dataset is an image-folder classification corpus without a checked-in frozen split manifest in the formal rules.
- Draft split: deterministic stratified 80/20 split, seed 0, grouped only by image path because no patient/source grouping is available in the visible tree.
- Probe: frozen HS6 features, StandardScaler + multinomial LogisticRegression, fixed seed 0.
- Metrics: balanced accuracy, macro F1, accuracy, per-class support.
- Formal gate: add a split manifest with path hashes before official-table inclusion.

## Model Runs

### HS6 S+
- Machine: `5090-lyx-xr`
- Checkpoint: `/data/xuzijing/biodino/outputs/01_training_runs/HS6_Splus_robust_biosafe256_gb1024_lr2e4_wu3_tw30_nosig_e15_seed0_8x5090xr_20260821b/ckpt/9224/checkpoint.pth`
- Config: `/data/xuzijing/biodino/outputs/01_training_runs/HS6_Splus_robust_biosafe256_gb1024_lr2e4_wu3_tw30_nosig_e15_seed0_8x5090xr_20260821b/config.yaml`
- Output root: `/data/xuzijing/biodino_eval_git_20260917/dinov3/outputs/02_eval_runs/lc25000_nct100_observation_20260917/splus_ck9224`
- Active scope: `lc25000`, `nct-crc-he-100`.

### HS6 H+
- Machine: `suxin-8H100-1`
- Checkpoint: `/data_2/suxin/runs/HS6_Hplus_robust_biosafe256_gb1024_lr5e5_wu3_tw30_nosig_e15_seed0_4xH100_20260818/ckpt/15374/checkpoint.pth`
- Config: `/data_2/suxin/runs/HS6_Hplus_robust_biosafe256_gb1024_lr5e5_wu3_tw30_nosig_e15_seed0_4xH100_20260818/config.yaml`
- Output root: `/data_2/suxin/biodino_eval_git_20260917/dinov3/outputs/02_eval_runs/lc25000_nct100_observation_20260917/hplus_ck15374`
- Active scope: `lc25000`, `nct-crc-he-100`, after data and Python environment verification.

## Transfer Rule Used

Small datasets under the one-day-at-800KB/s threshold are staged to HXW first, then distributed outward when possible:
- LC25000: lyx-xr -> HXW completed.
- NCT100 parquet: local -> HXW completed.
- BloodCell: lyx-xr -> HXW completed.
- H100 distribution uses local SSH pipe because direct HXW <-> H100 SSH is not currently available.

Large or task-specific datasets remain manual-transfer candidates and should be listed separately with exact paths and sizes.
