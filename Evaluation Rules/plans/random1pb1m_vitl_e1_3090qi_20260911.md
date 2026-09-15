# Random 1PB 1M ViT-L one-pass campaign (3090-qi, 2026-09-11)

## Question and comparison

Train one HS6 ViT-L arm for exactly one pass over the audited random sample of
1,000,000 images from the 1 PB pool.  This is the random-1PB arm of the user's
planned random-1PB versus random-100TB comparison.  The 100TB arm is not part
of this launch and must receive its own precommitted plan after its 1M-image WDS
is complete.

## Training

- Host: `3090-qi` (remote hostname `server`); GPUs: local CUDA 0-7, 8 x RTX 3090.
- Model/config: ViT-L/16, `dinov3/configs/train/microscopy_continual_vitl16.yaml`.
- Initialization: `/mnt/huawei_deepcad/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`.
- Dataset: `packwds_robust:/mnt/huawei_blm/random_1pb_100tb_each_1m/1pb/wds_patched_shuffle/filtered_mixed_train*.tar::pct=1,99`.
- Dataset audit: 1,000,000 samples, 221 shards, zero WDS audit errors; source
  `/mnt/huawei_blm/random_1pb_100tb_each_1m/1pb/manifests/1pb_final_wds_audit.json`.
- Normalization: robust percentile RGB mean
  `[0.33664738282345813,0.3372330548904934,0.33792517941669703]`, population
  std `[0.3137400759359795,0.3119889017167462,0.31299109437827144]`, estimated
  from 100,000 samples with zero failures.
- One epoch, `OFFICIAL_EPOCH_LENGTH=977`; per-GPU batch 16; accumulation 8;
  effective global batch 1024.  The final padded batch makes 1,000,448 sample
  visits, i.e. one dataset pass with 448 repeated/padded visits.
- LR `1e-4`; proportional e1 schedule; bio-safe augmentation; global/local
  crops 256/112; SIGReg off; seed 0; endpoint checkpoint period 977.
- Output: `/mnt/huawei_deepcad/dinov3/outputs/01_training_runs/HS6_L_random1pb1m_robust_biosafe256_gb1024_lr1e4_e1_seed0_8x3090qi_20260911`.
- Training launcher: `scripts/launch_hs6_cscale_duration_prop.sh`; automatic
  legacy quick evaluation is disabled for this campaign.

The checkout is owned by an NFS identity not trusted by the login user's Git,
so Git refuses `rev-parse` with `dubious ownership`.  No global safe-directory
exception or source-code change is made by this campaign.  This limitation is
recorded and is a formal-evaluation preflight blocker until independently
resolved/audited.

## Formal v3 evaluation

- Protocol: `Evaluation Rules/protocol_v3.json` (`bio-eval-formal-v3`).
- Checkpoint: endpoint `checkpoint.pth`, teacher/EMA branch only.  The exact
  resolved checkpoint path and stable size/hash must be recorded after train.
- Same machine: all evaluation runs on `3090-qi`; checkpoint and data are read
  in place.  No checkpoint or dataset is transferred.
- Global representation: L2-normalized final CLS plus mean final patch tokens.
- ViT-L dense layers: even4 `[4,11,17,23]`; last1 block 23.
- Batch: frozen classification/regression/retrieval/OOD 64; segmentation
  feature/probe 32; CTC 8.  Probe seed 0; segmentation 50 epochs/eval at 50.
- Tier A classification: bloodmnist, pathmnist, tissuemnist, breastmnist,
  organamnist, organcmnist, organsmnist, dermamnist, octmnist, pneumoniamnist,
  retinamnist, chestmnist, bbbc048-cellcycle, cyclops-protein-loc,
  midog25-atypical.
- Tier A regression/retrieval: bbbc005; nct-crc-he-1k, crc-val-he-7k.
- Tier A segmentation: cellpose, conic, livecell, monuseg, pannuke, tissuenet.
- Tier B classification: pcam, nct-crc-he, chammi-allen-task1/2,
  chammi-cp-task1/2/3, chammi-hpa-task1/2.
- Tier B regression/retrieval: bbbc013; hpa-subcellular, rxrx1-cross,
  rxrx3-core.
- Tier B segmentation/tracking/OOD: multimodal_cellseg; ctc; xray, cryo.
- No formal detection cells.  BBBC038 is observational only and is excluded
  from the formal aggregate.  All v3 forbidden datasets/pairs remain excluded.
- Output root:
  `/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/HS6_L_random1pb1m_e1_formal_v3_3090qi_20260911`.
- Initial GPU allocation after training: at most GPUs 0-3, respecting the
  deepcad four-GPU limit; start with one job per GPU and only increase within
  evaluator memory checks.  Logs live below the output root's `logs/` folder.
- Skip only a cell already marked `VALID_COMPLETE` with matching checkpoint,
  config, protocol, split/hash, resolution, layers, batches, probe and seed.
  Retry a failed cell at most twice; do not treat process exit 0 alone as
  completion.

## Hard launch gate

Before formal evaluation, generate the command manifest and require
`scripts/validate_bio_eval_formal_v3.py` to exit 0 and write an independent
`validation_report.json`.  At plan time this gate is expected to fail because
`protocol_v3.json` marks CTC as `APPROVED_PENDING_FIXED_HEAD_AND_LINKER`, while
the validator requires `READY_NATIVE_EVALUATOR`.  Formal testing must not be
silently downgraded to v2 or an old proxy suite; the post-train watcher may
launch evaluation only after this repository-owned v3 blocker and the Git
audit blocker are resolved and the full command manifest passes validation.
