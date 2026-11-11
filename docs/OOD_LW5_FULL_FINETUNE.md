# BioDINO-L lw5 OOD adaptation

This implementation adds full-backbone fine-tuning for RSNA bone age, RSNA
pneumonia detection, and BBBC041. It retains separate frozen complementary
analyses for Cryo single-particle representations, PXCT synapse confidence,
and single-volume FILM self-supervised restoration. Those complementary jobs
are **not** full fine-tuning and are **not** all independent official ML tests.

All jobs in the 2026-11-10 campaign run on suxin. Source code is versioned on
GitHub; data, model weights, credentials, runtime state and predictions are
external to the repository. The scheduler starts disabled.

## Model and resolution

The initial model is BioDINO-L lw5, step79,543:

- checkpoint SHA256: `f63544ca11ab72d7c3e05b5765c613ebc9d6374ba45ee5ca0c619eaa2f4cc423`
- training YAML SHA256: `e33cb26b1eecbec0efa7a466b17fb59f16f7a77ae25aa6b4b138028aa95eabe3`

Its supplied YAML records global crops256 and local crops112. The ViT/RoPE
implementation supports variable spatial sizes; that capability does not
establish a particular pretraining resolution distribution.

Fine-tuning explicitly uses multiscale training. Evaluation resolution512 uses
training sizes256/384/512; evaluation768 uses512/640/768; evaluation1024 uses
768/896/1024. Detection preserves aspect ratio with a long-side bound twice
the evaluation short-side. Bone age retains the full field and pads to square.
The old frozen detector's forced224 resize is not used.

## Scientific protocol

- Bone age: fixed12,611/1,425/200 train/validation/test. Image-only and image+sex
  are separate branches; sex is joined from released metadata and never inferred.
- Pneumonia: fixed24,078/2,606 internal train/validation from the official26,684,
  and all3,000 stage2 test. Existing official train/test patient overlap is disclosed.
  Score is the published strict-IoU RSNA mean precision, not COCO AP. Local scorer
  parity with private server code is not certified.
- BBBC041: fixed1,087/121/120, all seven categories including difficult. COCO
  AP@[.50:.95], maxDets300, is an explicit project metric, not an author leaderboard.
- Each of four branches has3 encoder LRs ×2 resolution regimes ×2 development
  seeds101/102:48 validation-only runs. Best epoch is selected on validation.
  Recipe selection uses mean validation score across both development seeds.
- Freeze the recipe, then train independent downstream seeds0–4 from the same
  initial checkpoint:20 formal runs. Five seeds are not five independent pretrainings.
- All five formal training runs in a branch finish before its separate test jobs
  become eligible. The operator must also enable `allow_test` in `caps.json`.
- For pneumonia, the confidence threshold is selected on validation over the
  declared grid and carried unchanged into test evaluation.
- AdamW, fp32 master parameters/state, bf16 autocast, activation checkpointing,
  layerwise LR decay, warmup/cosine, gradient clipping and effective-batch-preserving
  accumulation are explicit. Every trainable backbone parameter is in the optimizer.
  No silent missing-key loading or head-only checkpoint saving is allowed.
- Save source-byte receipts, code commit, all recipe hashes, seed, validation history,
  selected full model, resumable full optimizer/RNG state and native-coordinate predictions.
- `summarize_ood_transfer.py` reports all five values, mean and sample SD; its separate
  clustered bootstrap estimates test-unit uncertainty conditional on those runs.
  Unknown patient/slide grouping is disclosed, not invented. Literature point estimates
  without comparable predictions do not support paired significance claims.

## Entry points

```bash
python scripts/prepare_ood_transfer_inputs.py --help
python scripts/prepare_frozen_ood_transfer_inputs.py --help
python scripts/verify_ood_transfer_inputs.py --help
python scripts/configure_ood_transfer_campaign.py --help
python scripts/run_ood_full_finetune.py train --help
python scripts/run_ood_full_finetune.py test --help
python scripts/run_ood_frozen_transfer.py --help
python scripts/run_ood_transfer_campaign.py --help
python scripts/summarize_ood_transfer.py --help
```

Prepare manifests outside the Git checkout, verify each resident input file, then
configure the campaign. Set its `enabled` field only after code/model/input identity
checks and resource review. Start with one training process per GPU and microbatch1.
The scheduler admits only jobs whose input verification receipts exist; incomplete
source transfers cannot silently fall back to another dataset. Failed jobs remain
visible and are not silently discarded from hyperparameter selection.

Runtime dependencies: PyTorch/torchvision with working CUDA detection operators,
NumPy, Pillow, tifffile, scikit-learn, OmegaConf, pycocotools and psutil, plus the
base repository dependencies. FILM preparation additionally needs the author's
released SPEND/CSBDeep package; generated pairs must match the canonical array hashes.
Record installed package versions on the execution host.

### Concurrent jobs on one GPU

After measuring real jobs, `caps.json` can set `max_jobs_per_gpu`, optional
per-GPU `gpu_slots`, and `gpu_peak_mib_by_task`. The scheduler reserves each
resident job's remaining peak allocation in addition to all memory currently
used on the GPU, including other users' processes. The20% GPU/RAM reserves,
CPU limit and disk reserve still apply. Unknown tasks retain the original
conservative peak budget. Concurrent jobs retain their prescribed training
budgets and independent seeds; throughput must be measured after stacking.

`disk_peak_gib_per_active` controls the additional free-space reservation for
each concurrent job; the original default is 12 GiB. On the suxin lw5
campaign, the measured full 1.2 GiB best checkpoint, 3.5 GiB latest optimizer
checkpoint and its temporary 3.5 GiB replacement fit under 9 GiB. The separate
120 GiB host reserve remains. Ready independent test calls and validation-locked
formal seeds receive scheduling priority over unrelated development runs.

To update only scheduling while training is active, set `runner_repo` and
`runner_commit` in the campaign to the original clean scientific checkout.
Run the newer scheduler from a separate committed checkout. Every child train,
frozen analysis and test command uses the pinned original runner. This preserves
resume/test code identity without restarting active training.

Frozen complementary analyses use256 inputs and the model's recorded normalization;
this is a separately named adaptation protocol and does not overwrite frozen-v2 scores.
Cryo excludes each query from its cosine1-NN bank and reports structure/quality jointly.
PXCT reports both specimen folds. FILM reports noisy-pair validation and full126-band
reconstruction/spectra, without clean-GT PSNR/SSIM or independent-test claims.
Cryo pose stays blocked until its canonical manifest and reconstructed labels are audited.

Do not compare a literature number as a matched baseline until test members, inputs,
metric, external training data, TTA and ensemble use have been recorded. The checkpoint
was historically selected on ID results; it is fixed by the user before this campaign,
not selected using these OOD results. Same L/7B biological pretraining data is a user-
confirmed premise; detailed source/initialization provenance still belongs in the paper.

## Sources

- [RSNA bone-age challenge](https://pmc.ncbi.nlm.nih.gov/articles/PMC6358027/)
- [RSNA pneumonia evaluation](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/overview/evaluation)
- [BBBC041](https://bbbc.broadinstitute.org/BBBC041)
- [Cryo-IEF](https://www.nature.com/articles/s41592-025-02916-8)
- [PXCT](https://www.nature.com/articles/s41592-025-02891-0)
- [FILM](https://www.nature.com/articles/s41592-026-03090-1)

No SOTA performance or completed scientific acceptance is implied by publishing this code.
