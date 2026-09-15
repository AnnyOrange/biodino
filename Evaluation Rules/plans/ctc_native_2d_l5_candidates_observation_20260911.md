# CTC native 2-D L5 candidate observation (precommitted 2026-09-11)

Status: `PRECOMMITTED_BEFORE_RUN`.

## Purpose and admission status

- Run a real native-resolution SEG/DET/TRA evaluation while the full 20-domain
  2-D/3-D CTC implementation remains pending.
- This row is `OBSERVATIONAL_NATIVE_2D`, not formal CTC, and cannot enter the
  formal aggregate or be relabeled as the 20-domain result.
- The candidate set was fixed before any native CTC result was observed:
  HS6-L 5TB checkpoints 15615, 17079, 20007, and 21959.

## Fixed split and head

- Include the 10 official 2-D domains only. Filter the existing v3 five-fold
  manifest without changing fold IDs; each 2-D domain is held out once.
- For each fold, train only on sequence 01 SEG-annotated frames from non-held-out
  2-D domains. No held-out image, annotation, trajectory, or aggregate dataset
  statistic is used for fitting, selection, diameter estimation, or thresholds;
  the fixed per-image p01/p99 input transform is applied to every image.
- Use frozen teacher backbone final block 23 and the binary DINOHoVerNet decoder
  with `feature_size=32`, `embed_proj=384`, and the current bucket-concat decoder.
- Initialize with seed 0. Train exactly 50 epochs with AdamW, constant learning
  rate 1e-3, weight decay 1e-4, batch size 8, BF16 autocast, and no validation,
  scheduler, early stopping, or best-epoch selection. Grade the epoch-50 head.
- Each annotated frame contributes one deterministic foreground-aware 256 crop
  per epoch. Apply one uniformly sampled D4 transform. Scale each source image
  by its exact full-image p01/p99, repeat grayscale to RGB, then use the fixed
  microscopy RGB mean and standard deviation.

## Fixed native inference and scoring

- Infer every sequence 02 frame at native aspect ratio. Use 256 tiles, stride
  192 (64 overlap), uniform logit averaging, tile batch size 8, and no TTA.
- Post-process once per full frame with NP threshold 0.5, HV energy threshold
  0.4, and minimum area 10. Link with the deterministic v3 Hungarian/division
  linker; its diameter is the median equivalent diameter from that fold's
  training SEG masks only.
- Write CTC `maskNNN.tif` and `res_track.txt`. Score `Valid`, `DET`, `SEG`, and
  `TRA` using py-ctcmetrics commit
  `59481c48a62d4376fe34bed3e3606b4ec4d60972`, plus the repository AP and mean
  foreground Dice on sequence 02 SEG-annotated frames.
- Report all 10 per-domain rows and their unweighted macro average. A campaign
  is complete only at 4 checkpoints x 5 folds x 10 held-out domain rows, with
  an immutable pre-run campaign manifest and fail-closed validation.

## Paths and execution

- Cache: `outputs/02_eval_inputs/observational_v1/ctc_native_2d`.
- Output: `outputs/02_eval_runs/ctc_native_2d_l5_candidates_observation_20260911`.
- One candidate runs on each of four idle single-card RTX 3090 hosts. The busy
  `3090-qi` GPUs are not preempted and may only be used after their current jobs
  release a card.
