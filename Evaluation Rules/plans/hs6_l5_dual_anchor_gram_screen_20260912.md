# HS6-L5 decoupled dual-anchor Gram screen

Status: `LOCKED_BEFORE_MATCHED_488_UPDATE_RUN`.

## Question

Can an early spatial Gram anchor improve within-image patch geometry without
erasing the cross-image geometry present at the continuation branch point?

This is a fully self-supervised training intervention. Dataset labels and
frozen-probe scores do not select either anchor, the loss weights, or samples.

## Locked method

- Student and EMA state: full HS6-L5 optimizer checkpoint ck20007.
- Patch anchor: ck7807, the raw peak of the pre-existing label-free block-24
  spatial-consistency curve.
- Global anchor: ck20007, fixed mechanically as the continuation branch point.
- Patch objective: official-structure normalized within-image Gram, student
  crops 256, clean teacher crops 512, image-level loss weight 2.0.
- Patch-teacher refreshes: after completed updates 20200 and 20400 in this
  screen; the global anchor is never refreshed.
- Global objective: normalized CLS sample-relation Gram, independently weighted
  at 0.5. Each of the two global views forms its own graph, and equal local
  batches are gathered across all ranks before the relation matrix is formed.
- No legacy same-teacher `inter_image_loss_weight` is active.

## Matched screen

- Updates: 20008 through 20495 inclusive (488 optimizer updates).
- Hardware geometry: 4 ranks x 16 samples, effective global batch 64.
- Data, seed, augmentation, schedule, and checkpoint start match the completed
  control and official ck7807 arms.
- Required integrity: 488/488 finite updates and exact per-update sample-digest
  equality with the matched control.

## Gates

1. The cross-rank relation loss must be finite and nontrivial; all ranks must
   finish a real backward pass before the matched run starts.
2. Compare label-free paired spatial correspondence and frozen-anchor relation
   drift before any labeled task is used for a decision.
3. Frozen retrieval, clustering, and classification are evaluation only. They
   may reject the candidate but cannot tune its anchors or weights.
4. A GB1024 extension requires both label-free gates and the separately locked
   legacy dataset-test audit; this 488-update screen is not evidence of scale.

### Frozen-anchor relation-drift gate

Locked before the dual endpoint exists:

- Use 128 unlabeled HS6-L5 training images and two deterministic clean global
  crops per image (seed 20260912, crop-area range 0.32--1.0, resolution 512).
- Form a separate normalized CLS sample-relation Gram for each crop view over
  all 128 images. The primary drift is mean squared error to the frozen
  ck20007 relation Gram; diagonals are included exactly as in the training
  loss. Also report row-wise off-diagonal drift, anchor top-1-neighbor
  agreement, and anchor top-5-neighbor overlap.
- Required relation pass: dual drift is at most 80% of the official ck7807
  patch-only arm's drift and at most 110% of the matched no-Gram control's
  drift. These ratios are fixed before observing any dual endpoint.
- Required spatial pass: on the existing deterministic 128-image nested-crop
  diagnostic, dual block-24 true-minus-shifted cosine must be no lower than
  the matched control and must retain at least 50% of the patch-only arm's
  improvement over control. Report paired image-bootstrap intervals but do not
  alter the thresholds from them.
- Both gates are label-free. A labeled frozen probe cannot rescue a failed
  label-free gate or tune these thresholds.

Implementation validation artifacts:

- `outputs/01_training_runs/HS6_L5_ck20007_dualgram_pa7807_ga20007_u1_gb1_1xdeepcad_dual_smoke2_20260912`
- `outputs/01_training_runs/HS6_L5_ck20007_dualgram_pa7807_ga20007_u1_gb2_2xdeepcad_dual_crossrank_smoke_20260912`
