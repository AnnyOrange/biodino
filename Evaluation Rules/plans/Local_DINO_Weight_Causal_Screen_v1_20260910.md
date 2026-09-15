# Local DINO Weight Causal Screen v1

Status: complete. Training and the final core gate completed 2026-09-10.

## Question

Does the standard DINO local-crop CLS objective impose harmful whole-image
invariance on microscopy representations? A constant loss weight is only a
causal probe. It is not, by itself, the proposed contribution.

All representation learning in this screen is self-supervised. Labels are used
only after training for frozen evaluation.

## Label-free signal gate

The frozen nested-crop diagnostic maps each 7x7 local-token footprint to its
exact parent region in a 16x16 global-token grid. The matched target is compared
with a coherent within-image spatial shift and a cross-image mismatch. Inference
is bootstrapped over images, not patches.

The gate was fixed in `scripts/diagnose_local_global_spatial_signal.py` before
the 256-image measurements:

- final-block matched-minus-shifted cosine mean >= 0.03;
- its image-bootstrap 95% CI lower bound > 0;
- final-block hit@1 enrichment over geometric chance >= 3x;
- at least three of four measured layers have positive-margin and positive-
  enrichment CI lower bounds.

Both the official ViT-L initialization and mature HS6-L checkpoint 15374 pass.
The mature checkpoint result rules out the claim that ordinary HS6 simply
destroys all local/global spatial correspondence.

The photometric comparison holds the 256 images and crop coordinates fixed by
isolating crop RNG from augmentation RNG. Independent training-matched
intensity jitter, Gaussian noise, channel dropout, and blur are then applied to
the two views while preserving trackable geometry. For the official
initialization, final-block margin / hit@1 enrichment is 0.06949 / 13.738x
without photometric augmentation and 0.06880 / 13.770x with it. For mature
HS6-L, the corresponding values are 0.24204 / 18.416x and 0.23837 / 18.088x.
The augmented margin 95% CI lower bounds are 0.06279 and 0.22151,
respectively. Thus the gate is not explained by copying an undistorted pixel
crop.

## Training intervention

The three arms are `local DINO weight = 1.0, 0.25, 0.0`. They share:

- official DINOv3 ViT-L/16 initialization;
- seed 20260910 and deterministic WDS resampling;
- the same per-update sample-key digest;
- robust 1/99 normalization, bio-safe augmentation, 256/112 crops;
- 2 GPUs x batch 64 x accumulation 8 = effective global batch 1024;
- the complete HS6 15-epoch LR/EMA schedule, stopped at 1024 updates;
- identical forwards, including all eight local crops when weight is zero;
- checkpoints at updates 255, 511, 767, and 1023.

Only the scalar multiplying the local DINO gradient differs. Global DINO,
iBOT, KoLeo, optimizer, teacher update, data order, and compute path remain
unchanged. `scripts/audit_local_dino_weight_screen.py` verifies this contract.

Lowering the local scalar also lowers total DINO weight relative to iBOT. Thus
this screen identifies a useful objective allocation but cannot, on its own,
attribute a gain uniquely to false local/global invariance. If weight 0.25
passes the core gate, a required follow-up holds total DINO mass fixed (weight
0.25 with an approximately 3x outer DINO multiplier) before making a mechanism
claim.

## Core frozen gate

The following gate was encoded in
`scripts/summarize_local_dino_weight_core_gate.py` before any u255 result was
available:

| Family | Dataset | Primary metric |
|---|---|---|
| classification | BloodMNIST, BBBC048, Cyclops | balanced accuracy |
| regression | BBBC005 | R2 |
| retrieval | HPA-subcellular, RxRx1-cross | Recall@1 |
| clustering | HPA-subcellular, RxRx1-cross | NMI |

Before any u255 result, the multi-row retrieval outputs are pinned to exact
rows: HPA uses global same-gene retrieval and robust-ge10 location clustering;
RxRx1 uses macro-cell-type cross-experiment retrieval and global-perturbation
clustering. The summarizer rejects missing or ambiguous rows instead of taking
whichever row happens to be serialized first.

Protocol: final CLS concatenated with final patch mean, L2 normalization,
dataset-best resolution, batch 64, bf16 extraction, fixed probe and seed 0.
The zero-update anchor is evaluated identically to detect the case where all
trained arms regress.

The zero-update anchor completed before any u255 checkpoint was available. Its
eight fixed cells are: BloodMNIST 0.93357, BBBC048 0.56526, and Cyclops 0.55380
balanced accuracy; BBBC005 R2 0.91018; HPA Recall@1 0.01624 and NMI 0.21395;
RxRx1 Recall@1 0.01489 and NMI 0.53135.

A non-baseline arm enters dense follow-up only if it:

- wins at least 5 of 8 core metric cells against weight 1.0;
- has positive mean delta in at least 3 of 4 families;
- has no family mean delta below -0.01.

The u255 result is an early gate, not a final efficacy claim. The same core gate
must be repeated at u1023.

### u255 early result

The complete machine-readable report is
`outputs/03_comparisons/local_dino_weight_screen_v1_20260910/u255_core_gate.json`.

- Weight 0.25 versus weight 1.0: 6/8 winning cells, 3/4 positive family
  means, median cell delta +0.000451; it passes the preregistered relative gate.
- Weight 0.25 versus the zero-update anchor: 4/8 winning cells and 2/4
  positive family means; it fails the same gate.
- Weight 0.0 versus weight 1.0: 4/8 winning cells despite 3/4 positive family
  means; it fails the relative gate.
- Weight 0.0 versus the zero-update anchor: 5/8 winning cells but only 2/4
  positive family means; it also fails.

All three u255 arms retain essentially identical label-free final-block spatial
signal: matched-minus-shifted cosine is 0.06873, 0.06872, and 0.06874 for
weight 1.0, 0.25, and 0.0, respectively; hit@1 enrichment is 13.674x,
13.616x, and 13.680x. All pass with 4/4 supporting layers. The early result
therefore supports only a provisional "less regression than weight 1.0"
interpretation, not an absolute representation gain or a spatial-mechanism
claim.

## Dense follow-up and interpretation

Passing candidates are compared with weight 1.0 on Cellpose (last block), CoNIC
(even-four blocks), and a BBBC038 detection observation under the approved v2
protocol. A second training seed is required before any positive claim.

The u255 dense follow-up is complete and audited in
`outputs/03_comparisons/local_dino_weight_screen_v1_20260910/u255_dense_followup.json`.
Weight 0.25 minus weight 1.0 test-mDice deltas are -0.0000279 on Cellpose and
+0.0000005 on CoNIC (one win out of two; mean -0.0000137). The BBBC038
observation test-patch-F1 delta is -0.0076 percentage points. These are
effectively tied and provide no early dense evidence for the candidate.

### u1023 final result

The causal audit passed with all 1024 optimizer updates present in every arm,
zero sample-key digest mismatches, identical normalized configs outside the
intervention, finite health metrics, and readable checkpoints. The complete
reports are
`outputs/00_diagnostics/local_global_spatial_signal/local_dino_weight_formal_audit.json`
and
`outputs/03_comparisons/local_dino_weight_screen_v1_20260910/u1023_core_gate.json`.

| Metric | weight 1.0 | weight 0.25 | weight 0.0 | zero-update anchor |
|---|---:|---:|---:|---:|
| BloodMNIST balanced accuracy | 0.93909 | 0.93937 | 0.93845 | 0.93357 |
| BBBC048 balanced accuracy | 0.55943 | 0.55974 | 0.55904 | 0.56526 |
| Cyclops balanced accuracy | 0.55389 | 0.55172 | 0.55341 | 0.55380 |
| BBBC005 R2 | 0.90871 | 0.90948 | 0.90973 | 0.91018 |
| HPA Recall@1 | 0.01568 | 0.01568 | 0.01512 | 0.01624 |
| RxRx1 Recall@1 | 0.01512 | 0.01512 | 0.01534 | 0.01489 |
| HPA NMI | 0.21886 | 0.20935 | 0.21203 | 0.21395 |
| RxRx1 NMI | 0.53165 | 0.53252 | 0.53346 | 0.53135 |

- Weight 0.25 versus weight 1.0: 4/8 winning cells and 1/4 positive
  family means; it fails the final relative gate. Its family-mean deltas are
  classification -0.000525, regression +0.000765, retrieval 0.000000, and
  clustering -0.004321.
- Weight 0.25 versus the zero-update anchor: 3/8 winning cells and 0/4
  positive family means; it fails the absolute gate. The family-macro mean
  delta is -0.000795 with bootstrap 95% CI [-0.001434, -0.000301].
- Weight 0.0 versus weight 1.0: 3/8 winning cells and 1/4 positive family
  means; it fails the final relative gate.
- Weight 0.0 versus the zero-update anchor: 3/8 winning cells and 1/4
  positive family means; it fails the absolute gate.

All three u1023 arms retain the label-free spatial signal with 4/4 supporting
layers. Final-block matched-minus-shifted cosine is 0.06904, 0.06999, and
0.07007 for weights 1.0, 0.25, and 0.0; hit@1 enrichment is 13.621x, 13.637x,
and 13.669x. This is signal retention, not an efficacy gain.

Neither candidate qualifies for u1023 dense follow-up. The preregistered stop
condition is met: constant local-DINO downweighting is rejected as an efficacy
direction in this one-epoch causal screen. No mass-matched control or second
training seed is warranted for this scalar intervention. Because all 1024
updates lie within the original three-epoch learning-rate warmup, this result
is a screening decision rather than a full-budget comparison.

Loss rebalancing and geometry-based local self-distillation already exist in
the general SSL literature, so neither a constant local-loss scalar nor exact
geometric region matching alone is a novelty claim. A method follow-up must
test a microscopy-motivated, crop-scale-conditioned transition between global
CLS and exact parent-region patch targets against both of those controls.

- If weight 0.25 wins while spatial correspondence is retained, test a
  crop-area-conditioned local weight as the actual method.
- If weight 0 wins, replace whole-image local CLS matching with exact
  parent-region teacher patch matching; do not present zero weighting as the
  method.
- If weight 1 wins or both alternatives regress versus the zero-update anchor,
  stop this direction.

The final result follows the third branch: stop constant scalar reweighting.
