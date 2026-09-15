# HS6-L 5TB label-free spatial-consistency curve

Status: `EXPLORATORY_EXTENSION_LOCKED_BEFORE_REMAINING_48_MEASUREMENTS`.

The 512-image ck17079 gate in
`spatial_correspondence_signal_ck17079_20260911.md` was observed before this
extension was specified. This scan therefore remains hypothesis-generating;
it is not a fresh confirmatory checkpoint-selection experiment.

## Fixed scan

- Scan all 49 available HS6-L 5TB teacher checkpoints.
- Use the same 128 unlabeled training images and deterministic nested crops at
  every checkpoint: seed 20260911, 16 batches of 8, zero data workers.
- Use 256 global crops, nested 112 local crops, and the existing `bio_safe`
  photometric policy.
- Read zero-based blocks 5, 11, 17, and 23 from the frozen backbone.
- Compare the true parent region with a coherent same-image spatial shift and
  a cross-image control. No labels or trained probes enter this scan.
- Use 2,000 image-level bootstrap samples per checkpoint.

## Locked summaries

- Primary label-free score: block-24 mean true-minus-shifted cosine.
- Secondary score: block-24 top-1 spatial-hit enrichment over chance.
- Report the raw curve and a three-checkpoint centered median of the primary
  score; do not optimize a weighted composite after seeing the curve.
- For hypothesis assessment only, report Spearman correlations of the primary
  score with the already-computed common-33 non-dense mean-rank curve and
  RxRx3-core R@5. These labels never select samples, crops, losses, or model
  weights.
- Report values for ck15615, ck17079, ck20007, ck21959, and ck23911 explicitly.

## Interpretation

A declining label-free spatial score would support using spatial consistency
as an anchor/update trigger. Correlation with downstream curves is exploratory
because those curves motivated this diagnostic. A matched SSL intervention
and a held-out model family are required before treating the trigger as a
method contribution.

Output directory:
`outputs/00_reports/hs6_l5_label_free_spatial_curve_20260911`.
