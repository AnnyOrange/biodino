# HS6-L spatial peak-to-endpoint paired diagnostic

Status: `POSTHOC_PAIR_LOCKED_BEFORE_RERUN`.

The all-49 exploratory spatial curve was already observed and selected ck7807
as the raw block-24 true-minus-shifted peak. This rerun therefore estimates the
paired difference on identical unlabeled images; it does not turn the selected
peak into a confirmatory result.

- Compare ck7807 with endpoint ck23911.
- Reuse exactly the all-49 scan settings: 128 images, seed 20260911, `bio_safe`,
  256/112 nested crops, blocks 5/11/17/23, and 2,000 within-checkpoint summary
  bootstrap samples.
- Persist image keys and every per-image diagnostic value.
- Require the ordered image keys and local crop areas to match exactly.
- Primary paired value: block-24 true-minus-shifted cosine.
- Report its mean difference and a 50,000-sample image bootstrap interval.
- Also report a two-sided paired sign-flip p-value as descriptive only; it is
  not corrected for selecting ck7807 from 49 checkpoints.
- No labels or trained probes are used.

Outputs:
`outputs/00_reports/hs6_l5_label_free_spatial_curve_20260911/paired_endpoint`.
