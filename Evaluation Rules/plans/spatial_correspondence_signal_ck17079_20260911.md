# Frozen spatial-correspondence signal diagnostic

Status: `LOCKED_BEFORE_MEASUREMENT`.

## Question

Does the frozen HS6-L ck17079 representation retain enough within-image
position information to support a spatially corresponding local/global patch
SSL objective?

This diagnostic is label-free. It does not train a probe, use dataset labels,
or select an SSL checkpoint. ck17079 was fixed because it is the
pre-existing RxRx3-core R@5 peak candidate.

## Fixed comparison

- Checkpoint: HS6-L 5TB ck17079, frozen teacher backbone.
- Data: the checkpoint's original unlabeled training mixture.
- Views: 256 global crop and a nested 112 local crop.
- Layers: zero-based blocks 5, 11, 17, and 23.
- Sample: 64 batches of 8 images (512 images), deterministic seed 20260911.
- Arms: no photometric augmentation and the existing `bio_safe` photometric
  policy, run with otherwise identical settings.
- Controls: a coherent nonzero spatial shift in the same image and the same
  coordinates in a different image.
- Inference unit: image; 5,000 image-level bootstrap resamples.

## Locked gate

The signal is considered usable only if the final layer has true-minus-shifted
cosine >= 0.03 with a positive 95% bootstrap lower bound, top-1 spatial-hit
enrichment >= 3x chance, and at least three of four layers have a positive
true-minus-shifted lower bound and spatial-hit enrichment lower bound above 1x.

Passing this gate supports testing a spatially aligned patch objective. It is
not evidence that the objective improves downstream performance; that requires
a matched self-supervised intervention run.

## Outputs

`outputs/00_reports/hs6_l5_spatial_signal_ck17079_20260911/{none,bio_safe}.json`
