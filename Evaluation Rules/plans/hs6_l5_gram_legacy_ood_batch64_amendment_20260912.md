# HS6-L5 Gram legacy OOD batch-64 amendment

Status: `LOCKED_BEFORE_ANY_VALID_OOD_RESULT`.

The legacy Gram campaign manifest locks frozen evaluation batch 64, but its
fleet worker overrode OOD to batch 32. The mismatch was detected before any
OOD lane completed. Running batch-32 processes were stopped, and all partial
outputs, claims, terminal markers, and failure records were retained under
each arm's `_invalid_protocol/ood_batch32_20260912T041800Z` directory.

This amendment changes only OOD execution:

- OOD batch is 64, inherited from `FROZEN_BATCH_SIZE` unless explicitly set.
- Dataset, split, channel policy, resolution, readout, seed, checkpoint, and
  all SSL training inputs remain unchanged.
- The corrected OOD lanes run on deepcad GPUs 3/4, where physical GPU 0 is
  available for the evaluator's temporary model initialization.
- Invalid batch-32 artifacts cannot be aggregated and remain audit evidence.

The campaign remains observational. No OOD label is used to train or select
the self-supervised intervention.
