# HS6-L5 dual-anchor legacy dataset-test addendum

Status: `LOCKED_BEFORE_DUAL_ENDPOINT_EVALUATION`.

## Question

Does the label-free dual-anchor candidate retain the early anchor's spatial
benefit while improving the fixed-endpoint retrieval and clustering behavior
under the same observational legacy dataset test?

## Locked arm

- Student/EMA/optimizer start: HS6-L5 ck20007.
- Endpoint: ck20495 after exactly 488 matched updates at effective global batch
  64 and the same per-update sample stream as the matched control.
- Patch Gram anchor: ck7807, clean 512-pixel teacher crop, weight 2.0.
- Frozen global relation anchor: ck20007, clean global crops, weight 0.5.
- No label enters SSL training or selects either anchor or loss weight.

## Evaluation

Run the same 12 frozen-evaluation lanes, runtime registry, seed, batch size,
channel policy, split protocol, and resolution protocol locked in
`hs6_l5_gram_legacy_dataset_test_20260912.md`. Store the dual arm in a separate
campaign root and join it to the original comparison only through the
aggregator's `--extra-arm` input. This preserves the original three-arm
campaign manifest.

This matrix remains `OBSERVATIONAL`. Frozen labels can reject the candidate,
but cannot tune the anchors, weights, relation gate, or spatial gate.
