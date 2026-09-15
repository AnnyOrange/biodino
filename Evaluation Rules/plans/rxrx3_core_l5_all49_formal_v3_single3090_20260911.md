# RxRx3-core HS6-L 5TB all-49 formal v3 campaign

Status: `PRECOMMITTED_BEFORE_RUN` (2026-09-11).

## Question and scope

- Measure the full HS6-L 5TB checkpoint trajectory on the newly admitted
  `rxrx3-core` Tier-B retrieval task.
- Run all 49 fixed 0.5M-visit checkpoints, `487 + 488*k` for `k=0..48`.
- Do not reuse the prior 37 results because their result records have no
  campaign-manifest fingerprint.

## Frozen protocol

- Backbone: consolidated `teacher` state from the existing HS6-L 5TB run.
- Readout: final CLS concatenated with final patch-token mean, followed by L2
  normalization.
- Input: six channels normalized independently by p01/p99, then pair-mean
  `(1,2), (3,4), (5,6)` into RGB.
- Split: all 734 eligible genes, one gallery and one query well per gene,
  plate-disjoint within gene. Same-well and same-plate positive pairs are
  excluded.
- Encoder input: crop 224 after resize 256; logical and physical batch 64.
- Retrieval: cosine ranking. Clustering: MiniBatchKMeans with `n_init=5`,
  `max_iter=200`, seed 0.
- Fixed split manifest SHA256:
  `94d570cb66d71de20e9ded203d8727623fe85a807d8b5b9cdbe0de3fce1318f5`.

## Execution and validation

- Initial worker: one idle RTX 3090 on `cpu2`; `3090-qi` may take an additional
  lane only after its current training workload releases GPUs.
- Shared checkpoint, data, and output paths are read directly; no model or data
  transfer is permitted.
- Output:
  `outputs/02_eval_runs/rxrx3_core_formal_l5_all49_v3_single3090_20260911`.
- Before the first model starts, `campaign_manifest.json` locks each checkpoint
  path, size, mtime, and SHA256 plus config, evaluator, data, and split hashes.
- A result is valid only when it fingerprints that exact immutable campaign
  manifest and passes the independent 49/49 fail-closed validator.
