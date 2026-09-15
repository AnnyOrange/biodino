# CTC native evaluation protocol (implementation contract)

Status: `IMPLEMENTATION_PENDING`; this document freezes the prediction and scoring
contract but does not make the CTC row launch-ready.

## Scope and split

- Use all 20 official labelled CTC training datasets, including both 2-D and 3-D.
- Use `outputs/02_eval_inputs/formal_v3/ctc/split_manifest.jsonl` exactly. Fold
  `k` holds out the four sorted domains `domains[k::5]`; sequence `01` from the
  other 16 domains is head-training data and sequence `02` from the four held-out
  domains is test data.
- The backbone is frozen. A decoder is trained separately for each backbone /
  checkpoint and fold; no held-out-domain frame, mask, trajectory, or dataset
  statistic may be used for fitting or threshold selection.

## Fixed instance head

- Reuse the repository `DINOHoVerNet` decoder with final-block spatial tokens,
  binary NP and two-channel HV outputs, `feature_size=32`, `embed_proj=384`, and
  frozen backbone.
- Train 50 epochs with AdamW, seed 0, batch size 8, learning rate `1e-3`, weight
  decay `1e-4`; select no checkpoint or threshold on held-out data. Use the epoch
  50 head.
- Preserve native aspect ratio and pixel geometry. Run tiled 256 x 256 inference
  with 64-pixel overlap and merge logits before watershed. Single-channel images
  are repeated to RGB. Multi-page 3-D TIFFs are encoded slice-wise; predicted
  binary volumes and watershed markers are consolidated with 3-D connectivity.
- Fixed post-processing: NP threshold 0.5, HV energy threshold 0.4, minimum 10
  pixels/voxels. These values are global and may not be tuned per domain or model.

## Deterministic linker

- Link consecutive-frame instances by maximum overlap with Hungarian assignment.
  The cost is `0.7 * (1 - IoU) + 0.3 * normalized_centroid_distance` and pairs
  with IoU 0 and normalized distance above 1 are forbidden.
- Normalize centroid distance by the median equivalent-instance diameter measured
  only on that fold's head-training sequences. Unmatched objects start/end tracks.
- A parent with two unmatched children that each overlap at least 0.1 of the
  parent creates two child tracks with the parent ID recorded in `res_track.txt`.
  Ties are resolved by ascending label ID. The same code is used for 2-D and 3-D.

## Native scoring and admission

- Emit standard CTC `maskNNN.tif` and `res_track.txt` outputs.
- Score `TRA`, `SEG`, and `DET` with `CellTrackingChallenge/py-ctcmetrics` commit
  `59481c48a62d4376fe34bed3e3606b4ec4d60972`; also save instance mDice and
  detection AP from the same predicted instances.
- Report per-domain values and an unweighted 20-domain macro average. CTC enters
  the formal aggregate only after an oracle-format smoke test, all five folds,
  all 20 domains, and a fail-closed completeness validator pass.
- The old `ctc_2d_count_proxy_v1` remains observational and can only shortlist
  checkpoints. It must never be renamed or aggregated as native CTC.
