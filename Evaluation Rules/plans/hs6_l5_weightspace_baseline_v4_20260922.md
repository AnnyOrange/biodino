# HS6-L 5TB weight-space (parameter-averaging) baseline for the E/M/L coexistence diagnostic (v4)

Status: user-requested 2026-09-22 ("加一个特别便宜的 weight-space baseline"); executed immediately
as a companion of `hs6_l5_early_late_coexistence_v4_3090_20260921`. Diagnostic only: it asks
whether asynchronous early/mid/late capabilities can be merged inside one parameter basin,
not whether averaging is a method.

## Question and arms

- Same-trajectory EMA teachers: E=ck12687, M=ck20007, L=ck29279 (SHA256 in the coexistence
  `launch_manifest.json`; re-verified before merging).
- `theta_alpha = (1-alpha) theta_M + alpha theta_L`, alpha in {0, 0.25, 0.5, 0.75, 1}.
  alpha=0 is M and alpha=1 is L byte-for-byte, so their v4 cells are reused from the
  coexistence campaign after identity checks; only WA025/WA050/WA075 are new checkpoints.
- `theta_avg3 = (theta_E + theta_M + theta_L)/3` (AVG3).
- Merge in float64 over every stored tensor, stored as float32 in the original
  `{"teacher": state_dict}` layout; formula re-verified on reload (max abs err 2e-6).
  Backbone geometry recorded in `checkpoints/checkpoint_manifest.json`.
- Comparison columns per v4 cell: E, M(=alpha 0), alpha .25/.5/.75, L(=alpha 1), AVG3, and the
  feature-fusion arms E+L / M+L / PCA(E+L) from the coexistence campaign.

## Protocol (unchanged v4; identical pinned code)

Every extraction runs from the coexistence campaign's immutable snapshots with the exact
arguments of its launchers: frozen classification/regression (`source_snapshot`, B64 BF16,
workers 2, BLAS 1, seed 0, auto/TTA8, dataset-best resolution, current split, `--save-paths`);
within-set retrieval/clustering + HPA (`source_snapshot_retrieval_v3`, B64, cosine, CPU metrics);
detection proxy (`source_snapshot_dense_v4`, B8 224 stretch, 5 epochs, AdamW 1e-3/1e-4, seed 0);
v3-only segmentation (`source_snapshot`, formal-v1 splits incl. PanNuke 3 rotations, B32/B32,
E20+E50, seeds 0/1/2, eval every epoch, test once). Nothing is lowered for 24 GiB cards.
Identity-matched pairing (ordered sample paths + labels equal to the E/M/L banks) is done on CPU
from `source_snapshot_weightspace_v1` (= `source_snapshot_lc_v7` + pairing modules); the E/M/L
single-arm probes are re-run there and must reproduce the coexistence fusion JSON exactly.

Inventory: classification 25 + regression 4 (x4 arms = 116), retrieval/clustering 5 datasets
(4 within-set + HPA; rxrx1/rxrx3 have no E/M/L baseline and are listed as not scheduled),
detection proxy 3 (bbbc038 x4 arms; conic/livecell E/M/L + 4 arms since no baseline existed),
segmentation 6 (cellpose, conic, livecell, multimodal_cellseg, tissuenet, pannuke; MoNuSeg
BLOCKED_NOT_TESTED). CTC/OOD not scheduled (no baseline evaluator run). 178 GPU tasks.

## Execution

Output root `outputs/02_eval_runs/hs6_l5_weightspace_baseline_v4_20260922` with
`campaign_manifest.json`, `tasks.json`, per-task `claims/<id>/status.json` (host, GPU, pid,
command, rc, done-check), `logs/`, `node_telemetry/<host>.jsonl` (start + 30 min samples).
Shared claim queue (atomic mkdir) drained by workers on local 5090s (all 8 cards), 3090-qi
(cards 0-7, respecting resident jobs) and single-3090 fleet nodes whose card is <50% used.
Admission per launch: memory used <60% and >=8 GiB free, <=5 campaign processes per card,
launches serialized per card with a 75 s settle; segmentation additionally needs >=80 GiB RAM,
>=4 GiB /tmp and a per-host cap. Two consecutive failures stop a worker; six FAILED tasks
stop all claiming. No checkpoint/dataset transfer; no deletion or overwrite of any output.
