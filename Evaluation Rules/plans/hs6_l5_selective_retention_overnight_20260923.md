# 5TB selective retention overnight campaign

User authorization: 2026-09-23 user explicitly delegates local 8×5090 and
3090-qi 8×3090 for autonomous implementation, trials, training and v4 evaluation.
This authorizes the concrete plan below; no additional approval round is required.
The user's >=5 evaluation tasks/GPU and >70% memory target overrides the older
five-job cap. Batch/resolution/layers of formal v4 evaluation remain fixed.

Training: common ck12687 EMA teacher, fresh identical optimizer/head initialization
policy, deterministic 30% 1M/70% 5TB WebDataset stream, LR 1e-4 on the original
15×4098 schedule, start 12688, effective batch 1024. DDP only; hardware smoke
selects the largest common safe real microbatch (initial B64, test B128), no FSDP.
All arms use the same microbatch, accumulation, seed0 and activation checkpointing.
Four initial arms: vanilla; official fixed-anchor Gram weight2; fixed recovery;
adaptive recovery. Local GPU pairs 0,1 / 2,3 / 4,5 / 6,7. Run up to 2440 updates,
save EMA every488, with finite-loss/resource/failure guards. Further variants are
budgeted and selected only by validation/label-free diagnostics, never test scores.

Recovery: local patches and CLS/patch-mean, frozen anchor sees identical pixels
and masks; current implementation uses the live anchor (not an offline feature bank).
Linear ridge decoder uses synchronized running sufficient statistics on even-indexed
images; odd-indexed images monitor prequential residuals. Fit/controller histories
are checkpoint buffers. First32 microsteps calibrate stochastic-depth noise.
Shrinkage and bounded per-coordinate dual weights replace indiscriminate whitening.
Mask-matched recovery and clean512 Gram are distinct auxiliary objectives; this
input difference is disclosed. Vanilla extra-compute control is a later ablation.

Readout tuning: existing E/M/L frozen banks, all four regression datasets;
alpha=10^-4,...,10^6 plus1; same v4 outer split, train-only group-aware inner CV;
BBBC013 nested replicate-row CV with per-compound log1p target. Report fixed
alpha1 v4 result alongside v4-split tuned-readout extension, applied to all arms.
No outer-test selection of penalty/representation/checkpoint. Baseline caches
retain their original source and data hashes.

Evaluation: `bio-eval-union-v4`, teacher/EMA only. Classification25 (LC25000
provisional), regression4, retrieval/clustering7 each, segmentation7 (PanNuke3
rotations), detection proxies3, tracking1, OOD2. Preserve missing/admission states.
Frozen B64/BF16, final CLS||patchmean L2, source split/geometry/TTA8. Segmentation
B32/B32, independent E20/E50, seeds0/1/2, best validation epoch, test once;
detection B8/224/5epochs. OOD/provisional/low-N/proxy results separate.
Reuse immutable coexistence v4 evaluator snapshots where validated; improved
admission (MoNuSeg) uses a separately hashed snapshot and matched baselines.
Initial 3090-qi scheduling fills useful baseline gaps (Gram in v4) and later model
endpoints, starting >=5 independent slots per GPU, adding slots toward70% memory
only with measured headroom. No dummy VRAM allocations or duplicate tasks.

Outputs: training `outputs/01_training_runs/hs6_l5_selective_retention_20260923`;
evaluation/CPU tuning `outputs/02_eval_runs/hs6_l5_selective_retention_v4_20260923`.
Code: `/mnt/huawei_deepcad/dinov3_selective_retention_snapshot_20260923`, copied
from dirty working tree with base SHA/provenance; source fingerprint recorded
before formal launch. Manifests contain exact commands, paths, hashes and seeds.
Retain existing jobs; GPU0 on qi has an unrelated active density regression run.

Theory: for anchor a=Dz+e and fixed linear old readout w, old-logit MSE is bounded
by ||w||² E||e||² (Cauchy–Schwarz). This supports recoverability as a sufficient
surrogate under a bounded readout and population error; it does not guarantee
accuracy, cosine retrieval, or same-width feasibility. Learned decoders and gradient
constraints have CaSSLe/GEM precedents. Novelty/effectiveness remain hypotheses.

Runtime addendum (before reading new-method downstream results): all four B128,
24-block activation-checkpointed DDP smokes passed10 updates; formal runs use
that common configuration. Existing Gram ck13175 and ck15127 are added as
predeclared first/last-budget references, alongside ck20007/29279. Each new
model's completed regression banks enter the same train-only ridge sweep.
Native-final primary CoNIC/LIVECell/PanNuke/MoNuSeg use distinct task/run names;
secondary dataset-best results remain separate. PanNuke follows the protocol's
three fixed train/val/test assignments verbatim (not an inferred cyclic split).
RxRx1 uses the standard full core and RxRx3 the hash-locked734/734 eligible-gene
manifest. Baseline E/M/L have the same added primary/RxRx cells. CTC/OOD gaps and
all56 expected cells per arm are explicitly recorded in V4_INVENTORY.json.
The post-launch runtime scripts are hashed separately in RUNTIME_PROVENANCE.json;
the running training source remains unchanged. Task supervisor adoption and an
at-most-once OOM retry preserve previous attempts. Float32 cross-host regression
parity tolerates1e-3 R² after confirming exact same-host reproduction and <1e-13
float64 cross-host discrepancy on the flagged LIVECell control; original v4
scores remain intact. Existing sample digests audit the last accumulation
microbatch on the logging rank, not every rank/image/augmentation.
