# Repaired Dataset Protocols

Reviewed 2026-09-17. This is the authoritative human-readable continuation
record for the expansion candidates, not an amendment of the formal ID ranking.
It supersedes stale missing-image, duplicate-conflict, missing-adapter and
incomplete-HEST statements in `07_Unprotocolized_Dataset_Expansion.md` **only
where a repair is documented below**. Historical audit JSON remains evidence of
the earlier state, not a current readiness certificate.

**Selection status: completed local S+ 1TB development sweeps exist for
CellFMCount (15/15), OpenCell (15/15), HEST SKCM only (9/9),
VGG_Cell_Counting (15/15), and FILM nested selection (10/10). No expansion
protocol is frozen.** The completed winners below are real development results,
not final synchronized benchmark scores. Other datasets retain
`selected_using_1tb=false`, `selected_representation=null` and
`selected_hyperparameters=null`; all retain `frozen=false`. The environment
specification now uses portable headless OpenCV pins (the actual cv2 binary is
unchanged). New VGG/FILM development sweeps use that portable environment;
the three original sweeps retain their historical fingerprint. Do not silently
rebind artifacts to a newer code/config/environment identity. The subsequent
explicit tensor-normalization candidate hash requires a final matching FILM
selection invocation before admission/freeze. No new candidate is ready for
5TB/20TB yet.

All five completed development sweeps used local hostname `server`, admitted
S+ model `hs6-splus-1tb-ck9224`, depth12, extraction batch32, and CPU BF16
extraction (`cuda_available=false` in the saved run evidence). Checkpoint
SHA256 `831aee2f7f296a4184273f3458364ba39879ba1148c5c1e430dbabb690bbba0d`;
training config SHA256
`882f970a80c3c9f6059813931e9c5eee6d4beaf5a88d1e2e70b082da24116121`.
The original checkpoint/config paths, source manifest/config/registry hashes,
package versions and every candidate are retained in each `sweep.json`.
No H+ selection score or frozen 5TB/20TB result is asserted here.
The three original sweeps use environment fingerprint
`e3861fd8a92361edce9695003dfae5240d776974fb9aba41740dbe8b34ef27ad`;
the two new VGG/FILM sweeps use portable fingerprint
`5f1b70182e9063fb91fb2310a24be5a3bd8eef5c4f99b119128a64ef27382ae4`.
Their full code commit is `c05be366a52f2f0159ae3d2b590abe4e75cd180b`.

## Scope And Authority

The inventory covers all 31 entries of `unprotocolized_ood_audit.json`, all nine
entries of `unprotocolized_external_audit.json`, and all seven entries of
`unprotocolized_representation_audit.json`: **41 distinct physical datasets**.
`NF-kB` is Transloc; `BioSR_OOD_copy` is BioSR; `BBBC005_OOD_copy` is the existing
formal BBBC005 task. BBBC021, FMD, DeepLIIF and KimiaPath24C occur in multiple
audits but appear once below. Aliases/copies never receive independent scores or
extra weight in an aggregate.

Executable candidate specifications:

- [repaired_protocol_candidates.json](repaired_protocol_candidates.json): HEST,
  CellFMCount, AllenCell_Morphology, CytoImageNet, OpenCell,
  VGG_Cell_Counting and FILM; runner
  `dinov3.eval.bio_frozen_eval.expansion_campaign`.
- [native_detection_candidates.json](native_detection_candidates.json): BCCD,
  BBBC041; runner `dinov3.eval.bio_detection.campaign`.
- [registration_protocol_candidates.json](registration_protocol_candidates.json):
  ANHIR, CIMA, ACROBAT, CLEM_Reg; native `dinov3.eval.bio_registration` adapters.
  Candidate review plus a registration smoke is not a completed gated hs6 sweep.
- [unprotocolized_protocols.json](unprotocolized_protocols.json): IDCIA candidate
  `idcia-condition-count`, preflight only until usage permission is verified.
  Its old CellFMCount entry is superseded by the repaired registry above.
- [protocol_v3.json](protocol_v3.json): existing BBBC005 protocol; do not rerun a
  duplicate OOD-copy task or replace its formal settings with candidate defaults.

Other machine-readable entries are explicitly **review-only**, addressed as
`unprotocolized_ood_audit.json:datasets[dataset=NAME]` (`O:NAME`) or
`unprotocolized_external_audit.json:datasets[dataset=NAME]` (`E:NAME`). They are
not executable frozen benchmark configurations. `R:AllenCell` refers to the
representation audit. Source URLs and local metadata evidence in these entries
are part of this record.

## Shared Selection And Reproduction Rules

`OFFICIAL` labels only the component actually prescribed upstream. A proposed
probe, repaired split, search or metric does not become official because the
underlying dataset is official. Mixed-component decisions are specified below.

Global-feature candidates reuse the existing frozen extraction/probes, not a new
parallel implementation. `last` selects the final block; `4-even` uses the
existing architecture-dependent `_even4_layers(actual_depth)` (depth12:
2,5,8,11; depth24:4,11,17,23; depth40:9,19,29,39); `last4` selects the final
four blocks and is **not** `4-even`. Global vectors concatenate selected CLS
tokens and the final selected patch mean, then L2-normalize; CUDA extraction is
BF16, feature storage float16. Spatial tasks use normalized patch descriptors,
not CLS vectors. Actual zero-based taps and backbone depth must be recorded.

For executable repaired candidates, the exact preprocessing, feature choices,
search grid, metric, seed and aggregation below are mandatory. Scaling/PCA/probe
fitting use training data only. Validation selects one candidate; training is
not silently refit on validation. FILM selects nine fold-specific recipes using
the corresponding inner folds, never a single global inner-score winner.
HEST's explicitly disclosed development use
of outer folds is the exception to an independent validation-selection claim.
Store every candidate, failures, metrics, hyperparameters, layer taps and model
hashes in `outputs/02_eval_runs/`; freeze requires the complete grid, not only a
winner. Seed0 is the default, HEST uses1, VGG repeats seeds0..4, and FILM repeats
outer seeds0/1/2 with inner folds. All repetitions are retained, not best-seed
selected.

For review-only classification/regression candidates, `last`/`4-even` and the
existing k-NN/logistic/linear/ridge defaults are starting candidates, **not an
approved search grid or selected recipe**. Exact preprocessing, hyperparameters,
seed/repetitions and aggregation remain unresolved unless explicitly specified
below. Image restoration requires a native image-output evaluator; 3D centroid
detection requires native physical-coordinate scoring. Scalar counts, patch
occupancy and image-vector retrieval must not be relabeled as these tasks.

Required progression: admitted **1TB -> full selection -> FREEZE -> 5TB ->
20TB**, with no budget-specific retuning. Native detection heads are trained
fresh for each DINO checkpoint using the identical frozen head-training recipe.
Keep model family/depth compatible; do not silently substitute L checkpoints for
requested S+/H+ assets. Checkpoint budget provenance comes from validated
training configuration and hashes, not a checkpoint iteration or filename.

Local checkout is authoritative. Changes must be locally tested, committed and
pushed, then the exact commit and reproducible environment synchronized on
local, `5090-hxw-xzj`, `5090-lyx-xr`, and `H100`. Before freeze/distributed runs,
verify hostname, Git commit, Python/PyTorch/CUDA/package fingerprint, active
registry hash and runtime compatibility on every machine. A verified Git hash
without compatible dependencies/CUDA is insufficient. Fix remote-discovered
bugs locally and republish; never retain remote-only patches. Do not combine
results from different benchmark commits or frozen protocol hashes.

## Repaired Global-Feature Tasks

### HEST_Benchmark

- Task: morphology-to-50-variable-gene multivariate regression. Upstream folds,
  targets/PCA/ridge baseline and metric are `OFFICIAL`; representation and
  alpha-multiplier selection are `PROPOSED_BY_US`. Primary source:
  [HEST benchmark code](https://github.com/mahmoodlab/HEST).
- Split/grouping: every released tissue `train_N.csv`/`test_N.csv`, grouping by
  `sample_id`, never random spots. All spots of each sample stay together.
  Sample-disjoint folds are verified. Authoritative released anonymized patient
  metadata covers all72 slides, with71 known labels; every official fold passes
  known-patient overlap checks. Only COAD `TENX111` has a blank patient field,
  so complete patient-disjointness is not claimed for its affected folds.
  Patient keys normalize released labels within tissue/cohort, never join
  generic anonymized labels globally or infer/reidentify a patient. Official
  authors describe patient-stratified folds; that assertion remains distinct
  from our explicit released-label audit. Evidence:
  `outputs/02_eval_runs/unprotocolized_continuation_20260917/hest_full_preflight/preflight_patient_checked.json`
  (`patient_grouping`, known-label checks PASS; no known train/test overlaps).
  The authoritative runner requires this patient-checked preflight and verifies
  the pinned metadata and regenerated grouping before experiment admission.
- Data: full ten-task preflight/release payload verification PASS; all 212
  release files (approximately40GB) verified.
  Barcode identities, selected genes, expression matrices and finite
  log1p targets checked in all folds; patches and target barcodes align by ID.
  Evidence: `outputs/02_eval_runs/unprotocolized_continuation_20260917/hest_full_preflight/{preflight,release_verification}.json`.
- Preprocessing: released RGB patches, resize224/crop224, ImageNet RGB,
  normalized frozen features; barcode-aligned `log1p` counts for official
  `var_50genes`. No library-size/total-count normalization.
- Evaluator/search: train-only StandardScaler/PCA256; Ridge `lsqr`, no intercept,
  max_iter1000; alpha=`multiplier*100/(256*genes)`. Search
  `last,4-even,last4` x multiplier `0.1,1,10`, seed1.
- Primary metric/aggregation: Pearson per gene, equal genes per fold, equal
  official folds per tissue, equal original nine tissues. Undefined correlations
  fail rather than silently reducing the gene denominator. HCC is separate;
  any ten-tissue extension must be explicitly labeled.
- Limitation: outer evaluation folds can be used for agreed 1TB protocol
  development, but selected 1TB score is then not independent confirmation.
  SKCM-only selection is not the full nine-tissue benchmark or a frozen
  HEST-wide winner. Data license CC-BY-NC-SA4.0.
- Machine entry: repaired registry `HEST_Benchmark`,
  `hest-official-pca256-ridge-v2`. Selected settings pending full actual sweep.
  SKCM development subtask completed9/9 candidates in
  `outputs/02_eval_runs/unprotocolized_continuation_20260917/selection_splus_hest_skcm_c89258f/sweep.json`:
  `last4` (layers8/9/10/11), alpha_multiplier10 (alpha0.078125), seed1;
  mean gene Pearson **0.5158728977041481**, fold0 **0.588080860799204**,
  fold1 **0.44366493460909223**. It uses the released outer folds for agreed
  protocol development, not independent validation. Status
  `DEVELOPMENT_NOT_FROZEN`; full canonical tissue selection remains pending.

### CellFMCount

- Task: native DAPI scalar cell-count regression. DAPI scope and source test
  membership derive from the official release; quarantine/validation/frozen
  Ridge are `PROPOSED_BY_US`. Sources:
  [official release](https://zenodo.org/records/17088532),
  [author repository](https://github.com/NRT-D4/CellFMCount).
- Root cause: `DUPLICATE_WITH_CONFLICTING_SOURCE_LABELS`, not a local conversion,
  extraction, parser or row-join bug. All 3023 extracted images, annotations and
  metadata byte-match the original checksum-verified ZIP
  (MD5 `e87d6247e6459268f5cf4535ec25e709`); all coordinate row counts agree.
  Total431321 source dots: DAPI1373/386702, Cy31428/42797,
  AF488222/1822. All 51 decoded-pixel duplicate groups are byte-identical;
  42 conflict and nine agree. Conflicting members share source biological/count
  definition fields; no legitimate alternate-count definition is demonstrated.
- Deterministic rule: quarantine **every** member of conflicting pixel groups,
  never average/choose a target. Equal-target duplicates retain a released test
  member first, otherwise the smallest source ID. Source maps/XML are not
  released, so author-side mapping versus annotation error cannot be separated.
- DAPI exclusions: six pairs `(2840,2877),(2839,2876),(2842,2882),
  (2843,2883),(2841,2879),(2846,2884)`, 12 observations (nine trainval,
  three test), 3137 dots. Preserve surviving official test; numeric-sorted
  trainval IDs with default_rng(seed0), ceil10% validation. Final
  **980train/109val/272test**, 1361 images/383565 dots. All encoded and
  decoded identities are split-disjoint. Biological specimen/donor IDs absent.
- Preprocessing: existing grayscale minmax RGB, mean-color square padding,
  resize256/crop256, ImageNet RGB. Finite edge clicks retained for scalar count
  only, with three remaining edge-click diagnostics; no localization claim.
- Evaluator/search: existing train-fitted scaler/Ridge,
  `last,4-even,last4` x alpha `0.01,0.1,1,10,100`, seed0.
  Primary pooled image MAE; R2/Spearman secondary. Curated test272 is not
  untouched official test275. Usage CC-BY-SA4.0.
- Machine entry: repaired registry `CellFMCount`,
  `cellfm-dapi-source-quarantine-ridge-v2`. Evidence under
  `outputs/02_eval_runs/unprotocolized_20260917/repairs/`:
  `cellfmcount_source_audit.json` plus `cellfmcount_manifest.json`.
  Manifest internal SHA256
  `27b9a3e3f80a6a9e9972eab72912c9e627257aee99794cfbac7db2b153209248`.
  Completed15/15 local S+1TB development candidates are retained in
  `outputs/02_eval_runs/unprotocolized_continuation_20260917/selection_splus_cellfm_c89258f/sweep.json`.
  Development winner: **4-even**, layers2/5/8/11, **alpha100**, seed0,
  validation MAE **67.06449078201155** (R2 **0.946957427513389**,
  Spearman **0.8717021978859725**). `selected_using_1tb=true` for this
  development sweep, `frozen=false`; source test was not used for selection.
  Code `c89258f7078a03fd20f2aa5a229f7c8c5ff8ab1b`, extraction batch32,
  historical environment hash
  `e3861fd8a92361edce9695003dfae5240d776974fb9aba41740dbe8b34ef27ad`.
  Portable final-environment selection/freeze remains required.

### AllenCell_Morphology

- Task: `log1p(cell_volume)` regression, `PROPOSED_BY_US`, not an official
  Allen volume-regression benchmark. Sources:
  [CHAMMI author release](https://github.com/chaudatascience/channel_adaptive_models),
  [Allen usage terms](https://www.allencell.org/terms-of-use.html).
- Leakage cause: original CHAMMI cell-row splits shared3502 FOV IDs. Train and
  Task_one share587 wells; neither shares wells with Task_two. Therefore merge
  Train/Task_one for development, hold out ceil10% whole `PlateId:WellId`
  groups by SHA256(seed0:group), retain Task_two unseen structures as test.
  Keep every FOV/cell of a well together; do not assert plate-disjointness.
- All65103 images fully byte/pixel audited. Remove30 equal-target decoded
  duplicate copies; no conflicting targets. Final
  **39126train/4323val/21624test**, 529/59/279 wells. Full loader geometry audit
  verified65073 retained cells, exactly three real WTC11 channels, actual
  `(3,238,374)` shape. Finite nonnegative raw volumes checked.
- Preprocessing: existing flattened multichannel percentile loader, true
  channel dimensions from metadata, encoder channel policy `auto`,
  resize256/crop224, fixed S+1TB microscopy mean
  `[0.514666,0.488834,0.498267]`, std `[0.338707,0.339202,0.336091]`.
  Current registry explicitly mandates these `tensor_mean`/`tensor_std` pins,
  not ImageNet values or budget-dependent checkpoint statistics. No
  channel-count inference from a stitched apparent width.
- Evaluator/search: train-fitted scaler/Ridge,
  `last,4-even,last4` x alpha `0.01,0.1,1,10,100`, seed0.
  Primary pooled cell MAE in log1p-volume units; R2/Spearman secondary.
  Well-macro MAE is an optional separately labeled diagnostic, not a substituted
  primary metric. Custom Allen research/noncommercial terms apply.
- Machine entry: repaired registry `AllenCell_Morphology`,
  `allen-well-disjoint-volume-ridge-v2`; grouped adapter manifest under
  `outputs/02_eval_runs/unprotocolized_20260917/repairs/allen_grouped_manifest.json`,
  internal SHA256
  `cbcfe9fd929b162343c60358b464805b76ab9bd2de0ed21e8ee462a476537b7a`.
  `allen_loader_geometry_audit.json` binds that manifest. Sweep pending.

### CytoImageNet

- Task:894-class weak-label morphology classification. Source official author
  training code uses row-level stratified90/10 and has **64805 original source
  IDs crossing train/val**. The repaired protocol is `PROPOSED_BY_US`, not an
  official crop-random reproduction. Source:
  [author repository](https://github.com/stan-hua/CytoImageNet).
- Grouping/split: released `idx` is original source-image identity; every crop
  and scale stays together. Within each class SHA256(seed0:idx) order,
  ceil10% source groups test and ceil10% validation, rest training. Classes with
  fewer than three surviving groups are explicitly ineligible (actual:none).
- Full890737-file decode and encoded/pixel hash audit PASS:9617 decoded duplicate
  groups,10274 equal-label copies removed,434 conflicting-label observations
  quarantined. All894 classes occur in all three splits. Final
  **702649train/88715val/88665test**, 327969/41351/41351 source groups;
  880029 images. No source/pixel identity crosses splits.
- Preprocessing: released grayscale PNG to RGB, existing ImageNet deterministic
  resize256/crop224. No reconstruction of unreleased upstream channels.
- Evaluator/search: existing normalized cosine k-NN with temperature-weighted
  votes; `last,4-even,last4` x k `10,20,200` x temperature `0.07,0.2`, seed0.
  Primary micro image accuracy, macro-class balanced accuracy secondary.
- Limitations: donor/well/plate identity absent; source-disjoint is not verified
  donor/well OOD. Training-corpus overlap remains a separate audit. Published
  dataset license CC0 does not resolve all upstream component provenance.
- Machine entry: repaired registry `CytoImageNet`,
  `cyto-original-source-grouped-knn-v2`; evidence
  `outputs/02_eval_runs/unprotocolized_20260917/repairs/cytoimagenet_grouped_manifest.json`,
  internal SHA256
  `85d8cd08a7f64d8ff5dfed4851a487ee9ca155f64a6a933394aa36e205dee8f3`.
  No skip-pixel-audit manifest qualifies as complete. Sweep pending.

### OpenCell

- Task: unambiguous major-localization FOV classification, `PROPOSED_BY_US`;
  not protein-identity classification, original localization-clustering ARI,
  or Cytoself cell-crop reproduction. Sources:
  [official projection download](https://opencell.sf.czbiohub.org/download),
  [annotation repository](https://github.com/czbiohub-sf/2021-opencell-figures).
- Pixel source resolved and required bounded acquisition/preflight complete.
  Official public
  `s3://czb-opencell/microscopy/raw/` inventory
  has6301 projections. Required bounded task is **all1064 FOVs of215 selected
  proteins in14classes**, not a missing full stack collection. Projections are
  uint16 `(2,600,600)` CYX, channel0 Hoechst, channel1 tagged protein. Official
  object listing and selected listing retained under
  `benchmark/external_benchmarks_20260901/OpenCell_projections/`.
  All1064 selected projections pass source listing, complete decode/hash and
  metadata preflight. Actual **678train/198val/188test** FOVs, respectively
  **135/40/40 proteins**; all215 proteins are split-disjoint.
- Label/scope: exactly one grade3 localization; other grade1/2 labels allowed.
  Require10 eligible proteins/class; select up to16/class by stable
  SHA256(seed0:ENSG), retain every selected FOV. This threshold/scope is ours.
- Split/grouping: per-class sorted ENSG IDs, default_rng(seed0) permutation,
  ceil15% test, ceil15% validation, resttrain. All cell lines/FOVs of a protein
  stay together. CID/FID are preserved; gene symbols are not grouping keys.
- Preprocessing: full official projection, per-channel minmax; tagged-protein
  red, Hoechst green, bluezero, full-FOV square padding, resize256/crop256,
  ImageNet RGB; never use precomputed Cytoself embeddings as DINO features.
- Evaluator/search: existing logistic probe, `last,4-even,last4` x C
  `0.01,0.1,1,10,100`, seed0. Primary FOV-level macro-class balanced accuracy;
  image accuracy/macroF1 secondary. Protein-averaged accuracy may be separately
  reported. Acquisition-batch grouping is not verified.
- Usage: raw pixels CC-BY-SA4.0 per AWS Open Data registry; figure-analysis code
  has separate Biohub terms. Data license not inferred from that code.
- Machine entry: repaired registry `OpenCell`,
  `opencell-protein-heldout-localization-logistic-v2`; `opencell.py` manifest
  builder. Repair evidence:
  `outputs/02_eval_runs/unprotocolized_continuation_20260917/data_repair/opencell_transloc_protocol_review.md`.
  Source-bound full selected manifest:
  `outputs/02_eval_runs/unprotocolized_continuation_20260917/data_repair/opencell_manifest.json`.
  Full6301-FOV inventory does not imply all6301 were downloaded.
  Completed15/15 local S+1TB development candidates are retained in
  `outputs/02_eval_runs/unprotocolized_continuation_20260917/selection_splus_opencell_a902adc/sweep.json`.
  Development winner: **4-even**, layers2/5/8/11, **C0.01**, seed0,
  validation balanced accuracy **0.6732324223395653** (accuracy
  **0.696969696969697**, macroF1 **0.6630868809195605**).
  `selected_using_1tb=true` for this development sweep, `frozen=false`;
  heldouttest188 was not used for selection. Code
  `a902adca8a9084d8535f512f63c71731da21ab3d`, extraction batch32, historical
  environment hash
  `e3861fd8a92361edce9695003dfae5240d776974fb9aba41740dbe8b34ef27ad`.
  Portable final-environment selection/freeze remains required.

### VGG_Cell_Counting

- Data repaired directly from the primary
  [Oxford VGG release](https://www.robots.ox.ac.uk/~vgg/research/counting/):
  exactly200 RGB cell images and200 paired red-channel binary dot maps,
  native256x256. Archive CRC, every extracted ZIP-member byte hash, annotation
  colors and whole-image counts verified. Count each red255 pixel, including
  adjacent clicks; do not count connected components or average RGB labels.
- Official first100 development/second100 test pools and published N32/five
  draw scope retained. Our exact deterministic draw indices are
  `PROPOSED_BY_US`: seeds0..4 each permute development100, take32train/32val,
  leave36unused, retain same100test. Source image/dot pair is grouping unit;
  all five draws pass encoded/decoded/source-group leakage preflight.
- Existing native `vgg_count` adapter, train-only StandardScaler/Ridge,
  full256RGB FOV/no count-changing crops, ImageNet normalization. Search
  last/4-even/last4 x alpha0.1/1/10/100/1000. Select one representation/alpha
  by equal mean five validation MAEs, not test. Frozen comparison uses the same
  five recipes/draws and reports mean/populationSD of five test MAEs. Those
  share100test images and are not independent test cohorts; no trainval64 refit.
- Machine entry repaired registry `VGG_Cell_Counting`,
  `vgg-n32-five-draw-ridge-v2`. Approved manifest
  `outputs/02_eval_runs/unprotocolized_continuation_20260917/vgg_count/vgg_count_manifest_v2.json`;
  internal signature
  `a0411465d1dd2699919e5cd31b8020b88a1238f76219298b7575c9e5d5b293be`.
  Earlier v1manifest superseded. Full200image histogram-feature CPU adapter
  smoke is not hs6 selection. Actual S+1TB development completed15/15 candidates:
  `outputs/02_eval_runs/unprotocolized_continuation_20260917/selection_splus_vgg_c05be36/sweep.json`.
  Winner **last4**, layers8/9/10/11, **alpha100**, seeds0..4; equal five-draw
  validation MAE **7.460883855819702**, mean R2 **0.9724583381963836**, mean
  Spearman **0.9849677289200246**. Fixed100test was not used for selection.
  `selected_using_1tb=true`, `frozen=false`, status `DEVELOPMENT_NOT_FROZEN`;
  code `c05be36`, CPU BF16, batch32, portable fingerprint recorded above.
  Final matching authoritative-code/config admission and four-machine
  synchronization are required before freeze/evaluation.
- Paper density/MESA method not reproduced by global scalar count probe.
  Source MATLAB code has research/noncommercial terms/citation requirement;
  standalone image ZIP has no explicit separate license. NoCClicense,
  commercial permission or redistribution rights are inferred.

### FILM

- Native age task `PROPOSED_BY_US`, not original lysosomal spectroscopy assay.
  Full official release archive MD5/CRC/member hashes, calibration, all28 finite
  float32 `(126,200,200)` stacks and all normalized loaders PASS. Four ages
  D2/D4/D6/D10,28 FOVs/27 named biological groups; D6sample1's two FOVs stay
  together and their embeddings are averaged before fitting/scoring.
  [Author data](https://api.figshare.com/v2/articles/31302607), CC-BY4.0.
- Repeated stratified sample-group3fold outer CV seeds0/1/2, independent
  stratified group3fold inner CV within each outer training partition. Choose
  **feature/C independently for each of nine outer folds** using only its
  corresponding three inner folds; never globalize one inner-score winner.
  Freeze the resulting nine-entry fold recipe, then use those exact splits and
  per-fold recipes for5/20, no retuning. Frozen feature extraction covers all28
  FOVs in manifest order; each outer fold's test features/labels are excluded
  from its probe fitting and inner validation/selection. Named sample identity
  is verified, hidden animal/batch linkage not.
- All126 bands stay inseparable; ImageJ Z/T tags are storage, not126 independent
  observations. Shared per-stack p1/p99 affine clipping, existing fixed
  `mean3` spectral mean replicatedRGB, resize256/crop224, fixed S+1TB microscopy
  mean `[0.514666,0.488834,0.498267]`, std `[0.338707,0.339202,0.336091]`.
  Current registry explicitly mandates these `tensor_mean`/`tensor_std` pins
  across all budgets. No first3
  bandsRGB, learned/test-fit band choice, or hypothetical native126channel
  encoder substituted for admitted checkpoint. Train-only scaler/balanced
  logistic probe; search last/4-even x C0.01/0.1/1/10/100 perouterfold.
- Primary balanced accuracy over equal-weight named groups, accuracy/macroF1
  secondary; equal outer-fold means then equal seed means, populationseedSD.
  Repeated folds overlap, so dispersion is not independent-cohort uncertainty.
  Released SPEND-denoised inputs and unavailable denoiser training provenance
  remain limitations; spectral-mean RGB deliberately loses chemical information.
- Machine entry repaired registry `FILM`,
  `film-age-nested-grouped3fold-v2`, approved source/CVmanifest
  `outputs/02_eval_runs/unprotocolized_continuation_20260917/data_repair/film_manifest.json`.
  Manifest's earlier source protocol label does not supersede authoritative
  nested-v2 selection rule. Synthetic-feature logistic smoke is not hs6 score.
  Real S+1TB nested development completed10/10 candidates:
  `outputs/02_eval_runs/unprotocolized_continuation_20260917/selection_splus_film_c05be36/sweep.json`.
  Both feature banks match all28 manifest IDs/labels in order. All270 inner
  evaluations and nine selected recipes were independently recomputed and match
  saved evidence exactly. Each recipe maximizes its own outer training
  partition's mean three-inner-fold balanced accuracy; no global feature/C or
  outer-test result is claimed. `selected_using_1tb=true`, `frozen=false`,
  status `DEVELOPMENT_NOT_FROZEN`; code `c05be36`, CPU BF16, batch32, portable
  fingerprint recorded above.

| Outer seed | Outer fold | Selected representation | Zero-based layers | Selected C |
|---|---|---|---|---|
| 0 | 0 | 4-even | 2,5,8,11 | 10 |
| 0 | 1 | 4-even | 2,5,8,11 | 10 |
| 0 | 2 | last | 11 | 100 |
| 1 | 0 | 4-even | 2,5,8,11 | 0.01 |
| 1 | 1 | 4-even | 2,5,8,11 | 0.01 |
| 1 | 2 | last | 11 | 0.01 |
| 2 | 0 | last | 11 | 0.01 |
| 2 | 1 | 4-even | 2,5,8,11 | 1 |
| 2 | 2 | 4-even | 2,5,8,11 | 0.1 |

The c05 FILM artifact actually used the same microscopy tuple above through
checkpoint-config tensor statistics. Its old description incorrectly said
ImageNet; that description/hash is superseded by the explicit current pins.
Do not rewrite or relabel the old artifact. Portable final rerun under the new
authoritative implementation/config hash, complete synchronization and freeze
remain pending. All nine recipes must be reused identically for5TB/20TB.

### Transloc

- Task: continuous nuclear-translocation ratio regression. Released ChAdaViT
  linear-head convention is `ESTABLISHED_CONVENTION`; grouped holdout/Ridge
  extension is `PROPOSED_BY_US`. Sources:
  [ChAdaViT](https://github.com/nicoboou/chadavit),
  [official anonymous image release](https://zenodo.org/records/8287453).
- Continuous targets/source names **recovered** from official CSVs:8457train,
  2114val,10571total,152wells,605FOVs. Published crop split shares151wells.
  Filename encodes day/well/FOV/cell coordinates; binary labels are not ratios.
- Remaining root cause: official complete15999-image anonymous ZIP verifies
  size3154120281 and MD5 `794a5ea07d2a0894a0c59df3249d81dc`, but author conversion
  discarded original names. Required original-name pixels:0/10571 locally.
  Original paper supplementary source workbook was downloaded/parsed: numeric
  ratios/conditions exist but no image/crop/FOV identifiers. This is **not** an
  incomplete local download or parser bug; public verified identity mapping
  remains unavailable. Never fabricate a row-order/dose/class join.
- Proposed split once mapping is verifiable: per-treatment day/well groups,
  sorted then default_rng(seed0), ceil15%test/ceil15%val/resttrain; keep all
  FOV/cell crops in their well. Require exact original-pixel/target joins.
- Intended evaluator/features: existing OLS/Ridge frozen regression,
  `last`/`4-even`/supported`last4`; final alpha grid and multichannel
  preprocessing must be admitted after identity repair, not represented as
  selected now. Primary raw-ratio R2; pooled crop R2, MAE/Pearson secondary,
  well-level diagnostic separate, seed0. No independent official test exists.
- Usage CC-BY4.0. NF-kB alias excluded from independent aggregation.
- Machine entry: `E:Transloc`; `transloc.py` metadata/manifest helpers;
  source proof `.../data_repair/transloc_recovery.json` and repair review above.
  No admitted sweep/freeze until authoritative identity mapping exists.

## Native Registration And Correspondence

Common2D candidate method: full RGB FOV, maxside512 bilinear resize, dimensions
rounded to backbone patch size, ImageNet normalization, no center crop/padding;
map patch centers back to native XY. Reuse existing dense DINO spatial forward;
`last`/`4-even`, L2 descriptors, Euclidean nearest neighbors, mutual matches,
Lowe ratio `0.8,0.9`, RANSAC threshold target-diagonal fractions
`0.002,0.005,0.01`, OpenCV affine estimate, confidence0.99/maxiterations2000,
seed0. Method is an experimental appearance-affine method, **not** an official
nonrigid baseline. Fit failure uses identity and scores every public
correspondence, never drops failed pairs. Parameter selection is pending actual
1TB DINO experiments; SIFT/neighborhood-descriptor CPU smokes are not hs6 scores.

### ANHIR

Official medium cover:481directedpairs,230public training-target pairs,
251private evaluation-target pairs. Allimages/source landmarks exist; hidden
target absence is intentional. Official pair split shares112images and is not
specimen-disjoint. Local **PROPOSED_BY_US** protocol uses public230pairs only,
49tissue groups, SHA256(seed0:group)-ranked25% development, rest evaluation;
CIMA lung/mammary groups cannot be ANHIR development. Preserve pair direction,
all stains/resolutions/pairs of a specimen together. Eight public unequal-count
pairs align common explicit CSV IDs as upstream BIRL does; record unmatched IDs.

Primary: group-macro mean of within-group mean pair median rTRE, official cover
diagonal taking precedence; pair-micro median-rTRE mean, robustness, TRE,
failures/runtime secondary. Official challenge mean rank across methods is
**not** this metric. Public preflight/CPU scoring PASS; locally reproducible
public evaluation is distinct from hidden251-pair challenge-server submission.

License resolved from [official Data Usage Agreement](https://anhir.grand-challenge.org/Data/#data-usage-agreement):
CC-BY-NC-SA2.0, research/noncommercial, attribution/source citations/sharealike.
Do not infer terms from an Unknown mirror license or BIRL software license.
Machine entry registration registry `ANHIR`, adapter `datasets.anhir_pairs`;
evidence `.../registration/anhir_public_preflight_cpu_smoke.json`. Actual1TB
sweep/freeze pending; hidden competition accuracy remains unavailable locally.

### CIMA

Native multistain registration, established108 unorderedpairs/ninetissues at
25%scale; lexicographically earlier filename source, later target. Tissue-group
development/evaluation holdout is `PROPOSED_BY_US` (seed0SHA256,25% development).
Keep allstains/resolutions/pairs of a specimen together; do not pool scales as
independent examples. Primary group-macro within-group mean pair median rTRE
by native target diagonal, pair-micro and group uncertainty separately. Common
2D method/search above; actual1TB selection pending. Full108pair preflight PASS.

[BIRL/CIMA source](https://github.com/Borda/dataset-histology-landmarks);
original-author Kaggle CC-BY-SA4.0, while ANHIR copies use ANHIR terms. CIMA and
ANHIR overlap; no independent cross-dataset macro claim. Machine entry
registration registry `CIMA`, `datasets.cima_pairs`; evidence
`.../registration/cima_preflight_cpu_smoke.json`.

### ACROBAT

Official2022 IHC-to-HE WSI registration:750training/100validation/303released
testcases (not2023mean-p90 protocol). Patient/tumor-block grouping. Local
validation100pairs/200WSIs, **5020official public source landmarks repaired**;
all HE targets blank intentionally. Training has no landmark labels. Native WSI
adapter/pyramid physical-coordinate smoke PASS; local target accuracy cannot be
scored. Challenge server/private targets needed for official validation/test.

Official metric: median across cases of90th-percentile TRE micrometers,
per-landmark average distance across at leasttwo annotators before percentile.
CSV IHC10Xpixels convert once using mpp; Docker world output is micrometers.
Read overview pyramids while retaining native coordinates. Common2D appearance
search is proposed, not validation-selected without targets. SND data release
CC-BY4.0; challenge participation/publication rules remain separate from data
license. Sources: [official challenge](https://acrobat.grand-challenge.org/data/),
[official evaluator](https://github.com/rantalainenGroup/ACROBAT/blob/main/evaluation.py).
Machine entry registration registry `ACROBAT`, adapter `acrobat`; evidence
`.../registration/acrobat_preflight.json`, `acrobat_real_cpu_smoke.json`.

### CLEM_Reg

Native3D FM-to-EM registration. Whole-volume leave-one-volume-out development,
three independent cell/volume pairs, never slice-level train/test. Proposed
calibrated descriptor-affine landmark method is `PROPOSED_BY_US`, not paper
organelle-mask/reference overlay reproduction. ExplicitFMchannel, sampled3D
appearance (no zprojection), calibratedXYZmicrometers; BigWarp landmarks are
already worldXYZ and must not be scaled twice. Search last/4-even, ratios0.8/0.9,
3D RANSAC thresholds0.5/1/2micrometers, `estimateAffine3D`, seed0. Primary mean
whole-volume medianTREmicrometers; physicalEMdiagonal rTRE secondary.

Genuine3D adapter/CPU smoke implemented; volume preflight PASS but
**EMPIAR-11666 BigWarpY exceeds CZI physicalY bounds**: authoritative coordinate
orientation/calibration repair needed before that volume is scored. All-three
selection/freeze and papermask scores remain pending. Entry-specificEMPIAR/
BioStudies terms, not MIT code license; FM S-BSST1175 explicitlyCC0.
Sources: [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12446066/),
[author code](https://github.com/krentzd/napari-clemreg).
Machine entry registration registry `CLEM_Reg`, adapter `volume`; evidence
`.../registration/clem_volume_preflight.json`, `clem_real_3d_cpu_smoke.json`.

## Native Bounding-Box Detection

Shared **PROPOSED_BY_US** evaluator: torchvisionFasterR-CNN, frozen DINO body,
trainable1x1neck256/RPN/RoI, native boxes/classes, pycocotoolsCOCO-style
AP50:95primary (AP50secondary), maxDets300 declared crowded-microscopy extension.
FullFOV resize min512/max1024, ImageNetRGB, frozen bodyBF16 onCUDA, headsFP32,
noaugmentation. Anchors16/32/64/128/256, aspect0.5/1/2. Seed0; search
last/4-even x LR0.001/0.005,10epochs,SGDmomentum0.9,wd0.0005,
StepLR3epochs/gamma0.1. Validation **final-epoch** AP selects; save losses/AP/head
checkpoints for every candidate/epoch. Frozen5/20 retrain fresh heads with the
same recipe; test scored only finalepoch. No center occupancy/count proxy.

### BCCD

Officialsource lists **205train/87val/72test**, source-image grouping; patient
identity absent. All364annotation-linked images decoded/hash/box/class/split
checked. The extra exampleJPEG is outside benchmark. OfficialXML itself has
two degenerateRBC pointboxes: object4ofBloodImage_00343 `[181,329,181,329]`,
object13ofBloodImage_00338 `[504,337,504,337]`. Proposed documented quarantine
removes **only thoseobjects**, retaining bothimages/otherboxes/originalsplits;
never invent one-pixel boxes. This is a cleaned variant, not untouched labels;
the now-unannotated cell region may introduce minor false-positive noise.
[Author source](https://github.com/Shenggan/BCCD_Dataset), MIT.
Machine entry native registry `BCCD`; source-bound approvedmanifest
`outputs/02_eval_runs/unprotocolized_continuation_20260917/repairs/bccd_native_manifest.json`.
ActualS+1TB CPU loss/backward/frozenbody/inference smoke PASS, not benchmarkAP.
Full native1TB sweep pending.

### BBBC041

Released officialtest, proposedSHA256(seed0)-ranked ceil10%training validation:
**1087train/121val/120test**,1328annotation-linkedimages. Retain allseven source
categories including difficult (not a sixclass staging reproduction), allboxes
and negativeimages. Full sourceMD5/decode/box/class/pixel/split preflight PASS.
Portal1364-image release is distinct from annotation-linked1328scope; do not
claim allportalimages scored. Source-image grouping verified; patient/smear/
researcher mapping unavailable, so officialtest is not verified patientOOD.
[Official portal](https://bbbc.broadinstitute.org/BBBC041), CC-BY-NC-SA3.0.
Machine entry native registry `BBBC041`; approvedmanifest
`.../repairs/bbbc041_native_manifest.json`. Full native1TB sweep pending.

## Remaining Reviewed Local Candidates

This table is part of each dataset's protocol record. Together with the shared
rules, it specifies the task, source label, split/grouping, allowed evaluator,
preprocessing constraints, primarymetric/aggregation and limitation. **No exact
search or preprocessing recipe is admitted for these review-only entries**;
selectedfeatures/hyperparameters remain null, seed/repetition unresolved except
where specified, no1TBselection and nofreeze. The VGG/FILM cross-reference rows
are exceptions: their old missing-data/review-only entries are superseded by
the executable repaired protocols above. A source-known task/split is not a
completed executable adapter. Local payload statements below are inventory
evidence, not fullcount/decode certifications or claims that download sources
have been exhaustively revisited during this continuation.

| Dataset / machine entry | Task / source | Split and grouping | Preprocessing / existing task mapping / search status | Metric and aggregation / limitation |
|---|---|---|---|---|
| AllenCell / R:AllenCell | BF-FL retrieval, PROPOSED_BY_US | Local35FOVs,7structures,5each; FOV plus plate/well/day development/eval manifest still needed | ChannelIDs and whole-volume/zprojection rule unresolved; reuse retrieval, last/4-even starting choices only | BidirectionalR@1/5/10,mAP@k,MRR proposed; FOV/plate aggregation/bootstrap, not slices; custom Allen noncommercial terms |
| IDCIA / unprotocolized_protocols:idcia-condition-count | Scalar cellcount, PROPOSED_BY_US | Wholeconditions2train/1val/1test, seed0; 127/65/66images afterfour jointly quantified subfield exclusions, keepdate/marker/FOV/channel pairs together | Existing grayscale minmax/pad/resize256/crop256; scaler/Ridge last/4-even/last4 xalpha0.01/0.1/1/10/100 candidate | PooledimageMAEprimary, RMSE/ACPsecondary; condition is treatment not specimen; fullpreflightPASS, noexperimentadmission untilusage terms verified |
| RSNA_Bone_Age / O:RSNA_Bone_Age | Skeletalage regression, PROPOSED_BY_US localholdout | Official12611train/1425val/200test; locally labeledtrain and unlabeledtest; fixedcaseIDholdout needed | Verifycompleteimage-labeljoin, intensity conversion; reuse ridge/linear, exactgridpending | MAEmonths, aggregationpending; patientlinkage absent, not officialtest; noncommercial/citation |
| VGG_Cell_Counting / repaired registry | Synthetic full-field count, PROPOSED_BY_US probe with official pools/N32 | See repaired VGG entry above:200 verified pairs, five32/32 draws, seeds0..4, fixed100 test | Executable native adapter and fullFOV RGB Ridge; last/4-even/last4 x alpha0.1/1/10/100/1000; actual15/15dev winnerlast4/alpha100 | Five-draw validation MAE7.460883855819702; frozen test mean/populationSD pending; developmentNOTFROZEN |
| BBBC005 / protocol_v3:tier_a.regression | Simulatedcount, OFFICIAL existingprotocol | Existing formal simulation/count/blur grouping and split | Existing formal loader/preprocessing/probe only; no duplicate candidate sweep | Existing formalmetric/seed/aggregation; OODcopy excluded as duplicate, this record doesnot supersede formalprotocol |
| BioSR / E:BioSR and O:BioSR_OOD_copy | Superresolution, OFFICIAL task; splitproposal notadmitted | Cell/specimen/FOV, allnoiselevels/LR-HR/crops together; exactauthorsplitpending | Pairedpixeloutput, preserveLRinput/HRtarget separation; native restorationhead missing, scalarRidge/retrieval notsubstitutes | PSNR/SSIM/dynamicrange/resolution and cellmacroaggregationpending; officialFigshareCC-BY4.0 |
| FPM_INR / O:FPM_INR | Fourierptychographic reconstruction, PROPOSED_BY_US | Measurementspecimen, allillumination/target together; onlytwoconfirmedpairs/sixinput-onlyfiles | MATmeasurement/target conversion verification; native reconstruction missing | Officialmetric/split/aggregation/seedunresolved; metadataCaltechrecord required, notscalarregression |
| RSNA_Pneumonia / O:RSNA_Pneumonia | Opacitybboxdetection, PROPOSED_BY_US labeledtrainingholdout | patientId grouped, allboxes andTarget0negativeimages retained; hiddenofficialtest | NativeFRCNN path reusable butDICOM/TIFprovenance and exactsource/evaluator adapterneeded; gridpending | Officialmeanimageprecision TP/(TP+FP+FN), IoU0.40:0.05:0.75, emptymatching semantics required; notCOCOAP |
| NuCLS / O:NuCLS | Native nucleusdetection/instancesegmentation, OFFICIAL | Correctedsingle-rater5slide/institutionfolds; fold1selection,folds2-5evaluation; patient/slidepatch grouping | CurrentnoQCsubset wrong, correctedlabels/folds needed; nativebox/segmentation framework reusable afteradapter; noinputtargetleak | AP50/AP50:95detection; paperinstance/classmetric and equalfoldaggregationpending; noapprovedbenchmarkscore |
| Spheroid_LSFM / O:Spheroid_LSFM | 3Dnucleicentroiddetection, OFFICIAL | SH-SY5Y30manualtrainpatches/18test; weakLN18-REDpretrain distinct; wholespheroid/spatialorigin grouped, overlapsaudited | KeepXYZ/physicalunits; nativecentroidmatching adaptermissing, no inventedbboxproxy | PAC-MAP centroidmatching threshold/evaluator verificationpending; weak386570pointinventory notmanualtestGT |
| AGAR / O:AGAR | Colonybboxdetection/counting, OFFICIAL | Plate/acquisition, splitunresolved; no usablelocalpayload | Officialportalrequiresauthentication/manualaccess; nativeboxpath afterauthorizeddata/adapter | Officialmetric/aggregationpending; automaticaccess notassumed |
| FILM / repaired registry | C.elegans age classification, PROPOSED_BY_US observational | See repaired FILM entry above:28 verified stacks/27 groups; nested3folds, seeds0/1/2, nine independent fold recipes | Executable native126CHW loader, fixed mean3/microscopytuple, logistic last/4-even x C0.01/0.1/1/10/100 perouterfold; actual10/10dev recipes above | Equal-fold/seed balanced accuracy; finalmatchingconfig rerun/freeze pending; CC-BY4.0 |
| DHM_Erythrocyte / O:DHM_Erythrocyte | nRBC/tRBCclassification candidate, PROPOSED_BY_US | Numericnames notdonorIDs; source23549577/labels/split/lineage unresolved | Existingpixels present, hologram/phasepairmeaning mustresolve beforechannelmapping/classifier | Officialmetric/aggregation/seed/licenseunresolved; norandomimage split admitted |
| Microscopic_Hyperspectral_Choledoch / O:Microscopic_Hyperspectral_Choledoch | HSI/RGBpathologyclassification candidate, PROPOSED_BY_US | Patient/specimenprefix acrossRGB/HSI/ROI; mappingunresolved |45.4GBarchive present, extraction/labels/geometry verificationneeded; no bandselection usingtest | Officialmetric/folds/seed/licenseunresolved |
| ColonCancerHSI / O:ColonCancerHSI | HSIpathologyclassification candidate, PROPOSED_BY_US | Patient/slide grouping required; no payload | OnlyportalHTML/linkmetadata; spectralpreproc/adapter/gridpending | Officialmetric/split/license/aggregationunresolved |
| HMI_LUSC / O:HMI_LUSC | HSIlungpathologyclassification candidate, PROPOSED_BY_US | Patient/slide grouping required; no usablepayload | Figsharemetadata andzero-bytefile1, notimages; spectralpreproc pending | Officialmetric/split/license/aggregationunresolved |
| HistologyHSI_GB / O:HistologyHSI_GB | HSIglioblastomaclassification candidate, OFFICIALresource | Patient/specimen/slide; splitpendingdownload | TCIAresource identified, no currentlocaldirectory/payload; exactadapter/bandpolicy pending | Officialtaskmetric/split/aggregation/license admissionpending |
| OpenSRH / O:OpenSRH | SRHtumourclassification, OFFICIALtask | Must useofficialpatientlists, allslides/tiles together; no currentlocalpayload | Officialrepositoryidentified; tile/patientaggregation/preproc andadapterpending | Exactofficialmetric/aggregationpending, no newrandomtileholdout |
| Annotated_QPI_Cell / O:Annotated_QPI_Cell | QPIsegmentation/classification/detection/count candidate, PROPOSED_BY_US | Acquisition/cell/timeseries grouping required; danglingtaskindex | Referencedphysicalsegmentation directoryabsent, ood_segmentationempty; verifyactualtask/sourcebeforeadapter | Officialmetrics/splits/usageunresolved; taskindex isnotdata |
| RIED / O:RIED | Reaction-enabledrestoration, PROPOSED_BY_US; rejectbiologicalclassifier | Threeacquisitions; modality andputativebiologicallabel confounded | Nativeimage-output restoration/resolution only; notmicrotubule-vsmitochondriaclassification | Nativeresolution/reconstruction metricsunresolved; frames/cropsnotindependentcohorts |
| Altair-LSFM / O:Altair-LSFM | 3Ddeskew/deconvolution, PROPOSED_BY_US; rejectbiologicalclassifier | Acquisition/sample, channels/rawprocessedvariants together | Bead/celldemonstrations notlabelledbiologicalclasses; nativevolume-output evaluatorneeded | Resolution/restorationmetric/aggregationpending |
| Brightness_demixing / O:Brightness_demixing | Singlemoleculemixturedemixing, PROPOSED_BY_US | Acquisition/cell/connectedmoleculeeventacrossframes | Intensity/coordinate tables have noevent/fluorophoreGT; circularintensitylabels rejected | Nativedemixingrecoveryunresolved; supervisedclassification inadmissible |
| SMLDM / O:SMLDM | Synthetictrajectorydiffusivityregression, PROPOSED_BY_US | TrajectoryID/simulationseed, extrapolationdiffusionsettings separately |39settings/about690ktrajectories/onerealPaxillinsequence; no nativeDINOimage task; rasterproxywouldneed explicitnewprotocol | Diffusioncoefficienterror/evaluatorpending; notimageclassification |
| diffractive_sim_hyperspectral / O:diffractive_sim_hyperspectral | WF-SRrestoration, PROPOSED_BY_US; representationproxy separate | Acquisition/pair, allspectralbands grouped | Nativeimageoutputheadmissing; record5filecountdiscrepancy unresolved; retrievedpairs notofficialrestoration | PSNR/SSIMrange/aggregationpending; CC-BY-NC-ND4.0 restrictions requireusage review |
| BBBC021 / E:BBBC021 and O:BBBC021 | MOAclassification/retrieval, OFFICIAL/NSC convention | LOCO acrossalldoses/replicates; plate/well/FOV nested; NSCB additionallyexcludesbatch | Jointhreechannels/image-compound-MOACS V; excludeDMSO from12classes; aggregateFOV->well->treatment explicitly; reuseexistingprobes/retrieval aftercompoundadapter | MOAaccuracy/NSC/NSCB, perwell/pertreatment separately;103noncontroltreatments/38compounds/12classes; missingtrainclassfolds disclosed, nocrop-randomfallback; AstraZenecausage pending |
| FMD / E:FMD and O:FMD | Fluorescencenoisy-to-cleanregression, OFFICIAL | Hold19thFOV/configtest; other19FOVstrain/val; all50rawrepeats/avg2/4/8/16/GT ofFOVtogether | Existingpayload; test_mix/full50 scopesdistinct; native restorationheadmissing; no noiseclassification substitute | PSNR/SSIMofficialrange, declaredconfig/noiselevelmacro; exactval/terms stillpending |
| DeepLIIF / E:DeepLIIF and O:DeepLIIF | IHCinstancesegmentation/quantification plusseparatevirtualstaining, OFFICIALtasks | Released575train/91val/598testsets; BC-DeepLIIF separate; originalslicepatient provenancepending |3072x512sixpanels: IHCinputonly, finalSegtarget; stitchedcompositeinput leaksGT; reuseexistingsegmentation aftermask/paneladapter; imagegeneration separate | Officialinstance/positive-negativecell/IHCscorepostproc andmetrics pending; countarchive/slide/licenseverification required |
| KimiaPath24C / E:KimiaPath24C and O:KimiaPath24C |24slidetextureretrieval/classification, OFFICIAL |22591reference/1325querypatches ofsame24WSIs; notheldoutslidepathologyOOD | Patchpayloadabsent, disabledGitLFS; authorizedsource required; reuseclassification/retrieval afterexactquery/galleryjoin | Publishedpatchaccuracy/scanbalancedaccuracy/product exactcoloredscoringpending; research-onlyagreement retained |
| BloodCell_PBC / O:BloodCell_PBC | Bloodcellclassification, PROPOSED_BY_US | Patient/specimen ifavailable, original/augmentedimagesgrouped; no formalmanifest | Locallessfiles, remote copies reported notverified; resolveBloodMNISTsourceoverlap; classifier/probe aftersourcegroupmapping | Accuracy/balancedaccuracy/macroF1proposed; exactaggregation/seed/terms pending; no independentaggregateuntiloverlap resolved |

Primary source links for the unresolved inventory are retained in the three
audit JSON files; examples include [PAC-MAP](https://github.com/DeVosLab/PAC-MAP),
[NuCLS](https://github.com/PathologyDataScience/NuCLS),
[FMD](https://github.com/yinhaoz/denoising-fluorescence),
[BioSR](https://figshare.com/articles/dataset/BioSR/13264793),
[DeepLIIF](https://github.com/nadeemlab/DeepLIIF),
[BBBC021](https://bbbc.broadinstitute.org/BBBC021), and
[KimiaPath24](https://github.com/KimiaLabMayo/kimia_path24).

## Final Readiness Before Large Batch Testing

Snapshot is deliberately conservative. `Y` means the declared local evaluation
scope is verified, not complete worldwide challenge holdings. `P` means partly
resolved/unverified; `N` means not completed or not admitted. Protocol-resolved
`Y` may refer to a clearly labeled proposed protocol, not an official split.
Leakage `Y` certifies declared released identity/pixel units only, never
unpublished donor identities. `Rules` means this record exists, not that selected
settings have been filled in. Completed development sweeps are shown as
`Y(dev)`. Portable VGG/FILM evidence is distinct from historical original
sweeps; neither certifies final matching-code/config admission or a freeze.
Every new row still has frozen/5TB20TB=`N`.
BBBC005 is already formal and outside the new selection campaign (`existing`).
Paths prefixed `.../registration/` and `.../repairs/` above refer to
`outputs/02_eval_runs/unprotocolized_continuation_20260917/` unless a complete
different path is explicitly provided.

| Dataset | Data complete | Official protocol resolved | Leakage resolved | Adapter ready | 1TB sweep complete | Protocol frozen | Evaluation Rules updated | Ready for 5TB/20TB | Blocker |
|---|---|---|---|---|---|---|---|---|---|
| HEST_Benchmark | Y(ten tasks,212 files) | Y | Y(sample/71knownlabels), P(TENX111blank) | Y | P(SKCM9/9dev) | N | Y | N | Full canonical nine-tissue selection/freeze; portableenvironment; onlyCOAD TENX111 patientlabel unresolved |
| OpenCell | Y(bounded1064) | Y(proposed) | Y(protein/pixel), P(batch) | Y | Y(dev15/15) | N | Y | N | Portable final-environment selection/freeze; oldenvhash differs |
| Transloc | N(originalidentity) | Y(proposed) | P | P | N | N | Y | N | Authoritative original-name pixels or verified name mapping |
| AllenCell_Morphology | Y | Y(proposed) | Y(well/FOV/pixel) | Y | N | N | Y | N | Actual1TB sweep/freeze |
| CytoImageNet | Y | Y(proposed) | Y(source/pixel), P(donor) | Y | N | N | Y | N | Actual1TB sweep/freeze; pretraining overlap limitation |
| CellFMCount | Y(cleanedDAPI) | Y(proposed) | Y(pixel), P(specimen) | Y | Y(dev15/15) | N | Y | N | Portable final-environment selection/freeze; no original specimen map |
| ACROBAT | Y(public100cases) | Y | Y(case) | Y(prediction) | N | N | Y | N | Hidden target accuracy/selection requires authorized server |
| ANHIR | Y(public230pairs) | Y(publicproposed) | Y(specimenholdout) | Y | N | N | Y | N | Actual1TB selection/freeze; hidden251targets server-only |
| CIMA | Y(108pairs) | Y(convention/proposedholdout) | Y(tissue) | Y | N | N | Y | N | Actual1TB selection/freeze; ANHIR overlap |
| CLEM_Reg | P | Y(proposed3D) | Y(volume) | P | N | N | Y | N | EMPIAR11666 physicalY calibration;3volume sweep |
| AllenCell | P | N | P | P | N | N | Y | N | Channel/projection/FOVdevelopment/galleryprotocol |
| BCCD | Y(cleaned364images) | Y(officialsplit/proposedAP) | Y(image/pixel), P(patient) | Y | N | N | Y | N | Fullnative1TB sweep/freeze |
| BBBC041 | Y(annotationlinked1328) | Y(officialtest/proposedval/AP) | Y(image/pixel), P(patient) | Y | N | N | Y | N | Fullnative1TB sweep/freeze |
| IDCIA | Y(258scope) | Y(proposed) | Y(condition/FOV), P(specimen) | Y(preflightonly) | N | N | Y | N | Usage permission unresolved; no experimentadmission |
| RSNA_Bone_Age | P | P | P | P | N | N | Y | N | Labeledholdout/imagejoin/patientprovenance |
| VGG_Cell_Counting | Y(200pairs) | Y(officialpools/proposedindices) | Y(syntheticimage/pixel) | Y | Y(dev15/15portable) | N | Y | N | Finalmatching authoritative-code/config admission, fullsync/freeze; devwinnerlast4/alpha100 |
| BBBC005 | existing | existing | existing | existing | existing | existing | Y(existingreference) | existingformalonly | OODcopy isnotnewtask; useformalregistry |
| BioSR | P | P | P | N | N | N | Y | N | Native restorationhead/exactsplit/metrics |
| FPM_INR | P | N | P | N | N | N | Y | N | Native reconstructionGT/evaluator |
| RSNA_Pneumonia | P | P | P | P | N | N | Y | N | Exactofficialimageprecision andpatientmanifest |
| NuCLS | N(correctedsubset) | Y | P | P | N | N | Y | N | Correctedannotations/folds/sourceadapter |
| Spheroid_LSFM | P | P | P | N | N | N | Y | N | Native3Dcentroidscore/overlap/manualGTaudit |
| AGAR | N | N | N | N | N | N | Y | N | Authorized dataaccess |
| FILM | Y(28stacks/27groups) | Y(proposednestedCV) | Y(namedgroup/pixel), P(animal/batch) | Y | Y(dev10/10portable) | N | Y | N | Finalexplicit-pinned-normalization candidatehash rerun/fullsync/freeze; nine verifieddevrecipes above |
| DHM_Erythrocyte | P | N | N | N | N | N | Y | N | Source/labels/donor/usage |
| Microscopic_Hyperspectral_Choledoch | N(extraction) | N | N | N | N | N | Y | N | Archiveextraction/patient/class/sourceprotocol |
| ColonCancerHSI | N | N | N | N | N | N | Y | N | Authorizedpixelpayload/sourceprotocol |
| HMI_LUSC | N | N | N | N | N | N | Y | N | Zero-bytepayload/sourceprotocol |
| HistologyHSI_GB | N | N | N | N | N | N | Y | N | TCIAdownload/patientprotocol |
| OpenSRH | N | P | N | N | N | N | Y | N | Download/officialpatientlists/evaluator |
| Annotated_QPI_Cell | N | N | N | N | N | N | Y | N | Danglingphysicaltaskdirectory |
| RIED | P | N(native) | P | N | N | N | Y | N | Confoundedclassifierinadmissible;nativerestoration |
| Altair-LSFM | P | N(native) | P | N | N | N | Y | N | No biologicalclassbenchmark;nativerestoration |
| Brightness_demixing | P | N | P | N | N | N | Y | N | No event/fluorophoreGT; circularlabelsinadmissible |
| SMLDM | P | N(DINOimage) | P | N | N | N | Y | N | Native trajectorytask notimagebenchmark |
| diffractive_sim_hyperspectral | P | P | P | N(native) | N | N | Y | N | Filecount/usage/native restoration |
| BBBC021 | P | Y | P | P | N | N | Y | N | CompoundCV/channeljoin/usage |
| FMD | P | P | P | N | N | N | Y | N | Native restoration/exactvalidation/usage |
| DeepLIIF | P | P | P | N | N | N | Y | N | Paneltarget/slidejoin/scoring/usage |
| KimiaPath24C | N | P | P | N | N | N | Y | N | Authorizedpatchpayload/exactcoloredmetric |
| BloodCell_PBC | N(local) | N | N | N | N | N | Y | N | Verifyremotepayload/sourcegroups/BloodMNISToverlap |

Before changing any `N` to ready, persist consistent source-bound manifest,
preflight, all-machine environment/code proof, complete1TBsweep and signedfreeze;
update this document with actual selectedrepresentation/hyperparameters and
sweep/frozenartifactpaths. Record every run's hostname/commit/checkpoint/dataset/
configID/features/hyperparameters/seed/metric/output/status. Final result tables
must keep unavailable results null and include protocol status, machine and
blocker. No readiness table entry or import-only PASS replaces real evaluation.
