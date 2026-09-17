# Unprotocolized Dataset Expansion

Status: **PARTIAL IMPLEMENTATION; NO FULL SWEEP OR FROZEN MODEL MATRIX**.
This development registry does not amend formal `protocol_v3.json` or admit
proxy tasks into main rankings. A reviewed split is not a completed benchmark.

## Sources And Decisions

The machine-readable primary-source reviews are:

- `unprotocolized_external_audit.json`: HEST, OpenCell, Transloc and additional local candidates.
- `unprotocolized_representation_audit.json`: CytoImageNet, AllenCell and registration datasets.
- `unprotocolized_ood_audit.json`: all four local OOD task categories.

Each entry distinguishes `OFFICIAL`, `ESTABLISHED_CONVENTION`, and
`PROPOSED_BY_US`, with source URLs, grouping, metrics, local evidence and blockers.
Unknown folds/licenses are unresolved, never inferred from software licenses.
Generate the requested full summary table, with absent results explicitly absent:

```bash
python scripts/audit_unprotocolized_datasets.py --output /path/to/new/audit-output
```

## Implemented Adapters

HEST reuses sklearn PCA/scaling/Ridge with the upstream recipe: train-only
StandardScaler + PCA256, log1p expression (not library-size normalization),
Ridge `lsqr`, no intercept, max_iter1000 and `alpha=100/(256*50)`. Targets are
barcode/gene aligned, never matched by incidental file order. Retain released
sample folds; patient grouping requires companion metadata. Preserve mean
gene Pearson including undefined-correlation failure, then equal-fold/task
aggregation. Report original nine tissues separately from the extra local HCC.
This loader/probe has fixture tests; a full 1TB HEST sweep remains unimplemented.

The frozen encoder now accepts explicit zero-based layer taps without changing
legacy integer behavior. `last4` is not `4-even`. Architecture-specific even4
uses the existing dense implementation (S+ depth12:2,5,8,11; depth24:4,11,17,23;
depth40:9,19,29,39). Global vectors concatenate selected CLS tokens and the
last selected patch mean, then use the existing L2/BF16/fp16 convention.

IDCIA count candidate keeps whole treatment conditions across dates/markers/
FOVs/channels together and excludes the four jointly quantified subfields.
Seed0 gives train127/val65/test66 images; condition is not a specimen ID.
Dataset usage terms unresolved: preflight only, no experiment admission.

CellFMCount uses released DAPI-only scope and test IDs; our seeded validation
holdout and frozen count Ridge are separate proposed choices. Real preflight
fails on identical images with conflicting source targets (2839/2876:
262/252). No deduplication, target adjudication or unsafe manifest is implicit.

## Invocation And Gates

Run from a clean detached checkout using the published benchmark commit:

```bash
python -m unittest dinov3.tests.test_unprotocolized_campaign dinov3.tests.test_cellfmcount
python -m unittest dinov3.tests.test_hest_benchmark  # requires optional h5py
python -m dinov3.eval.bio_frozen_eval.protocol_campaign preflight \
  --dataset idcia-condition-count --benchmark-root /path/to/benchmark \
  --output /path/to/new/preflight-output
```

After committing/pushing, synchronize only Git objects into new remote clean
worktrees. Never overwrite original training trees or implement remote patches:

```bash
python scripts/sync_unprotocolized_campaign.py --output /path/to/new/sync-output
```

hxw clean invocations need its existing Git tool directory prepended to `PATH`:
`/home/xzj/git_sync_tools_20260917/bin`. Sync evidence records the Python/Git
executables and package versions. Dependencies/tests passing for frozen count
regression do not certify HEST's optional h5py environment.

The runner requires four-machine commit/registry/environment verification,
admitted checkpoint/config hashes bound to model family and budget, no source
duplicate/group leakage, fresh feature banks and immutable output paths.
`sweep` is 1TB-only, preserves every candidate result, uses validation without
test extraction, and freezes split/preprocessing/layers/evaluator/alpha/metric/
seed/aggregation plus source and sweep hashes. `evaluate` never searches;
5TB requires matching 1TB result, and 20TB matching 5TB result. All budgets
must retain the same model family and exact selected taps. A new model registry
or code commit requires a new consistent campaign, not mixed comparisons.

Current checkpoints: S+ 1TB ck9224 admitted by hashes; H+ hash admission pending.
No S+/H+ 5TB/20TB assets verified. Existing L-scale assets are not substitutions.
The prior ck9224 selection was retrospective on historical tests; disclose that
and do not describe it as a new validation-selected checkpoint.

## Remaining Work

Blocked datasets remain outside experiment admission. Registration requires
actual coordinate transforms and TRE/rTRE, not pooled-vector retrieval;
native box detection requires real AP, not center-grid occupancy. Restoration
regression requires the reconstruction task, not scalar counts. Missing payloads,
hidden official labels, licenses, group identities and contradictory annotations
must be resolved before appropriate adapters and searches are added.

Dataset audit and unit-test PASS do not certify full data, all environments,
checkpoint budget identity, a completed sweep, a frozen protocol or model scores.
