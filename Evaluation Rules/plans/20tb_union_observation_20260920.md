# 20TB evaluation union companion

Status: APPROVED by explicit user instruction on 2026-09-20.

Keep the existing `20tb_raw_online_20260918` campaign as the only formal-v3
campaign. This companion evaluates only the legacy/full-registry difference
and never contributes to Tier A, Tier B, overall means, checkpoint selection,
or main ranking.

The de-duplicated dataset union is reported as
`25/4/7/7/8/3* + CTC1 + OOD2`, where `3*` and every legacy-only cell are
observational. Formal-v3 remains `24/2/5/5/7/0 + CTC1 + OOD2`; components
whose formal readiness gate is not satisfied remain `BLOCKED_NOT_TESTED`.

The difference campaign contains:

- classification: `lc25000`;
- regression: `conic-cell-count`, `livecell-cell-count`;
- retrieval/clustering: `lc25000`, `nct-crc-he-100`;
- segmentation: `bbbc038`, legacy `monuseg`;
- detection observation: `bbbc038`, `conic`, `livecell`;
- OOD observation: `xray`, `cryo`.

Detection follows the 2026-09-20 matched observation amendment: 224 stretch,
final spatial patch map, batch 8, BF16, workers 2, seed 0, fixed center-patch
linear head, AdamW lr1e-3/wd1e-4, five epochs, patch F1. Historical batch-4
results are not reused.

All four 20TB arms are watched online. Every teacher checkpoint is admitted
only after the checkpoint is stable; inputs are symlink adapters and no model
or dataset is copied. The orchestration source, Evaluation Rules, training
config, runtime commit/status, command, environment, and checkpoint source are
recorded. Failures are isolated per arm/checkpoint/lane and never converted to
zero scores.

The formal RxRx3 component is dispatched separately under its fixed v3
manifest. Native CTC remains blocked until the approved fixed head and linker
exist; any count proxy is labelled observational and cannot fill the formal CTC
cell. MoNuSeg official-v3 remains blocked pending official identity evidence;
the old split may only appear in this observation companion.
