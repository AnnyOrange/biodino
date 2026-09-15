# HS6-L5 Gram legacy dataset-test reproduction

Status: `APPROVED_BY_USER_OBSERVATIONAL_RUNNING`.

## Question

Measure the marginal effect of corrected official-structure Gram anchoring on
the exact legacy dataset-test matrix that motivated the historical 5TB claim.
This is an apples-to-apples reproduction, not a formal-v3 aggregate.

## Arms

All new endpoints branch from the full HS6-L5 ck20007 optimizer/EMA state and
consume the same 488-update sample stream.

| Arm | Student start | Gram anchor | Endpoint |
|---|---:|---:|---:|
| historical start | ck20007 | none | ck20007 |
| matched continuation control | ck20007 | none | branch ck20495 |
| official Gram spatial anchor | ck20007 | ck7807 | branch ck20495 |
| official Gram global-anchor ablation | ck20007 | ck17079 | branch ck20495 |

The Gram arms use student crop 256, clean teacher crop 512, normalized
within-image patch Gram, mature-stage weight 2, and proportionally timed
teacher refreshes. No label enters SSL training.

## Locked legacy evaluation

- Runtime registry: `outputs/02_eval_runtime/dinov3_a029eef0_full_registry.GOjODo`.
- Frozen batch 64, bf16, seed 0, channel policy auto, split protocol current,
  and dataset-best classification/regression/segmentation resolutions.
- Lanes: classification a/b/c/d, regression, retrieval, detection,
  segmentation a/b/c/d, and OOD.
- Historical reference: the completed ck20007 cells in
  `outputs/02_eval_runs/hs6_l_5t_full_every_05m_3090qi_v2_fullregistry_20260908`.

The legacy matrix contains datasets and proxy tasks later excluded by formal
v3. They are intentionally retained here only to reproduce the historical
aggregate. Every report from this campaign must remain `OBSERVATIONAL` and
must not be mixed into a formal-v3 table.

## Gates

1. First report the legacy overall and each family using exactly the historical
   aggregation.
2. Separately report retrieval R@1 and clustering NMI/ARI/accuracy per dataset.
3. Do not select a training loss weight or anchor from downstream labels.
4. A longer matched training run is admissible only after the label-free
   spatial gate and the frozen legacy diagnostic are both audited.

