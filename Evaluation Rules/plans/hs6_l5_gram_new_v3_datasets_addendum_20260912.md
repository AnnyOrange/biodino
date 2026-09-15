# HS6-L5 Gram branches on newly admitted v3 datasets

Status: `LOCKED_BEFORE_DUAL_ENDPOINT_AND_BRANCH_CTC_RESULTS`.

## Scope

Complete the new-dataset audit for the matched ck20007 -> ck20495 branches:

| arm | SSL intervention |
|---|---|
| control | matched continuation, no Gram |
| anchor7807 | official within-image patch Gram from ck7807 |
| anchor17079 | official within-image patch Gram from ck17079 |
| dual | ck7807 patch Gram plus frozen ck20007 CLS-relation Gram |

All four endpoints consume the same 488-update unlabeled sample stream at
effective global batch 64. No downstream label enters SSL training or selects
an anchor, loss weight, refresh point, or checkpoint.

## RxRx3-core

The first three arms already have complete formal results under
`crispr-query-guide-plate-disjoint-all-eligible-genes-v1`. Add the dual arm
using the same 734-query/734-gallery manifest, compact-three channel mapping,
batch 64, final-CLS plus final-patch-mean readout, and seed 0. Compare it with
the matched control; do not tune from the result.

## CTC

Run all four endpoints through the already validated native 2-D observation
protocol: ten 2-D domains, five domain-held-out folds, frozen ViT-L backbone,
fixed 50-epoch head, deterministic linker, and native DET/SEG/TRA metrics.
These rows remain `OBSERVATIONAL_NATIVE_2D`; the formal 20-domain 2-D/3-D head
and linker are still pending in protocol v3.

The fixed downstream CTC head is supervised evaluation only. It may reject a
candidate but cannot tune or update the SSL backbone.
