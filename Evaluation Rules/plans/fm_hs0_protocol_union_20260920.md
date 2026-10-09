# FM14 + HS0 old/v3 protocol union

Status: APPROVED BY USER, 2026-09-20.

The reporting target for all 14 external foundation models and HS0 S+/B/L/H+
is the union of the historical 25/4/6/8/3 matrix and Evaluation Rules v3.
Formal-v3, legacy, smoke, proxy, and observation results remain separately
labelled. The union does not promote an excluded historical task into the v3
formal aggregate.

## Per-model execution inventory

- Existing runnable v3 campaign: 38 cells.
- Legacy/observation extension: 9 cells.
- Immediately runnable union: 47 cells.
- Blocked v3 components: 5 cells (RxRx3, MoNuSeg, CTC, X-ray OOD, Cryo OOD).
- Full union target: 52 cells.

PanNuke is represented by its three official rotations, so execution-cell
counts are two larger than dataset/task-pair counts.

## Extension cells

1. classification: LC25000, deterministic legacy stratified 80/20 split;
2. regression: CoNIC cell count and LIVECell cell count proxies;
3. retrieval/clustering: LC25000 and NCT-CRC-HE-100 within-set leave-one-out;
4. segmentation observation: BBBC038 fixed 470/100/100 labelled split;
5. detection observations: BBBC038, CoNIC, and LIVECell center-patch proxies.

Non-dense settings remain batch64, BF16, seed0, final CLS plus final patch mean,
L2 normalization, and the frozen sklearn probes. Detection remains batch8.
BBBC038 segmentation uses the approved independent E20/E50, seeds0/1/2,
best-validation/test-once protocol. Only the corresponding labelled protocol
column may consume each result.

Existing valid v3 cells are reused from `benchmark_runs/retest_20260918`; the
extension campaign never overwrites them. All new results record source/model
fingerprints and are excluded from the formal-v3 aggregate unless their task is
already formally admitted there.
