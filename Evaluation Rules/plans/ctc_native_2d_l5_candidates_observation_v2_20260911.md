# CTC native 2-D L5 observation: computational amendment v2

Status: `COMPUTATIONAL_AMENDMENT_LOCKED_BEFORE_V2_RUN`.

## Scope of the amendment

- The v1 candidate set, folds, head, training schedule, native inference,
  post-processing, linker, metrics, and aggregation were locked before any CTC
  metric was observed. V2 changes none of them.
- V1 was stopped as incomplete after its pinned py-ctcmetrics evaluator became
  non-scalable on the first HSC result (38,579 to 57,338 fragmented tracks),
  and one worker exited before committing a domain result.
- Before this amendment was locked, a diagnostic requested-only score was seen
  for ck17079 on HSC. It is not used to change candidates or hyperparameters.
  The amendment is therefore computationally neutral but not described as a
  new pre-result scientific preregistration.

## Fixed v2 scoring implementation

- Keep py-ctcmetrics at commit
  `59481c48a62d4376fe34bed3e3606b4ec4d60972`.
- Call its `load_data`, `det`, `seg`, `tra`, and
  `count_acyclic_graph_correction_operations` primitives directly. Do not
  compute merged-track products because none of Valid/DET/SEG/TRA consumes
  them.
- Replace only repeated parent-track and per-frame label scans in
  `create_edge_mapping` with lookup tables. The emitted edge table must be
  exactly equal to the pinned implementation on equivalence tests.
- Use ordered `multiprocessing.Pool.starmap` mask matching with 16 workers.
  Result ordering and arithmetic are unchanged.
- Supply the audited manylinux CPython 3.11 imagecodecs 2026.3.6 wheel with SHA256
  `e30a14aa2e1c6c90e00375292726486c1d90bf003b1414d608ea4d1f62fd8a79`.

## Acceptance and output

- Unit tests must establish exact edge-table equality and exact requested
  metric equality against pinned `calculate_metrics` on a graph containing
  continuations and divisions.
- Repeated full-HSC diagnostics must agree on every metric and AOGM count.
- A result remains `OBSERVATIONAL_NATIVE_2D`; it is not the formal 20-domain
  2-D/3-D CTC row.
- V2 output directory:
  `outputs/02_eval_runs/ctc_native_2d_l5_candidates_observation_v2_20260911`.
