# FM14 alignment amendment, 2026-09-17

Authority: user request to stop FM experiments and align all FM models.

- Stop all FM dispatch and evaluation until protocol and environment preflight pass.
- Keep checkpoints, data, all old results and all non-FM training/evaluation.
- Batch-only differences, including feature and probe batch, do not require reruns.
  Record these differences; do not describe probe optimization as batch-identical.
- Proven non-batch differences require targeted supplemental evaluation. Missing
  evidence requires inspection of manifests, logs and source, not a blanket rerun.
- Audit all 14 registered FMs and every task family, including historical and
  observational results. An old number is not evidence of protocol completion.
- Keep the approved segmentation primary comparison at last1 for every model.
  For the separate dataset-best view, use even4 where requested and supported;
  otherwise use last with a declared architecture-specific fallback.
- Non-dense features use the final layer. Real CLS models use CLS plus patch mean;
  architectures without CLS must declare their native readout exception.
- E20 and E50 are independent cosine schedules, validated every epoch. Select
  the earliest best-validation head and evaluate test exactly once. Never select
  models, head budgets or epochs by the largest test advantage over FM.

Implementation: `external_fm_protocol.py`, `external_fm_features.py`,
`run_external_standard.py`, and `run_external_dense_rules.py`.
The benchmark_model dense CLI delegates to the unified dense entrypoint.

Remaining launch gates: certify all 14 real asset adapters (including native
multichannel normalization, no-CLS exceptions and numerical parity), complete
all-task evidence review and split hashes, and record cache/storage/peak-memory
preflight. No new formal FM experiment is authorized by a successful unit test.
