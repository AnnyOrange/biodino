# NCI, NRI, and residual-MC retirement

This cleanup retires the NCI, NRI, and residual multichannel (RC) training
implementations while preserving their scientific record.

## Preservation rules

- Keep all evaluation reports, metrics, plots, logs, manifests, and summaries.
- Delete only evaluation directories named `cache`, `features`, `embeddings`,
  `embedding_cache`, or `feature_cache` below an NCI/NRI/residual-MC output.
- Keep every checkpoint in
  `retired_nci_nri_rc_best_checkpoints_20260915.txt`.
- Delete only non-best `checkpoint.pth` files in method-specific training roots.
- No `teacher_checkpoint.pth` existed in the candidate NCI/NRI/RC roots at the
  time of cleanup.
- Do not touch checkpoints or caches belonging to other methods, including HS6
  5TB runs.

## Best-checkpoint provenance

The primary source is `outputs/00_reports/best_checkpoint_index.csv`, using all
rows marked `recommended=1`. Six best rows had no absolute `checkpoint_path`,
but resolved unambiguously from their evaluation run and checkpoint number; they
were retained as well:

- M4 residual-MC Tissuenet segmentation: ck6149.
- Long residual-MC detection/segmentation: ck3074.
- NCI mechanism extensions (three arms): ck1023.
- NCI no-stop-gradient segmentation arm: ck127.

Both formal NCI/control ck15374 copies under `outputs/02_eval_inputs` are also
retained. The pre-retirement source implementation is available at Git commit
`e94e62a`; it is required to load residual-MC checkpoints after the live
implementation is removed.

## Pre-delete inventory

- Candidate checkpoints: 91.
- Retained best checkpoints: 34.
- Non-best checkpoints selected for deletion: 57.
- Evaluation cache directories deleted: 973 exact-name directories containing
  2,438 files and 2,422,870,977,682 bytes.
- Preserved evaluation-result inventory before and after deletion: 3,597 files
  and 94,563,989 bytes.

Deletion is fail-closed: the candidate count, whitelist count, file existence,
step-directory contents, and absence of teacher checkpoints must all match this
record before any deletion starts.

The machine-readable record, including every retained checkpoint, deleted
checkpoint, and deleted cache directory, is in
`retired_nci_nri_rc_cleanup_report_20260915.json`.
