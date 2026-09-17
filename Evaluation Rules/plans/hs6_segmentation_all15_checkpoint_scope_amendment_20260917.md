# HS6 1TB all-checkpoint segmentation scope amendment

Status: APPROVED_SCOPE_EXTENSION_BEFORE_QUEUE_LAUNCH, 2026-09-17 UTC.

User asked whether the retained S+/B/L/H+ trajectories were rerun and requested immediate start. This extends the 2026-09-15 locked plan's segmentation coverage from selected display anchors to all 15 numeric checkpoint candidates of each HS6 1TB main run. Cscale/Dscale/scratch variants are not included in this trajectory queue. The entire 25/4/6/8/3 task inventory remains in scope, but this amendment does not claim a rerun of its nonsegmentation tasks.

## Unchanged scientific protocol

Use the same frozen last1 feature, dataset-specific geometry, current split loaders, probe head/optimizer/hyperparameters, independent E20/E50 cosine schedules, validation every epoch, earliest best validation mIoU, one test per fit, probe batch32 and seeds0/1/2. PanNuke uses all three rotations. BBBC038 remains a separate observation. Both budgets are reported, never selected to maximize a test advantage. Main target: 60 encoder checkpoints x 60 fits = 3,600 segmentation fits, including valid prior anchor fits.

All-candidate trajectories are exploratory sensitivity analyses. Formal HS0/HS6 matched selection still uses the previously frozen common ck8199; endpoint ck15374 is a predeclared companion. Evaluating more candidates does not authorize test-based re-selection for a confirmatory claim.

## Placement and recovery

- S+: first nine checkpoints on lyx; last six on shared storage/3090-qi.
- B: all 15 on hxw. The machine currently runs an eight-GPU training job, so the queue waits for exclusive GPU availability. Do not kill or change that training job.
- L: first twelve on lyx; ck13324/14349/15374 on shared storage/3090-qi.
- H+: all 15 on H100, approved GPUs0-3, sharing allowed as explicitly approved earlier.
- Existing HS0-L ck8199 and L5 ck26351 pilot anchors complete their remaining four datasets on shared storage/3090-qi.

Extraction batch is held fixed by architecture: S+/B/H+=8; L=2, matching the completed L pilots and fitting 24GB GPUs. Probe batch is always32. A strict numerical batch-invariance certificate is not yet present for the extraction-batch difference, so do not claim that gate has passed or that serialized caches are byte-identical.

Per queue lane: verify frozen code and CoNIC/Multimodal split hashes; require at least64GiB disk and sufficient GPU memory; hash each checkpoint on its resident machine; skip only audit-valid existing fits; keep head/result files; delete only dataset feature caches after successful fit validation. Record queue state and per-checkpoint logs. Pause a lane after two consecutive checkpoint failures. No weight copying or deleting.

The placement CSV and fourteen queue configs live under `benchmark_model/fair_plot_20260915/`. Launch-time PID/status are recorded separately; queued or hashing is not reported as completed.
