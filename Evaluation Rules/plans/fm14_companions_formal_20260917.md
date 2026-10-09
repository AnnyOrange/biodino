# FM14 formal companions: RxRx3, OOD and native CTC

Authorization: user explicitly requested implementation and execution on 2026-09-17.
Existing FM queues keep their fixed commit and are not hot-updated.

Models: the same 14 published checkpoint directories listed in
`benchmark_model/benchmark_runs/fm14_rules_aligned_20260917/campaign_manifest.json`.
Checkpoints remain on Huawei shared storage; no checkpoints, datasets or feature
banks are transferred. Companion outputs are independent of the active queue.

RxRx3 uses the locked all-734-gene manifest (734 query / 734 gallery), hash
`94d570cb66d71de20e9ded203d8727623fe85a807d8b5b9cdbe0de3fce1318f5`.
All models receive the same six-channel p01/p99 pair-mean compact3 cache. Final
CLS/patch readouts and declared no-CLS exceptions are the audited FM14 adapter.
Resize256/crop224, batch64, BF16, seed0, workers2, BLAS1. Save R@1/5/10,
mAP, MRR, NMI, ARI and Hungarian cluster accuracy. No quickscreen reuse.

OOD uses xray and cryo separately. Freeze the canonical common ID3000 reference
(bloodmnist, BBBC048, CYCLOPS; deterministic 1000/source), ID bank/test seed0
70/30, cosine kNN k10. Use all eligible xray volumes, 8 evenly sampled slices
per volume, three_slices; cryo uses the existing deterministic canonical
20000-particle-per-CS-project selection. These are declared fixed benchmark
selections, not hardware-dependent caps. The entire selected record list,
ID indices, source metadata and sample counts are locked before execution.
Percentiles0.5/99.5, cryo invertFalse, final readout, resize256/crop224,
BF16/batch64/workers2/seed0. Only OOD AUROC/AP are computed here; no random
same-volume classification is substituted for OOD. Old OOD is not silently reused.

CTC uses all20 native2D/3D domains, five16-train/4-heldout-domain folds, locked
manifest `7a0f5bc2579f6ae22a2b8f2b16103530cd678466a4b2bbd1f617692b8e3cac27`.
Reuse fixed HoVerNet decoder feature_size32/embed_proj384, epoch50 head,
AdamW lr1e-3/wd1e-4, batch8/BF16/seed0/workers2. Do not use held-out validation
or threshold tuning. Preserve native image geometry, 256tiles/64overlap,
global thresholds0.5/0.4/minimum10, fixed linker trained-domain diameter only.
Published final FM maps are bilinearly aligned to the same stride16 decoder
grid; original legal backbone patch geometry is recorded, no pixel target
resize. Input MICRO normalization is undone before published FM normalization.
No-CLS CNNs use their true final spatial maps. Score pinned official
py-ctcmetrics `59481c48a62d4376fe34bed3e3606b4ec4d60972` after 2D/3D oracle smoke.
Save per-domain TRA/SEG/DET/AP and only aggregate after all five folds/20 domains.
Existing scorer extra Dice is foreground Dice, despite its legacy field name
instance_mDice; it must not be described as matched-instance Dice.

Resource schedule: shared fleet 3090-qi and repaired CPUI environments, count
all project evaluator roots together, maximum5/GPU; >=60% admission follows
Rules. CTC/high-resolution dense initially require >=20000MiB free, one task,
and companion queue admits no additional task to its own reserved GPU.
No lowering protocol batch/resolution on OOM. No training interruptions.
Use free slots or wait for existing queues to release capacity.

Output: `benchmark_model/benchmark_runs/fm14_companions_formal_20260917`.
One attempt per cell; immutable claims and explicit resource/protocol failures.
Every result needs checkpoint/config hash, split/input hash, fixed code commit,
actual environment and independent validation. This is an independent-component
campaign, not proof that HS0/HS6 full-v3 comparisons are already complete.
