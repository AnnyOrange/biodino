# 5TB + GRAM ck30255 registered frozen-v3 missing-only continuation

User-authorized continuation of the ongoing 5TB + GRAM benchmark as a new
teacher appears during training. Scope is **only** the 38 READY frozen-v3
task/split components already registered for checkpoint `30255` by the
original `retest_20260918` online watcher. Do not restore/relabel old
results, change task settings, or run blocked datasets. Source evaluator
snapshot, task dataset hashes, image geometry, split, teacher identity,
checkpoint bytes/stat, dependency fingerprint, and feature/probe settings
must exactly match the original admission receipts.

The checkpoint is on the existing shared training mount; never copy it or
mutate active training. The checkpoint's registered GRAM config SHA is
`beddda580f4a3d2d0a81e2fa0f7653df616e92519c5ff0aeae57db4bbc9b394d`.
The training `config.yaml` is mutable, so use only the byte-identical
historical YAML reconstructed from its training startup log and verify this
SHA for each invocation. Run only missing numeric components in the new
`outputs/02_eval_runs/old_v3_protocol_union/5tb_gram_online_ck30255`
campaign; append SHA-validated results to the union archive.

CUDA sharing: low-resolution frozen tasks may be stacked toward five/GPU
only after measured memory headroom; 384/512 px and dense segmentation
must retain their fixed batch and be admitted conservatively. A dense
segmentation evaluator has exclusive GPU until its observed peak allows
additional work under Rules03; no optimistic five-per-GPU count. All claims,
run logs and numeric JSONs must persist; never mistake markers for results.
