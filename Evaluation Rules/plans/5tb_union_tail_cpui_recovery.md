# Shared 5TB missing-only cpui recovery

Status: approved by the user's 2026-09-22 instruction to continue remaining
retests on idle single-RTX-3090 machines. Frozen READY components only; this
is not a claim that blocked RxRx3/native CTC/OOD or legacy-only v4 extensions
are complete.

Use the registered 2026-09-18 component manifest and its exact immutable
evaluator source snapshot SHA
`8f0e2de83568b3f673e7c9de07efc6a17768e400bd9a3a740808bbff70a21a04`.
The input inventory and dataset/split SHA, teacher branch, model checkpoint,
feature/probe settings and numerical dependency fingerprint are inherited
without modification. The latest Evaluation Rules and protocol_v4 are separately
copied and hashed as policy evidence; Rule09 does not retroactively change
these already registered frozen-v3 components. Keep the complete expected
inventory, with `full_v3_aggregate_allowed=false`.

The output is `outputs/02_eval_runs/old_v3_protocol_union/5tb_cpui_recovery`;
retain the original training roots and all historical/old campaign results.
Admit only validated-result-absent original components:

- 5TB no-GRAM: 6 cells, five frozen classification cells with stale claims
  and one dense Multimodal_CellSeg with an orphaned marker, only after proving
  no evaluator still runs on the old cpu18/cpu20 hosts.
- 5TB + GRAM: 124 cells in the current 35-checkpoint registered grid: ten
  LIVECell/Multimodal dense components from checkpoints 13175–15127, plus
  38 each from 28791, 29279, 29767. New checkpoints beyond this registered
  grid require their own hash/teacher/data admission and are not silently
  added to the fixed recovery campaign.
- FM14: three remaining errors are excluded here. CytoImageNet BBBC048 is
  verified CUDA OOM at the required batch64 on 24 GiB; CONCH and PE CoNIC
  exited by signal 11. These require a separately diagnosed same-protocol
  retry on a fitting machine, not an unverified 3090 resource fallback.

GRAM's live `config.yaml` is mutable training state: NEVER change it or
overwrite an earlier record. Extract startup-resolved config bytes from the
original training `logs/log.txt` into three separate SHA-named immutable
copies. Check the byte SHA against each registered checkpoint input receipt:
`ac0363fed5310d9f23a52526d07cb5a5c395f9c4a127aa9701843ae6a5d6e648`
(26 checkpoints), `2edaa329e0492b3d22133c7865de5324aee86b4fa17ecb0c20711c927fbe372d`
(7), `beddda580f4a3d2d0a81e2fa0f7653df616e92519c5ff0aeae57db4bbc9b394d`
(2). The first two differ only in YAML key order; the last changes gradient
accumulation 8 to 16. Check teacher key, checkpoint bytes/stat and the
dataset split hash before each new run. The protocol evaluation settings do
not change with the restored training configuration.

Placement: cpu1, cpu2, cpu9, cpu10, cpu12, cpu15, cpu20, GPU0 on each;
all read local mounts of the same shared checkpoint and benchmark. Check
memory and incumbent PIDs on each machine before launch, without stopping
anyone's work. Frozen cells: batch64, BF16, final CLS + mean patch, seed0,
workers2 and BLAS1. Dense cells: source-registered split/geometry/layers,
feature/probe batch32, E20 and E50 independently, validation each epoch,
seeds0/1/2, best val then once-only test. On 24 GiB, high-resolution dense
starts at one/card; low-memory frozen jobs can be stacked to 3–5/card only
after measured peaks and disk/NFS preflight. Do not change protocol batch,
image size or feature layer to fill a GPU. Failed tasks retain full logs and
are never silently reported as complete. Only SHA-verified numeric artifacts
and validation reports enter the union archive.

## Concurrent execution record (2026-09-22)

After starting one primary worker on each admitted host, additional
`--task-family frozen` lanes were started on cpu1, cpu2 and cpu10 to target
five concurrent tests per 3090. These share the *same* atomic claim/done
registry and pinned evaluator; no component is scheduled twice and no
hyperparameter is changed. cpu2 has its separate previously admitted second
lane as well. The stack launcher checks for a resident dense segmentation
task before admission. At the first verified snapshot cpu10 ran five, cpu1
four and cpu2 four tests; cpu9 ran one dense segmentation, while cpu12,
cpu15 and cpu20 ran one 512px classification each. These counts are dynamic,
not promised permanent occupancy: as the high-resolution jobs terminate,
add lightweight lanes only while the measured GPU peak leaves safe headroom.
Do not interpret five *processes* as five independently completed results.
