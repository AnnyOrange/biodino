# HS6-L5 Gram legacy OOD batch-64 dispatch v2

Status: `LOCKED_BEFORE_ANY_VALID_OOD_RESULT`.

The first batch-64 amendment correctly fixed the OOD batch but its one-shot
launcher still relied on normal fleet lane ordering plus zero stale timeout.
During dispatch it reclaimed three active classification claims and started
duplicate processes. They were stopped within 30 seconds, before any file in
the affected dataset output directories was created or modified. The original
claim owners were restored from their live worker records. No result from the
misdispatch is admissible.

Dispatch v2 adds two mechanical safeguards:

- `--include-lanes ood` makes the worker unable to select another lane.
- `--claim-reservation-marker PROTOCOL_HOLD_BATCH64` transfers only the exact
  pre-created OOD reservation; normal live claims are never treated as stale.

All scientific settings remain those of the batch-64 amendment. This is an
execution correction made before any valid OOD metric was produced.
