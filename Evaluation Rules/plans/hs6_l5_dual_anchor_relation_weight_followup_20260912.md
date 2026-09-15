# HS6-L5 dual-anchor relation-weight follow-up

Status: `LOCKED_BEFORE_FOLLOWUP_ENDPOINTS`.

## Motivation

The matched `patch_weight=2.0, relation_weight=0.5` screen failed the
pre-registered label-free relation gate: its ck20007-anchor relation drift was
1.8471 times the matched control and 1.1797 times the patch-only arm.  This
follow-up was selected from that label-free failure, not from any frozen probe
or downstream label.

At the observed training losses, the weighted relation term was materially
smaller than the weighted patch term.  Three matched 488-update screens
separate insufficient weighting from a misaligned relation observation:

1. `strong_dual`: patch Gram weight 2.0 and global relation weight 2.0.
2. `relation_only`: patch Gram weight 0.0 and global relation weight 2.0.
3. `mask_matched_dual`: patch Gram weight 2.0 and global relation weight 2.0;
   the frozen global anchor receives the exact augmented 256-pixel crops and
   iBOT masks seen by the student.  The patch anchor retains the official
   clean 512-pixel observation.

Both use student/EMA optimizer state ck20007, frozen patch anchor ck7807,
frozen global relation anchor ck20007, four ranks by 16 samples, identical
deterministic data order, and updates 20008--20495.  The relation-only arm
still loads the patch teacher for implementation parity, but its patch loss
has exactly zero optimization weight.  The mask-matched arm was specified
after a code audit showed that the original global anchor saw a clean,
unmasked 512-pixel observation while its student operand was augmented,
masked, and 256 pixels; it tests observation alignment without changing
either anchor or loss weight.

## Label-free decisions

- `strong_dual` is eligible for a longer self-supervised run only if its
  relation drift is no more than 1.1 times control and no more than 0.8 times
  patch-only, while its paired block-24 spatial margin is no lower than
  control and retains at least half of the patch-only gain.
- `relation_only` is a mechanism control.  It supports the relation objective
  only if its relation drift is lower than the matched no-Gram control.
- `mask_matched_dual` uses the same eligibility gate as `strong_dual` and must
  also record `gram_global_relation_mask_matched=1` on every update.
- All sample keys/crop coordinates must match their frozen controls exactly.
- No labeled task result may select weights, anchors, checkpoints, or gates.
- The already failed weight-0.5 endpoint remains immutable and is not replaced.
