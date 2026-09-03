#!/usr/bin/env python3
"""One-shot migrator for checkouts that have not yet pulled main.

The canonical fix lives in dinov3/train/train.py on origin/main (_iters casts).
New hosts: git pull --ff-only. Do not scp this file around or patch in place.

np.linspace's `num` and the last-layer slice bound both require exact ints, so
a fractional epoch count silently becomes a float and either raises or trips
the `len(schedule) == total_iters` assert.

Idempotent: python patch_fractional_epoch_schedules.py <repo>
"""

import sys
from pathlib import Path

HELPER = '''    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH

    def _iters(epochs):
        # Fractional epochs are allowed so schedule shape can be held fixed
        # across durations; np.linspace/slice bounds both require exact ints.
        return int(round(float(epochs) * OFFICIAL_EPOCH_LENGTH))

    lr = dict('''

EDITS = [
    (
        '    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH\n    lr = dict(',
        HELPER,
    ),
    (
        '        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,',
        '        warmup_iters=_iters(cfg.optim["warmup_epochs"]),',
    ),
    (
        '        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,\n'
        '        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,',
        '        total_iters=_iters(cfg.teacher["warmup_teacher_temp_epochs"]),\n'
        '        warmup_iters=_iters(cfg.teacher["warmup_teacher_temp_epochs"]),',
    ),
    (
        '    last_layer_lr_schedule.schedule[: cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH] = (',
        '    last_layer_lr_schedule.schedule[: _iters(cfg.optim["freeze_last_layer_epochs"])] = (',
    ),
]


def main() -> int:
    repo = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    target = repo / "dinov3" / "train" / "train.py"
    if not target.is_file():
        print(f"ERROR: not found: {target}", file=sys.stderr)
        return 2

    src = target.read_text()
    if '_iters(cfg.optim["warmup_epochs"])' in src:
        print(f"already patched: {target}")
        return 0

    for old, new in EDITS:
        if src.count(old) != 1:
            print(f"ERROR: expected 1 match, found {src.count(old)} for:\n{old}", file=sys.stderr)
            return 3
        src = src.replace(old, new)

    backup = target.with_suffix(".py.bak_fracsched")
    if not backup.exists():
        backup.write_text(target.read_text())
    target.write_text(src)
    print(f"patched: {target}  (backup {backup.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
