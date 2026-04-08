"""Shared utilities: statistics tracker, logging bootstrap, helpers."""

import logging
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class PipelineStats:
    """Mutable counters accumulated during a pipeline run."""

    total_samples: int = 0
    total_patches: int = 0
    patches_kept: int = 0
    patches_filtered: int = 0
    samples_with_missing_channels: int = 0
    read_errors: int = 0
    tiling_errors: int = 0
    channel_combo_counts: Counter = field(default_factory=Counter)
    _start: float = field(default_factory=time.monotonic, repr=False)

    def record_channel_combo(self, n_channels: int) -> None:
        self.channel_combo_counts[n_channels] += 1

    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def summary(self) -> str:
        elapsed = self.elapsed()
        lines = [
            "=" * 60,
            "Pipeline statistics",
            "=" * 60,
            f"  Total input samples : {self.total_samples:>10,d}",
            f"  Total patches       : {self.total_patches:>10,d}",
            f"  Patches kept        : {self.patches_kept:>10,d}",
            f"  Patches filtered    : {self.patches_filtered:>10,d}",
            f"  Missing-ch samples  : {self.samples_with_missing_channels:>10,d}",
            f"  Read errors         : {self.read_errors:>10,d}",
            f"  Tiling errors       : {self.tiling_errors:>10,d}",
            f"  Elapsed             : {elapsed:>10.1f} s",
        ]
        if self.total_patches > 0:
            keep_pct = 100.0 * self.patches_kept / self.total_patches
            lines.append(f"  Keep rate           : {keep_pct:>9.1f} %")
        lines.append("  Channel-count distribution:")
        for k in sorted(self.channel_combo_counts):
            lines.append(f"    {k:>2d}-ch : {self.channel_combo_counts[k]:>10,d}")
        lines.append("=" * 60)
        return "\n".join(lines)


def setup_logging(level: str = "INFO") -> None:
    """Configure the ``repackage`` logger hierarchy."""
    root = logging.getLogger("repackage")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(name)s %(levelname)s  %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root.addHandler(handler)


def format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} PB"


def merge_stats(all_stats: list) -> "PipelineStats":
    """Sum counters from multiple worker PipelineStats into one."""
    merged = PipelineStats()
    for s in all_stats:
        merged.total_samples += s.total_samples
        merged.total_patches += s.total_patches
        merged.patches_kept += s.patches_kept
        merged.patches_filtered += s.patches_filtered
        merged.samples_with_missing_channels += s.samples_with_missing_channels
        merged.read_errors += s.read_errors
        merged.tiling_errors += s.tiling_errors
        merged.channel_combo_counts.update(s.channel_combo_counts)
    return merged


def parse_shape(raw) -> Tuple[int, ...]:
    """Safely convert a JSON shape field (list or None) to a tuple."""
    if raw is None:
        return ()
    try:
        return tuple(int(x) for x in raw)
    except (TypeError, ValueError):
        return ()
