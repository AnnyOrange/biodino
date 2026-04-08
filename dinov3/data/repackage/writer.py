"""WebDataset shard writer for the packed multi-channel format.

Each output sample is a dict::

    {
        "__key__": "<unique_key>",
        "meta.json": <dict>,          # auto-serialised by wds
        "ch1.tif":   <bytes>,
        "ch2.tif":   <bytes>,
        ...
    }
"""

import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger("repackage.writer")


class PackedShardWriter:
    """Thin wrapper around :class:`webdataset.ShardWriter`."""

    def __init__(
        self,
        output_dir: Path,
        prefix: str = "filtered_mixed_train",
        max_count: int = 10_000,
        max_size: int = 3 * 1024 * 1024 * 1024,
    ) -> None:
        import webdataset as wds

        output_dir.mkdir(parents=True, exist_ok=True)
        pattern = str(output_dir / f"{prefix}-%06d.tar")
        self._sink = wds.ShardWriter(
            pattern,
            maxcount=max_count,
            maxsize=max_size,
        )
        self._written = 0
        logger.info("ShardWriter opened: %s", pattern)

    @property
    def written(self) -> int:
        return self._written

    def write(self, sample: Dict) -> None:
        """Write one packed sample dict (must contain ``__key__``)."""
        self._sink.write(sample)
        self._written += 1

    def close(self) -> None:
        self._sink.close()
        logger.info("ShardWriter closed — %d samples written", self._written)

    def __enter__(self) -> "PackedShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
