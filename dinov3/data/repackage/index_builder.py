"""Lightweight tar-path discovery (no member scanning).

Uses ``manifest.json`` files when present for instant shard discovery.
Falls back to directory globbing.  **No tar member enumeration** is done
here — that happens lazily during the streaming pipeline phase, avoiding
a multi-minute first-pass over NFS-hosted multi-GB shards.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("repackage.index")

_KEY_RE = re.compile(r"^(?P<base>.+?)_ch\d+$")


@dataclass
class TarInfo:
    """Descriptor for one tar shard on disk."""

    tar_path: Path
    channel_count: int
    approx_samples: int = 0


def discover_shards(
    input_root: Path,
    channel_dirs: Optional[List[str]] = None,
) -> List[TarInfo]:
    """Find all tar shards under ``input_root/ch*/``.

    Tries ``manifest.json`` first (O(1) per directory); falls back to
    ``glob("*.tar")``.

    Returns:
        Flat list of :class:`TarInfo`, one per tar file.
    """
    dirs = _resolve_channel_dirs(input_root, channel_dirs)
    logger.info("Discovering shards in %d channel directories", len(dirs))

    result: List[TarInfo] = []
    for ch_dir in dirs:
        ch_num = _parse_ch_number(ch_dir.name)
        manifest = ch_dir / "manifest.json"

        if manifest.exists():
            result.extend(_from_manifest(manifest, ch_num))
        else:
            result.extend(_from_glob(ch_dir, ch_num))

    logger.info(
        "Discovery complete: %d shards, estimated %d samples",
        len(result),
        sum(t.approx_samples for t in result),
    )
    return result


def extract_sample_id(member_name: str) -> str:
    """Strip ``_ch<N>`` suffix + file extension → channel-agnostic key.

    ``id000008707_oid71125183_ch1.json`` → ``id000008707_oid71125183``
    """
    base = member_name.rsplit(".", 1)[0]
    m = _KEY_RE.match(base)
    return m.group("base") if m else base


# ---------------------------------------------------------------------- #
# Internals                                                                #
# ---------------------------------------------------------------------- #

def _resolve_channel_dirs(root: Path, explicit: Optional[List[str]]) -> List[Path]:
    if explicit:
        return sorted(root / n for n in explicit if (root / n).is_dir())
    return sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("ch"))


def _parse_ch_number(name: str) -> int:
    digits = "".join(c for c in name if c.isdigit())
    return int(digits) if digits else 0


def _from_manifest(manifest: Path, ch_num: int) -> List[TarInfo]:
    try:
        data = json.loads(manifest.read_text())
    except Exception as exc:
        logger.warning("Bad manifest %s: %s", manifest, exc)
        return _from_glob(manifest.parent, ch_num)

    shards = data.get("shards", [])
    infos: List[TarInfo] = []
    for shard in shards:
        tar_path = Path(shard["path"])
        if not tar_path.exists():
            logger.debug("Shard listed in manifest missing: %s", tar_path)
            continue
        infos.append(
            TarInfo(
                tar_path=tar_path,
                channel_count=ch_num,
                approx_samples=shard.get("num_written", 0),
            )
        )
    logger.info("  %s: %d shards (from manifest)", manifest.parent.name, len(infos))
    return infos


def _from_glob(ch_dir: Path, ch_num: int) -> List[TarInfo]:
    tar_files = sorted(ch_dir.glob("*.tar"))
    logger.info("  %s: %d shards (from glob)", ch_dir.name, len(tar_files))
    return [TarInfo(tar_path=p, channel_count=ch_num) for p in tar_files]
