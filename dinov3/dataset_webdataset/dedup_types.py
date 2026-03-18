"""Whitelist data types for dedup index filtering."""

from dataclasses import dataclass, field
from typing import Dict, Set


@dataclass
class Whitelist:
    """Two-level whitelist for whole-file and frame-level matching."""

    path_set: Set[str] = field(default_factory=set)
    frame_map: Dict[str, Set[int]] = field(default_factory=dict)

    def add_path(self, path: str) -> None:
        self.path_set.add(path)

    def add_frame(self, path: str, frame_idx: int) -> None:
        if path not in self.frame_map:
            self.frame_map[path] = set()
        self.frame_map[path].add(frame_idx)

    def __len__(self) -> int:
        return len(self.path_set) + sum(len(v) for v in self.frame_map.values())

    @property
    def path_only_count(self) -> int:
        return len(self.path_set)

    @property
    def frame_entry_count(self) -> int:
        return sum(len(v) for v in self.frame_map.values())

    @property
    def frame_file_count(self) -> int:
        return len(self.frame_map)


