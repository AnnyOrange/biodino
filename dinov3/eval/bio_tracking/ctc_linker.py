"""Deterministic instance linking for Cell Tracking Challenge predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class TrackRecord:
    track_id: int
    birth: int
    end: int
    parent: int = 0


def _positive_ids(mask: np.ndarray) -> np.ndarray:
    ids = np.unique(mask)
    return ids[ids > 0].astype(np.int64, copy=False)


def equivalent_diameters(mask: np.ndarray) -> np.ndarray:
    """Return area/volume-equivalent diameters for every positive instance."""
    array = np.asarray(mask)
    ids = _positive_ids(array)
    if not len(ids):
        return np.empty(0, dtype=np.float64)
    _, counts = np.unique(array[array > 0], return_counts=True)
    counts = counts.astype(np.float64)
    if array.ndim == 2:
        return 2.0 * np.sqrt(counts / np.pi)
    if array.ndim == 3:
        return 2.0 * np.cbrt(3.0 * counts / (4.0 * np.pi))
    raise ValueError(f"expected a 2-D or 3-D instance mask, got shape {array.shape}")


def _instance_geometry(mask: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not len(ids):
        return np.empty(0, dtype=np.float64), np.empty((0, mask.ndim), dtype=np.float64)
    array = np.asarray(mask)
    coordinates = np.nonzero(array)
    labels = array[coordinates]
    index = np.searchsorted(ids, labels)
    counts = np.bincount(index, minlength=len(ids)).astype(np.float64)
    centroids = np.empty((len(ids), mask.ndim), dtype=np.float64)
    for axis in range(mask.ndim):
        sums = np.bincount(index, weights=coordinates[axis], minlength=len(ids))
        centroids[:, axis] = sums / np.maximum(counts, 1.0)
    return counts, centroids


def _overlap_matrix(
    previous: np.ndarray,
    current: np.ndarray,
    previous_ids: np.ndarray,
    current_ids: np.ndarray,
) -> np.ndarray:
    overlap = np.zeros((len(previous_ids), len(current_ids)), dtype=np.float64)
    if not len(previous_ids) or not len(current_ids):
        return overlap
    flat_previous = previous.reshape(-1)
    flat_current = current.reshape(-1)
    valid = (flat_previous > 0) & (flat_current > 0)
    if not valid.any():
        return overlap
    pindex = np.searchsorted(previous_ids, flat_previous[valid])
    cindex = np.searchsorted(current_ids, flat_current[valid])
    pairs = pindex * len(current_ids) + cindex
    overlap.flat[: len(previous_ids) * len(current_ids)] = np.bincount(
        pairs, minlength=len(previous_ids) * len(current_ids)
    )
    return overlap


def _render_tracks(mask: np.ndarray, track_by_local: dict[int, int]) -> np.ndarray:
    if not track_by_local:
        return np.zeros(mask.shape, dtype=np.uint32)
    lookup = np.zeros(int(np.max(mask)) + 1, dtype=np.uint32)
    for local_id, track_id in track_by_local.items():
        lookup[local_id] = track_id
    return lookup[np.asarray(mask, dtype=np.int64)]


class TrackLinker:
    """Link frame-local instances into deterministic CTC track IDs.

    Division detection precedes one-to-one assignment. A previous instance is a
    parent when its two strongest unclaimed children each cover at least
    ``division_overlap`` of the parent. Remaining objects use the fixed Hungarian
    cost from the v3 protocol.
    """

    def __init__(
        self,
        diameter: float,
        *,
        division_overlap: float = 0.1,
        iou_weight: float = 0.7,
        distance_weight: float = 0.3,
    ) -> None:
        if not np.isfinite(diameter) or diameter <= 0:
            raise ValueError("diameter must be finite and positive")
        if not 0 <= division_overlap <= 1:
            raise ValueError("division_overlap must be in [0, 1]")
        self.diameter = float(diameter)
        self.division_overlap = float(division_overlap)
        self.iou_weight = float(iou_weight)
        self.distance_weight = float(distance_weight)
        self.previous_local: np.ndarray | None = None
        self.previous_track_by_local: dict[int, int] = {}
        self.records: dict[int, TrackRecord] = {}
        self.next_track_id = 1
        self.next_frame = 0

    def _new_track(self, frame: int, parent: int = 0) -> int:
        track_id = self.next_track_id
        self.next_track_id += 1
        self.records[track_id] = TrackRecord(track_id, frame, frame, parent)
        return track_id

    def step(self, local_mask: np.ndarray) -> np.ndarray:
        current = np.asarray(local_mask)
        if current.ndim not in (2, 3):
            raise ValueError(f"expected 2-D or 3-D mask, got {current.shape}")
        if np.any(current < 0):
            raise ValueError("instance labels must be non-negative")
        frame = self.next_frame
        self.next_frame += 1
        current_ids = _positive_ids(current)

        if self.previous_local is None:
            current_track_by_local = {}
            for current_id in current_ids:
                track_id = self._new_track(frame)
                current_track_by_local[int(current_id)] = track_id
            output = _render_tracks(current, current_track_by_local)
            self.previous_local = current.copy()
            self.previous_track_by_local = current_track_by_local
            return output

        if current.shape != self.previous_local.shape:
            raise ValueError(
                f"all frames in a sequence must share a shape: {self.previous_local.shape} != {current.shape}"
            )

        previous_ids = _positive_ids(self.previous_local)
        previous_tracks = np.asarray(
            [self.previous_track_by_local[int(value)] for value in previous_ids],
            dtype=np.int64,
        )
        previous_areas, previous_centroids = _instance_geometry(self.previous_local, previous_ids)
        current_areas, current_centroids = _instance_geometry(current, current_ids)
        overlap = _overlap_matrix(self.previous_local, current, previous_ids, current_ids)

        used_previous: set[int] = set()
        used_current: set[int] = set()
        current_track_by_local: dict[int, int] = {}

        # Divisions preempt continuation. Parent order and child tie-breaking are
        # label-stable so reruns are byte-identical.
        parent_order = sorted(range(len(previous_ids)), key=lambda index: int(previous_tracks[index]))
        for pindex in parent_order:
            fractions = overlap[pindex] / max(previous_areas[pindex], 1.0)
            candidates = [
                cindex for cindex in range(len(current_ids))
                if cindex not in used_current and fractions[cindex] >= self.division_overlap
            ]
            candidates.sort(key=lambda cindex: (-fractions[cindex], int(current_ids[cindex])))
            if len(candidates) < 2:
                continue
            parent_track = int(previous_tracks[pindex])
            for cindex in candidates[:2]:
                child_track = self._new_track(frame, parent=parent_track)
                current_track_by_local[int(current_ids[cindex])] = child_track
                used_current.add(cindex)
            used_previous.add(pindex)

        remaining_previous = [i for i in range(len(previous_ids)) if i not in used_previous]
        remaining_current = [i for i in range(len(current_ids)) if i not in used_current]
        if remaining_previous and remaining_current:
            psel = np.asarray(remaining_previous, dtype=np.int64)
            csel = np.asarray(remaining_current, dtype=np.int64)
            selected_overlap = overlap[np.ix_(psel, csel)]
            union = (
                previous_areas[psel, None]
                + current_areas[csel][None, :]
                - selected_overlap
            )
            iou = selected_overlap / np.maximum(union, 1.0)
            delta = previous_centroids[psel, None, :] - current_centroids[csel][None, :, :]
            normalized_distance = np.linalg.norm(delta, axis=2) / self.diameter
            allowed = (iou > 0) | (normalized_distance <= 1.0)
            cost = self.iou_weight * (1.0 - iou) + self.distance_weight * normalized_distance
            cost = np.where(allowed, cost, 1.0e9)
            row_indices, column_indices = linear_sum_assignment(cost)
            for row, column in zip(row_indices, column_indices):
                if not allowed[row, column]:
                    continue
                pindex = int(psel[row])
                cindex = int(csel[column])
                track_id = int(previous_tracks[pindex])
                current_track_by_local[int(current_ids[cindex])] = track_id
                self.records[track_id].end = frame
                used_previous.add(pindex)
                used_current.add(cindex)

        for cindex, current_id in enumerate(current_ids):
            if cindex not in used_current:
                track_id = self._new_track(frame)
                current_track_by_local[int(current_id)] = track_id

        output = _render_tracks(current, current_track_by_local)
        self.previous_local = current.copy()
        self.previous_track_by_local = current_track_by_local
        return output

    def track_table(self) -> np.ndarray:
        rows = [
            (record.track_id, record.birth, record.end, record.parent)
            for record in sorted(self.records.values(), key=lambda item: item.track_id)
        ]
        return np.asarray(rows, dtype=np.int64).reshape(-1, 4)


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size + 1))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def consolidate_slices_3d(slices: list[np.ndarray]) -> np.ndarray:
    """Merge independently watersheded 2-D instances with 26-connectivity in z."""
    if not slices:
        raise ValueError("at least one slice is required")
    shape = np.asarray(slices[0]).shape
    if len(shape) != 2 or any(np.asarray(item).shape != shape for item in slices):
        raise ValueError("all inputs must be equally shaped 2-D instance masks")

    volume = np.zeros((len(slices), *shape), dtype=np.int64)
    next_label = 1
    for z, item in enumerate(slices):
        array = np.asarray(item)
        for label in _positive_ids(array):
            volume[z][array == label] = next_label
            next_label += 1
    sets = _DisjointSet(next_label)
    height, width = shape
    for z in range(1, len(slices)):
        previous, current = volume[z - 1], volume[z]
        for dy in (-1, 0, 1):
            py0, py1 = max(0, -dy), min(height, height - dy)
            cy0, cy1 = max(0, dy), min(height, height + dy)
            for dx in (-1, 0, 1):
                px0, px1 = max(0, -dx), min(width, width - dx)
                cx0, cx1 = max(0, dx), min(width, width + dx)
                left = previous[py0:py1, px0:px1]
                right = current[cy0:cy1, cx0:cx1]
                valid = (left > 0) & (right > 0)
                if not valid.any():
                    continue
                for first, second in np.unique(np.stack((left[valid], right[valid]), axis=1), axis=0):
                    sets.union(int(first), int(second))

    roots = sorted({sets.find(int(value)) for value in np.unique(volume) if value > 0})
    root_to_output = {root: index + 1 for index, root in enumerate(roots)}
    output = np.zeros_like(volume, dtype=np.int32)
    for value in np.unique(volume):
        if value > 0:
            output[volume == value] = root_to_output[sets.find(int(value))]
    return output
