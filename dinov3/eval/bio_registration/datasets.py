"""Native paired-image/landmark adapters with specimen-level partitions."""
import csv
from dataclasses import dataclass
import hashlib
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class RegistrationPair:
    pair_id: str
    group: str
    source_image: Path
    target_image: Path
    source_landmarks: Path | None
    target_landmarks: Path | None
    official_status: str
    coordinate_unit: str = "pixels"
    reference_diagonal: float | None = None


def load_landmarks(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"X", "Y"}.issubset(rows[0]):
        raise ValueError(f"Expected BIRL CSV X,Y columns: {path}")
    ids = [row.get("", str(i)) for i, row in enumerate(rows)]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate landmark IDs: {path}")
    points = np.asarray([[float(row["X"]), float(row["Y"])] for row in rows])
    if not np.isfinite(points).all():
        raise ValueError(f"Nonfinite landmarks: {path}")
    return ids, points


def aligned_landmarks(pair):
    if pair.source_landmarks is None or pair.target_landmarks is None:
        raise ValueError("Ground-truth landmarks unavailable (hidden challenge targets)")
    source_ids, source = load_landmarks(pair.source_landmarks)
    target_ids, target = load_landmarks(pair.target_landmarks)
    # Eight public ANHIR pairs have different annotation counts. BIRL scores
    # common annotations. Match explicit IDs instead of silently truncating
    # arbitrary arrays; missing predictions for these IDs remain an error.
    common = set(source_ids) & set(target_ids)
    if not common:
        raise ValueError(f"No common landmark IDs: {pair.pair_id}")
    lookup = {key: point for key, point in zip(target_ids, target)}
    keep = [i for i, key in enumerate(source_ids) if key in common]
    return source[keep], np.asarray([lookup[source_ids[i]] for i in keep])


def anhir_pairs(root, public_only=True):
    root = Path(root)
    if (root / "extracted/dataset_medium").exists():
        root = root / "extracted/dataset_medium"
    pairs = []
    with (root / "dataset_medium.csv").open(newline="") as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            status = row["status"]
            if public_only and status != "training":
                continue
            if Path(row["Source image"]).parts[0] != Path(row["Target image"]).parts[0]:
                raise ValueError("ANHIR pair crosses tissue groups; cannot apply specimen holdout")
            pairs.append(RegistrationPair(f"anhir_{i:04d}", Path(row["Source image"]).parts[0],
                root / row["Source image"], root / row["Target image"],
                root / row["Source landmarks"],
                root / row["Target landmarks"] if status == "training" else None, status,
                reference_diagonal=float(row["Image diagonal [pixels]"])))
    return pairs


def cima_pairs(root, scale="scale-5pc"):
    """Fixed unordered pair cover, lexicographic source->target at ONE scale.

    Both stains' CSV IDs must align. This is a declared local BIRL convention,
    not a claim of reproducing an unverified official 108-pair cover.
    """
    root = Path(root)
    if (root / "extracted").exists():
        root = root / "extracted"
    pairs = []
    for folder in sorted(root.glob("*/" + scale)):
        images = sorted(path for path in folder.iterdir()
                        if path.suffix.lower() in {".jpg", ".png", ".tif", ".tiff"})
        for source, target in combinations(images, 2):
            pairs.append(RegistrationPair(f"cima_{len(pairs):04d}", folder.parent.name,
                source, target, source.with_suffix(".csv"), target.with_suffix(".csv"), "public"))
    return pairs


def grouped_partition(pairs, seed=0, development_fraction=.25, exclude_development_groups=()):
    """Reproducible group holdout; all stains/scales/adjacent pairs stay together."""
    groups = sorted({pair.group for pair in pairs})
    if len(groups) < 2 or not 0 < development_fraction < 1:
        raise ValueError("Group-heldout selection needs at least two groups")
    candidates = [group for group in groups if group not in set(exclude_development_groups)]
    if not candidates:
        raise ValueError("No eligible development groups")
    ordered = sorted(candidates, key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest())
    count = min(len(groups) - 1, len(ordered), max(1, round(len(groups) * development_fraction)))
    development = set(ordered[:count])
    return {pair.pair_id: "development" if pair.group in development else "evaluation" for pair in pairs}


def preflight_pairs(pairs):
    records, failures = [], []
    ids = set()
    for pair in pairs:
        try:
            if pair.pair_id in ids:
                raise ValueError("Duplicate pair ID")
            ids.add(pair.pair_id)
            for image in (pair.source_image, pair.target_image):
                with Image.open(image) as opened:
                    if min(opened.size) <= 0:
                        raise ValueError("Empty image")
            if pair.target_landmarks is not None:
                source_ids, _ = load_landmarks(pair.source_landmarks)
                target_ids, _ = load_landmarks(pair.target_landmarks)
                source, target = aligned_landmarks(pair)
                if not len(source):
                    raise ValueError("Empty landmarks")
                with Image.open(pair.target_image) as image:
                    width, height = image.size
                if np.any(target < 0) or np.any(target[:, 0] > width) or np.any(target[:, 1] > height):
                    raise ValueError("Target landmarks outside image")
            record = {"pair_id": pair.pair_id, "group": pair.group,
                      "scorable": pair.target_landmarks is not None}
            if pair.target_landmarks is not None:
                record.update({"source_landmarks": len(source_ids), "target_landmarks": len(target_ids),
                    "common_landmarks": len(source),
                    "unmatched_source_ids": sorted(set(source_ids) - set(target_ids)),
                    "unmatched_target_ids": sorted(set(target_ids) - set(source_ids))})
            records.append(record)
        except (OSError, ValueError) as error:
            failures.append({"pair_id": pair.pair_id, "error": str(error)})
    return {"pairs": len(pairs), "groups": len({pair.group for pair in pairs}),
            "scorable_pairs": sum(record["scorable"] for record in records),
            "annotation_count_mismatches": [record for record in records
                if record.get("unmatched_source_ids") or record.get("unmatched_target_ids")],
            "failures": failures, "success": not failures}
