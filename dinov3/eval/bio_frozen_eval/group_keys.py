# Per-sample group keys for leakage-safe (group-aware) train/test splits.
#
# Several classification/regression benchmarks ship NO official train/test split,
# so the frozen probe historically did an internal random split. A random split at
# the crop/image level leaks when several crops come from the same source slide /
# well / field (the same biological source lands in both train and test, inflating
# scores). For publication we instead split by SOURCE group: every crop from one
# source goes entirely to train or entirely to test.
#
# This module derives, for each such dataset, a per-sample group string from the
# file path (or the dataset's CSV). The keys are consumed by ``make_group_splits``
# to produce the committed, documented ``splits/<dataset>.json`` files.
from __future__ import annotations

import csv
import hashlib
import re
from functools import lru_cache
from pathlib import Path

# Human-readable rule per dataset (recorded into the split JSON for documentation).
GROUP_KEY_RULES: dict[str, str] = {
    "cyclops-protein-loc": "source image = filename stem minus the trailing _<channel> (e.g. ACTIN_10001_gfp -> ACTIN_10001)",
    "bbbc048-cellcycle": "well/field = numeric filename prefix before the first '_' (e.g. 10005_Ch3.ome -> 10005)",
    "midog25-atypical": "source slide = the CSV 'filename' column for each image_id (e.g. 201.tiff)",
    "bbbc013": "plate well = letter-number position in the filename (Channel1-01-A-01.BMP -> A-01)",
    "bbbc005": "plate/count/field = SIMCEPImages_<well>_C<count>_F<field> (… A01_C1_F1_s01_w1.TIF -> A01_C1_F1)",
    "lc25000": "published LC25000-clean original-tile group_id; all augmented descendants stay in one split",
}

# Datasets with no official test split, evaluated with a fixed group-aware split.
GROUP_SPLIT_DATASETS = sorted(GROUP_KEY_RULES.keys())
LC25000_GROUPS = Path(__file__).resolve().parents[3] / "outputs/02_eval_inputs/formal_v4/lc25000_clean_groups.csv"
LC25000_GROUPS_SHA256 = "2ff9e992f759f265dfc9aa2363de0dfcc8e6559884d8bb02a4c925214fdf7e23"


@lru_cache(maxsize=1)
def _lc25000_groups() -> dict[tuple[str, str], str]:
    if hashlib.sha256(LC25000_GROUPS.read_bytes()).hexdigest() != LC25000_GROUPS_SHA256:
        raise ValueError("LC25000 source-tile annotations differ from the pinned publication")
    mapping = {}
    with LC25000_GROUPS.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["stem"], row["label"])
            if key in mapping or Path(row["filename"]).stem != row["stem"]:
                raise ValueError(f"Duplicate or malformed LC25000 annotation: {key}")
            mapping[key] = row["group_id"]
    if len(mapping) != 25000 or len(set(mapping.values())) != 1246:
        raise ValueError("LC25000 annotation inventory or group count changed")
    return mapping


def _sample_paths(dataset) -> list[Path]:
    """Paths in dataset iteration order (matches frozen-feature row order)."""
    out: list[Path] = []
    for s in dataset.samples:
        if hasattr(s, "image_path"):  # RegressionSample
            out.append(Path(s.image_path))
        else:  # (path, label) tuple
            out.append(Path(s[0]))
    return out


def _cyclops_key(p: Path) -> str:
    return p.stem.rsplit("_", 1)[0]


def _bbbc048_key(p: Path) -> str:
    return p.stem.split("_", 1)[0]


def _bbbc013_key(p: Path) -> str:
    m = re.search(r"Channel\d+-\d+-([A-Za-z]-\d+)", p.name)
    if not m:
        raise ValueError(f"bbbc013: cannot parse well from {p.name}")
    return m.group(1)


def _bbbc005_key(p: Path) -> str:
    m = re.search(r"_([A-Za-z]\d+)_C(\d+)_F(\d+)_", p.name)
    if not m:
        raise ValueError(f"bbbc005: cannot parse plate/count/field from {p.name}")
    return f"{m.group(1)}_C{m.group(2)}_F{m.group(3)}"


def _midog25_keys(dataset, benchmark_root) -> list[str]:
    # The dataset stores only (image_path=image_root/<image_id>, label); the source
    # slide lives in the CSV 'filename' column. Re-read the CSV to map image_id -> slide.
    csv_path = Path(getattr(dataset, "csv_path"))
    id_to_slide: dict[str, str] = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            id_to_slide[str(row["image_id"])] = str(row["filename"])
    keys = []
    for p in _sample_paths(dataset):
        image_id = p.name
        if image_id not in id_to_slide:
            raise ValueError(f"midog25: image_id {image_id} not in {csv_path}")
        keys.append(id_to_slide[image_id])
    return keys


def group_keys(dataset_name: str, dataset, benchmark_root=None) -> list[str]:
    """Group key per sample, aligned to ``dataset`` iteration order."""
    if dataset_name == "midog25-atypical":
        return _midog25_keys(dataset, benchmark_root)
    paths = _sample_paths(dataset)
    if dataset_name == "lc25000":
        mapping = _lc25000_groups()
        keys = [(p.stem, p.parent.name) for p in paths]
        if len(keys) == 25000 and set(keys) != set(mapping):
            raise ValueError("LC25000 image inventory differs from published group annotation")
        if any(key not in mapping for key in keys):
            raise ValueError("LC25000 sample has no original-tile group annotation")
        return [mapping[key] for key in keys]
    fn = {
        "cyclops-protein-loc": _cyclops_key,
        "bbbc048-cellcycle": _bbbc048_key,
        "bbbc013": _bbbc013_key,
        "bbbc005": _bbbc005_key,
    }.get(dataset_name)
    if fn is None:
        raise KeyError(f"No group-key rule for dataset {dataset_name}")
    return [fn(p) for p in paths]
