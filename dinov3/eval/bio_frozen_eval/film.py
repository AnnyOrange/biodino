"""Proposed small-sample FILM age classification, never spectral-plane splitting."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np
from scipy.io import loadmat
import tifffile
import torch
from torch.utils.data import Dataset

AGES = ("D2", "D4", "D6", "D10")
DEFAULT_ROOT = Path("/mnt/huawei_deepcad/benchmark/ood/ood_classification/datasets/FILM")
ARCHIVE_BYTES = 1822990074
ARCHIVE_MD5 = "5cf65ce14edf0f770984f375abfa49d6"
SOURCES = {
    "data": "https://api.figshare.com/v2/articles/31302607",
    "paper": "https://www.nature.com/articles/s41592-026-03090-1",
    "full_methods": "https://arxiv.org/abs/2504.04305v3",
    "code": "https://github.com/buchenglab/SPEND",
}


def _hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_stack(path: Path) -> np.ndarray:
    # Author calibration and methods identify the leading axis as wavenumber,
    # although ImageJ encodes most stacks as ZYX and one as TYX.
    array = tifffile.imread(path)
    if array.shape != (126, 200, 200) or array.dtype != np.float32:
        raise ValueError(f"FILM expected float32 (126,200,200), got {array.shape}/{array.dtype}: {path}")
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite FILM spectrum: {path}")
    return array


def normalize_stack(array: np.ndarray) -> torch.Tensor:
    if not np.isfinite(array).all():
        raise ValueError("FILM normalization requires finite values")
    lo, hi = np.percentile(array, (1, 99))
    if hi <= lo:
        raise ValueError("FILM stack lacks intensity dynamic range")
    normalized = np.clip((array.astype(np.float32) - lo) / (hi - lo), 0, 1)
    return torch.from_numpy(np.ascontiguousarray(normalized, dtype=np.float32))


def grouped_folds(records: list[dict], seed: int = 0, n_folds: int = 3) -> list[dict]:
    """Class-stratified group round robin; identical sample IDs stay together."""
    if n_folds < 2:
        raise ValueError("At least two folds required")
    labels: dict[str, int] = {}
    for record in records:
        group, target = record["group_id"], int(record["target"])
        if group in labels and labels[group] != target:
            raise ValueError(f"Conflicting group labels: {group}")
        labels[group] = target
    assignments = {}
    rng = np.random.default_rng(seed)
    for target in sorted(set(labels.values())):
        groups = sorted(group for group in labels if labels[group] == target)
        if len(groups) < n_folds:
            raise ValueError(f"Class {target} has too few groups for {n_folds} folds")
        for index, group in enumerate(rng.permutation(groups)):
            assignments[str(group)] = index % n_folds
    return [
        {"fold": fold,
         "train_indices": [i for i, r in enumerate(records) if assignments[r["group_id"]] != fold],
         "test_indices": [i for i, r in enumerate(records) if assignments[r["group_id"]] == fold]}
        for fold in range(n_folds)
    ]


def build_film_manifest(root: Path = DEFAULT_ROOT, seeds: tuple[int, ...] = (0, 1, 2)) -> dict:
    root = Path(root)
    archive = root / "raw/FILM_data.zip"
    if archive.stat().st_size != ARCHIVE_BYTES or _hash(archive, "md5") != ARCHIVE_MD5:
        raise ValueError("FILM official archive size/MD5 mismatch")
    records, seen_pixels = [], {}
    with zipfile.ZipFile(archive) as source:
        bad = source.testzip()
        if bad:
            raise ValueError(f"FILM corrupt archive member: {bad}")
        for path in sorted((root / "extracted/Fig4 Aging/replicates").rglob("*.tif")):
            age = path.parent.name
            match = re.fullmatch(r"my_model_Jianpeng(?:_master)?_4unet_sample([1-7])(?: fov2)? mipf\.tif", path.name)
            if age not in AGES or match is None:
                raise ValueError(f"Unrecognized FILM age identity: {path}")
            relative = path.relative_to(root / "extracted").as_posix()
            member = source.getinfo(relative)
            if member.file_size != path.stat().st_size:
                raise ValueError(f"Incomplete extracted stack: {relative}")
            with source.open(member) as stream:
                archive_sha = hashlib.sha256(stream.read()).hexdigest()
            if archive_sha != _hash(path):
                raise ValueError(f"Extracted stack differs from official archive: {relative}")
            array = read_stack(path)
            pixel_sha = hashlib.sha256(array.tobytes()).hexdigest()
            if pixel_sha in seen_pixels:
                raise ValueError(f"Duplicate FILM pixels: {relative} / {seen_pixels[pixel_sha]}")
            seen_pixels[pixel_sha] = relative
            group = f"{age}_sample{match.group(1)}"
            records.append({"sample_id": relative, "path": "extracted/" + relative,
                            "group_id": group, "age": age, "target": AGES.index(age),
                            "shape": list(array.shape), "sha256": archive_sha,
                            "pixel_sha256": pixel_sha})
    counts = Counter(r["age"] for r in records)
    groups = {r["group_id"] for r in records}
    if counts != Counter(dict.fromkeys(AGES, 7)) or len(groups) != 27:
        raise ValueError(f"Expected 28 age stacks /27 groups, found {dict(counts)}/{len(groups)}")
    wavenumbers = loadmat(root / "extracted/Power calibration/wavenumber.mat")["wavenumber"].ravel()
    if wavenumbers.shape != (126,) or not np.all(np.diff(wavenumbers) > 0):
        raise ValueError("Invalid FILM official spectral calibration")
    repetitions = []
    for seed in seeds:
        folds = grouped_folds(records, seed)
        for fold in folds:
            outer_train = fold["train_indices"]
            inner = grouped_folds([records[i] for i in outer_train], seed)
            fold["inner_folds"] = [
                {"fold": f["fold"], "train_indices": [outer_train[i] for i in f["train_indices"]],
                 "val_indices": [outer_train[i] for i in f["test_indices"]]} for f in inner]
        repetitions.append({"seed": seed, "folds": folds})
    return {
        "dataset": "FILM", "task": "c_elegans_age_classification",
        "protocol_source": "PROPOSED_BY_US", "protocol_id": "film-age-grouped3fold-v1",
        "official_split": "No released age-classification split; original study uses t-tests of lysosomal spectra",
        "license": "CC BY 4.0", "sources": SOURCES,
        "archive": {"bytes": ARCHIVE_BYTES, "md5": ARCHIVE_MD5, "crc_valid": True},
        "classes": list(AGES), "expected_stacks": 28, "expected_groups": 27,
        "class_counts": dict(counts), "wavenumbers_cm_minus1": wavenumbers.tolist(),
        "grouping": "Named age/sample identity; D6_sample1 and fov2 inseparable; all planes/tiles remain together",
        "preprocessing": "Released SPEND-denoised full stack; shared per-stack p1/p99 scaling and clipping; tensor CHW126; existing encoder resize256/center224",
        "channel_policy": "mean3",
        "channel_policy_candidates": ["mean3", "native (only true multichannel backbone)"],
        "channel_policy_note": "mean3 is the existing fixed spectral-mean RGB baseline, not a chemically calibrated reconstruction; native preserves all126 bands",
        "evaluator": "dinov3.eval.bio_frozen_eval.probes.run_classification_probe_split",
        "feature_aggregation": "Average FOV features per named sample before fitting/scoring; equal weight per group",
        "primary_metric": "balanced_accuracy", "secondary_metrics": ["accuracy", "macro_f1"],
        "aggregation": "Mean over three outer folds within each seed, then mean and population SD across seeds; save every fold; no independent-repetition CI claim",
        "seeds": list(seeds), "repetitions": repetitions, "records": records,
        "proposed_search": {"features": ["last", "4-even"], "logistic_C": [0.01, 0.1, 1, 10, 100], "max_iter": 10000},
        "selection": "Within each outer fold select on1TB inner grouped3fold validation only, freeze fold-specific feature/C/channel config, reuse exactly on5TB/20TB; no outer-test selection",
        "selected_using_1TB": False, "frozen": False, "data_preflight": "PASS",
        "limitations": ["Only27 observed named samplegroups; biological batch/animal linkage beyond filenames unavailable",
                        "Observational age labels; not an official classification benchmark",
                        "Author SPEND training provenance is not assigned to our folds; released denoised-data benchmark only",
                        "No fluorescence/power spectral calibration or lysosome segmentation reconstructed; not paper quantitative metabolic assay"]}


class FILMAgeDataset(Dataset):
    def __init__(self, root: Path, manifest: dict, split: str = "all", repetition: int = 0,
                 fold: int = 0, inner_fold: int | None = None):
        self.root = Path(root)
        if split == "all":
            indices = range(len(manifest["records"]))
        else:
            spec = manifest["repetitions"][repetition]["folds"][fold]
            if inner_fold is not None:
                spec = spec["inner_folds"][inner_fold]
            key = {"train": "train_indices", "test": "test_indices", "val": "val_indices"}.get(split)
            if key is None or key not in spec:
                raise ValueError(f"Split {split!r} unavailable for specified outer/inner fold")
            indices = spec[key]
        self.records = [manifest["records"][i] for i in indices]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return normalize_stack(read_stack(self.root / record["path"])), record["target"], record["sample_id"]


def aggregate_group_features(features: np.ndarray, records: list[dict], indices: list[int]):
    if len(features) != len(records) or not np.isfinite(features).all():
        raise ValueError("Finite features in full manifest order required")
    groups = sorted({records[i]["group_id"] for i in indices})
    x, y = [], []
    for group in groups:
        selected = [i for i in indices if records[i]["group_id"] == group]
        labels = {int(records[i]["target"]) for i in selected}
        if len(labels) != 1:
            raise ValueError(f"Conflicting group labels: {group}")
        x.append(features[selected].mean(axis=0))
        y.append(labels.pop())
    return np.asarray(x), np.asarray(y), groups


def run_film_fold_probe(features: np.ndarray, manifest: dict, repetition: int = 0, fold: int = 0,
                        C: float = 1.0, inner_fold: int | None = None) -> dict:
    from .probes import run_classification_probe_split

    spec = manifest["repetitions"][repetition]["folds"][fold]
    if inner_fold is not None:
        spec = spec["inner_folds"][inner_fold]
    evaluation = spec["val_indices" if inner_fold is not None else "test_indices"]
    train_x, train_y, train_groups = aggregate_group_features(features, manifest["records"], spec["train_indices"])
    test_x, test_y, test_groups = aggregate_group_features(features, manifest["records"], evaluation)
    if set(train_groups) & set(test_groups):
        raise ValueError("FILM sample leakage")
    result = run_classification_probe_split(train_x, train_y, test_x, test_y, C=C,
                                            seed=manifest["repetitions"][repetition]["seed"]).to_dict()
    result.update({"repetition": repetition, "fold": fold, "inner_fold": inner_fold,
                   "C": C, "train_groups": train_groups, "evaluation_groups": test_groups})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_film_manifest(args.root)
    dataset = FILMAgeDataset(args.root, manifest)
    for index in range(len(dataset)):
        image, _, _ = dataset[index]
        if image.shape != (126, 200, 200) or not torch.isfinite(image).all():
            raise ValueError("FILM loader smoke failed")
    manifest["loader_smoke"] = {"status": "PASS", "all_stacks_read": len(dataset)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "stacks": len(dataset), "groups": manifest["expected_groups"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
