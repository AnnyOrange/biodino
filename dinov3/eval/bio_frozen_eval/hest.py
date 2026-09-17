"""HEST's explicit sample folds and train-only PCA/ridge evaluation.

Sample-disjoint official CSVs do not establish patient-disjointness without a
separate sample-to-patient table. HCC is an additional local task, not one of
the nine tasks in the original publication.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

CANONICAL_TASKS = frozenset({"CCRCC", "COAD", "IDC", "LUNG", "LYMPH_IDC", "PAAD", "PRAD", "READ", "SKCM"})
SOURCE_URL = "https://github.com/mahmoodlab/HEST"


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _strings(values):
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def _index(group):
    name = group.attrs.get("_index", "_index")
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    return _strings(group[name][:])


def _unique(values, label):
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate {label}")


def _identity_hash(values):
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode("utf-8")).hexdigest()


def _genes(task_root):
    path = task_root / "var_50genes.json"
    if not path.exists():
        raise FileNotFoundError(f"Required official variable-gene list missing: {path}")
    genes = json.loads(path.read_text())["genes"]
    _unique(genes, "genes")
    if not genes:
        raise ValueError("Empty gene list")
    return list(genes), path


def _records(task_root, split, fold):
    path = task_root / "splits" / f"{split}_{fold}.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"sample_id", "patches_path", "expr_path"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing HEST split columns: {path}")
        records = list(reader)
    if not records:
        raise ValueError(f"Empty split: {path}")
    _unique([r["sample_id"] for r in records], "sample IDs")
    for record in records:
        for key in ("patches_path", "expr_path"):
            resolved = (task_root / record[key]).resolve()
            if not resolved.is_relative_to(task_root.resolve()):
                raise ValueError(f"Split path escapes task root: {record[key]}")
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
    return records, path


def inspect_hest(root):
    """Bounded preflight: inspect split/identity metadata, not image payloads."""
    import h5py

    root = Path(root)
    result = {"source_url": SOURCE_URL, "status": "PASS", "tasks": {},
              "grouping": "official sample_id", "patient_disjoint_verified": False}
    if not root.is_dir():
        return {**result, "status": "FAIL", "errors": [f"Missing root: {root}"]}
    for task in sorted(p for p in root.iterdir() if p.is_dir() and (p / "splits").is_dir()):
        report = {"canonical_paper_task": task.name in CANONICAL_TASKS, "folds": {}, "errors": []}
        result["tasks"][task.name] = report
        try:
            genes, gene_path = _genes(task)
            report.update(genes=genes, gene_source=gene_path.name, gene_sha256=_hash(gene_path))
            train_folds = {int(p.stem.split("_")[-1]) for p in (task / "splits").glob("train_*.csv")}
            test_folds = {int(p.stem.split("_")[-1]) for p in (task / "splits").glob("test_*.csv")}
            if not train_folds or train_folds != test_folds:
                raise ValueError("Train/test fold sets differ or are empty")
            checked = {}
            for fold in sorted(train_folds):
                fold_report = {}
                identities = {}
                for split in ("train", "test"):
                    rows, path = _records(task, split, fold)
                    identities[split] = rows
                    spots = 0
                    for row in rows:
                        key = (row["patches_path"], row["expr_path"])
                        if key not in checked:
                            with h5py.File(task / key[0], "r") as patches:
                                bars = _strings(np.asarray(patches["barcode"]).reshape(-1))
                                _unique(bars, "patch barcodes")
                                if not bars:
                                    raise ValueError("Empty HEST patch sample")
                                if len(patches["img"]) != len(bars):
                                    raise ValueError("Patch/barcode counts differ")
                                if patches["img"].ndim != 4 or patches["img"].shape[-1] != 3:
                                    raise ValueError("HEST patches must be NHWC RGB")
                            with h5py.File(task / key[1], "r") as expr:
                                obs, var = _index(expr["obs"]), _index(expr["var"])
                                _unique(obs, "expression barcodes")
                                _unique(var, "expression genes")
                                if not set(bars).issubset(obs) or not set(genes).issubset(var):
                                    raise ValueError("Missing expression barcodes or genes")
                            checked[key] = {"spots": len(bars), "barcode_sha256": _identity_hash(bars),
                                            "expression_obs_sha256": _identity_hash(obs),
                                            "expression_var_sha256": _identity_hash(var)}
                        spots += checked[key]["spots"]
                    fold_report[split] = {"sample_ids": [r["sample_id"] for r in rows],
                                          "samples": len(rows), "spots": spots,
                                          "split_sha256": _hash(path)}
                    fold_report[split]["sample_identity_hashes"] = {
                        row["sample_id"]: checked[(row["patches_path"], row["expr_path"])] for row in rows}
                for key in ("sample_id", "patches_path", "expr_path"):
                    a = {r[key] for r in identities["train"]}
                    b = {r[key] for r in identities["test"]}
                    if key != "sample_id":
                        a = {(task / value).resolve() for value in a}
                        b = {(task / value).resolve() for value in b}
                    if a & b:
                        raise ValueError(f"Train/test leakage in {key}: {sorted(str(v) for v in a & b)}")
                report["folds"][str(fold)] = fold_report
        except (OSError, KeyError, ValueError, TypeError) as error:
            report["errors"].append(str(error))
            result["status"] = "FAIL"
    if not result["tasks"]:
        result.update(status="FAIL", errors=["No HEST tasks with splits found"])
    return result


def load_hest_targets(expr_path, genes, barcodes):
    """Return log1p counts in the requested barcode/gene order (no totals scaling)."""
    import h5py

    genes, barcodes = list(genes), list(barcodes)
    _unique(genes, "requested genes")
    _unique(barcodes, "requested barcodes")
    with h5py.File(expr_path, "r") as handle:
        obs, var = _index(handle["obs"]), _index(handle["var"])
        _unique(obs, "expression barcodes")
        _unique(var, "expression genes")
        obs_map, var_map = {v: i for i, v in enumerate(obs)}, {v: i for i, v in enumerate(var)}
        missing = (set(barcodes) - obs_map.keys()) | (set(genes) - var_map.keys())
        if missing:
            raise ValueError(f"Missing HEST expression identities: {sorted(missing)}")
        rows, cols = [obs_map[b] for b in barcodes], [var_map[g] for g in genes]
        source = handle["X"]
        values = np.zeros((len(rows), len(cols)), dtype=np.float64)
        if isinstance(source, h5py.Dataset):
            for out, row in enumerate(rows):
                values[out] = np.asarray(source[row])[cols]
        else:
            encoding = source.attrs.get("encoding-type", "")
            if isinstance(encoding, bytes):
                encoding = encoding.decode()
            indptr = source["indptr"][:]
            if encoding == "csr_matrix":
                mapping = {col: out for out, col in enumerate(cols)}
                for out, row in enumerate(rows):
                    lo, hi = int(indptr[row]), int(indptr[row + 1])
                    for col, value in zip(source["indices"][lo:hi], source["data"][lo:hi]):
                        if int(col) in mapping:
                            values[out, mapping[int(col)]] += value
            elif encoding == "csc_matrix":
                mapping = {row: out for out, row in enumerate(rows)}
                for out, col in enumerate(cols):
                    lo, hi = int(indptr[col]), int(indptr[col + 1])
                    for row, value in zip(source["indices"][lo:hi], source["data"][lo:hi]):
                        if int(row) in mapping:
                            values[mapping[int(row)], out] += value
            else:
                raise ValueError(f"Unsupported H5AD X encoding: {encoding}")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("HEST targets must be finite nonnegative counts")
    return np.log1p(values)


class HESTPatchDataset:
    """Lazy RGB image reads with explicit split and aligned expression targets."""

    def __init__(self, task_root, split="train", fold=0, genes=None, max_samples=None):
        import h5py

        if split not in {"train", "test"}:
            raise ValueError("HEST supplies train/test folds, not a validation split")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self.task_root = Path(task_root)
        self.genes = list(genes) if genes is not None else _genes(self.task_root)[0]
        self.records, self.split_path = _records(self.task_root, split, fold)
        self.samples, targets = [], []
        for record in self.records:
            patch_path = self.task_root / record["patches_path"]
            with h5py.File(patch_path, "r") as handle:
                bars = _strings(np.asarray(handle["barcode"]).reshape(-1))
                _unique(bars, "patch barcodes")
                if not bars:
                    raise ValueError("Empty HEST patch sample")
                if len(handle["img"]) != len(bars):
                    raise ValueError("Patch/barcode counts differ")
                if handle["img"].ndim != 4 or handle["img"].shape[-1] != 3:
                    raise ValueError("HEST patches must be NHWC RGB")
            if max_samples is not None:
                bars = bars[:max_samples - len(self.samples)]
            targets.append(load_hest_targets(self.task_root / record["expr_path"], self.genes, bars))
            self.samples.extend((patch_path, i, record["sample_id"], barcode) for i, barcode in enumerate(bars))
            if max_samples is not None and len(self.samples) >= max_samples:
                break
        self.targets = np.concatenate(targets, axis=0)
        self.paths = [f"{path}::{sample_id}::{barcode}" for path, _, sample_id, barcode in self.samples]
        self.protocol_complete = max_samples is None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import h5py
        from PIL import Image

        path, spot, _, _ = self.samples[index]
        with h5py.File(path, "r") as handle:
            pixels = np.asarray(handle["img"][spot])
        if pixels.dtype != np.uint8:
            raise ValueError("HEST patches must contain uint8 RGB pixels")
        return Image.fromarray(pixels), self.targets[index], self.paths[index]


def run_hest_probe_split(x_train, y_train, x_test, y_test, alpha_multiplier=1.0, latent_dim=256, seed=1):
    """Official HEST recipe; nonunit regularization multiplier is our extension."""
    from scipy.stats import pearsonr
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x_train, x_test, y_train, y_test = [np.asarray(a, dtype=np.float64) for a in (x_train, x_test, y_train, y_test)]
    if any(a.ndim != 2 or not np.isfinite(a).all() for a in (x_train, x_test, y_train, y_test)):
        raise ValueError("HEST features and targets must be finite two-dimensional arrays")
    if len(x_train) != len(y_train) or len(x_test) != len(y_test) or not len(x_test):
        raise ValueError("Feature/target row mismatch or empty test split")
    if x_train.shape[1] != x_test.shape[1] or y_train.shape[1] != y_test.shape[1] or not y_train.shape[1]:
        raise ValueError("Train/test feature or gene dimensions differ")
    if not isinstance(latent_dim, int) or latent_dim <= 0 or latent_dim > min(x_train.shape):
        raise ValueError("PCA latent_dim exceeds training dimensions; no silent reduction is allowed")
    if not np.isfinite(alpha_multiplier) or alpha_multiplier <= 0:
        raise ValueError("alpha_multiplier must be finite and positive")
    transform = make_pipeline(StandardScaler(), PCA(n_components=latent_dim, random_state=seed))
    a, b = transform.fit_transform(x_train), transform.transform(x_test)
    alpha = alpha_multiplier * 100.0 / (latent_dim * y_train.shape[1])
    predictions = Ridge(alpha=alpha, solver="lsqr", fit_intercept=False, max_iter=1000,
                        random_state=seed).fit(a, y_train).predict(b)
    correlations = []
    for gene in range(y_test.shape[1]):
        truth, pred = y_test[:, gene], predictions[:, gene]
        correlations.append(float(pearsonr(truth, pred).statistic) if len(truth) > 1 and np.std(truth) > 0 and np.std(pred) > 0 else float("nan"))
    # Preserve official np.mean propagation for undefined genes, never omit them.
    headline = float(np.mean(correlations))
    return {"task": "regression", "n_train": len(a), "n_test": len(b),
            "metrics": {"gene_wise_pearson": headline, "per_gene_pearson": correlations,
                        "r2": float(r2_score(y_test, predictions)),
                        "mae": float(mean_absolute_error(y_test, predictions))},
            "hyperparameters": {"latent_dim": latent_dim, "alpha": alpha,
                                 "alpha_multiplier": alpha_multiplier, "seed": seed,
                                 "solver": "lsqr", "fit_intercept": False, "max_iter": 1000},
            "protocol_decision": "OFFICIAL" if alpha_multiplier == 1.0 and latent_dim == 256 else "PROPOSED_BY_US",
            "metric_valid": bool(np.isfinite(headline)),
            "predictions": predictions, "targets": y_test}
