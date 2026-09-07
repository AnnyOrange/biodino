#!/usr/bin/env python3
"""CPU k-shot probes on already-extracted frozen classification features.

Reuses train/test npz caches. Does not load checkpoints or GPUs.
Canonical probe: StandardScaler + class-balanced LogisticRegression.
Scans e15 15 ckpts and every D-scale epoch that has features.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("/mnt/huawei_deepcad/dinov3")
OUT_DEFAULT = ROOT / "plot/fig2/kshot"

DATASETS = [
    "pathmnist",
    "tissuemnist",
    "dermamnist",
    "bbbc048-cellcycle",
    "bloodmnist",
    "nct-crc-he",
    "organamnist",
    "organcmnist",
    "organsmnist",
    "octmnist",
    "pneumoniamnist",
    "retinamnist",
    "breastmnist",
    "lc25000",
    "midog25-atypical",
    "cyclops-protein-loc",
    "chammi-cp-task3",
    "chammi-allen-task2",
    "chammi-allen-task1",
    "chammi-cp-task1",
    "chammi-cp-task2",
    "chammi-hpa-task1",
    "chammi-hpa-task2",
    "pcam",
]
GROUP_SPLIT = {"bbbc048-cellcycle", "cyclops-protein-loc", "midog25-atypical"}
# LC25000 等只有整包 npz（无官方 train/test），按类别分层切一次，全模型共用。
WHOLE_STRATIFIED_SPLIT = {"lc25000"}
COMPUTE_CKPTS = (
    1024,
    2049,
    3074,
    4099,
    5124,
    6149,
    7174,
    8199,
    9224,
    10249,
    11274,
    12299,
    13324,
    14349,
    15374,
)
PARAMS = {"S+": 21e6, "B": 86e6, "L": 300e6, "H+": 840e6}
POOLS = {
    10: ("0.1M", 823, 104_877),
    20: ("0.2M", 1639, 209_754),
    50: ("0.5M", 4103, 524_385),
    100: ("1M", 8199, 1_048_771),
}
DSCALE_LR = {"S+": ("Splus", "lr2e4"), "B": ("B", "lr1p5e4"), "L": ("L", "lr1e4"), "H+": ("Hplus", "lr5e5")}
E15_EVAL = {
    "S+": ROOT
    / "outputs/01_training_runs/HS6_Splus_robust_biosafe256_gb1024_lr2e4_wu3_tw30_nosig_e15_seed0_8x5090xr_20260821b/eval/hs6_online_full_20260824",
    "B": ROOT
    / "outputs/01_training_runs/HS6_B_robust_biosafe256_gb1024_lr1p5e4_wu3_tw30_nosig_e15_seed0_8x5090hxw_20260818/eval/hs6_online_full_20260819",
    "L": ROOT
    / "outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_seed0_20260818/eval/hs6_online_full_20260819",
    "H+": ROOT
    / "outputs/01_training_runs/HS6_Hplus_robust_biosafe256_gb1024_lr5e5_wu3_tw30_nosig_e15_seed0_4xH100_20260818/eval/hs6_online_full_20260819",
}
CSV_FIELDS = [
    "model",
    "axis",
    "pool",
    "samples",
    "ckpt",
    "epoch",
    "params",
    "dataset",
    "k",
    "seed",
    "n_train",
    "n_test",
    "n_classes",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "error_balanced",
    "error_macro_f1",
    "full_macro_f1",
    "feature_train",
]


def dscale_eval(nick: str, label: int, run_root: Path | None = None) -> Path:
    key, lr = DSCALE_LR[nick]
    base = run_root or (ROOT / "outputs/01_training_runs")
    return (
        base
        / f"HS6_Dscale_{key}_robust_biosafe256_gb1024_{lr}_wu3_tw30_nosig_e8_random{label}_seed0_20260820"
        / "eval/e8_full_20260820"
    )


def e8_ckpts(frac: int) -> list[int]:
    last = POOLS[frac][1]
    return [int(last * k / 8) for k in range(1, 9)]


def epoch_of_ckpt(frac: int, ckpt: int) -> int:
    expected = e8_ckpts(frac)
    if ckpt in expected:
        return expected.index(ckpt) + 1
    return int(min(range(8), key=lambda i: abs(expected[i] - ckpt))) + 1


def load_npz_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as pack:
        return np.asarray(pack["features"], dtype=np.float32), np.asarray(pack["labels"]).reshape(-1)


def find_feature_pair(eval_dir: Path, dataset: str, ckpt: int) -> tuple[Path | None, Path | None, Path | None]:
    root = eval_dir / "bio_classification" / dataset / str(ckpt) / "features" / dataset
    if not root.is_dir():
        return None, None, None
    trains = sorted(root.glob("*_train.npz"))
    tests = sorted(root.glob("*_test.npz"))
    wholes = sorted(p for p in root.glob("*.npz") if "_train" not in p.name and "_test" not in p.name)
    train = tests_p = whole = None
    if trains and tests:
        stem = trains[0].name[: -len("_train.npz")]
        match = [p for p in tests if p.name.startswith(stem)]
        train = trains[0]
        tests_p = match[0] if match else tests[0]
    if wholes:
        whole = wholes[0]
    return train, tests_p, whole


def kshot_indices(labels: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    picked: list[np.ndarray] = []
    for label in np.unique(labels):
        candidates = np.flatnonzero(labels == label)
        n_keep = min(len(candidates), k)
        if n_keep <= 0:
            continue
        picked.append(np.sort(rng.choice(candidates, size=n_keep, replace=False)))
    if not picked:
        raise ValueError("no class had any training samples")
    return np.sort(np.concatenate(picked))


def probe(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=10000, class_weight="balanced", n_jobs=1),
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    bacc = float(balanced_accuracy_score(y_test, pred))
    f1 = float(f1_score(y_test, pred, average="macro", zero_division=0))
    return {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_classes": int(len(np.unique(y_train))),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": bacc,
        "macro_f1": f1,
        "error_balanced": float(1.0 - bacc),
        "error_macro_f1": float(1.0 - f1),
    }


def full_macro_f1(eval_dir: Path, dataset: str, ckpt: int) -> float | None:
    path = eval_dir / "bio_classification" / dataset / str(ckpt) / "last_result.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    value = payload.get("macro_f1")
    return None if value is None else float(value)


_GROUP_INDEX_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_STRAT_INDEX_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def stratified_indices(
    dataset: str, labels: np.ndarray, cache_dir: Path, *, test_frac: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    if dataset in _STRAT_INDEX_CACHE:
        return _STRAT_INDEX_CACHE[dataset]
    cache = cache_dir / f"{dataset}_stratified_indices.npz"
    if cache.is_file():
        pack = np.load(cache)
        train_idx, test_idx = pack["train_idx"], pack["test_idx"]
        _STRAT_INDEX_CACHE[dataset] = train_idx, test_idx
        return train_idx, test_idx
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        if len(idx) <= 1:
            train_parts.append(idx)
            continue
        n_test = max(1, int(round(len(idx) * test_frac)))
        n_test = min(n_test, len(idx) - 1)
        test_parts.append(np.sort(idx[:n_test]))
        train_parts.append(np.sort(idx[n_test:]))
    train_idx = np.sort(np.concatenate(train_parts)) if train_parts else np.array([], dtype=np.int64)
    test_idx = np.sort(np.concatenate(test_parts)) if test_parts else np.array([], dtype=np.int64)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache, train_idx=train_idx, test_idx=test_idx)
    _STRAT_INDEX_CACHE[dataset] = train_idx, test_idx
    print(f"[split] stratified {dataset} train={len(train_idx)} test={len(test_idx)} -> {cache}", flush=True)
    return train_idx, test_idx


def group_indices(dataset: str, cache_dir: Path, benchmark_root: Path) -> tuple[np.ndarray, np.ndarray]:
    if dataset in _GROUP_INDEX_CACHE:
        return _GROUP_INDEX_CACHE[dataset]
    cache = cache_dir / f"{dataset}_group_indices.npz"
    if cache.is_file():
        pack = np.load(cache)
        train_idx, test_idx = pack["train_idx"], pack["test_idx"]
        _GROUP_INDEX_CACHE[dataset] = train_idx, test_idx
        return train_idx, test_idx
    sys.path.insert(0, str(ROOT))
    from dinov3.eval.bio_frozen_eval.make_group_splits import group_split_indices
    from dinov3.eval.bio_frozen_eval.registry import build_dataset

    dataset_obj, _task = build_dataset(dataset, "train", None, None, benchmark_root=benchmark_root)
    train_idx, test_idx = group_split_indices(dataset, dataset_obj)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache, train_idx=train_idx, test_idx=test_idx)
    _GROUP_INDEX_CACHE[dataset] = train_idx, test_idx
    print(f"[split] cached {dataset} train={len(train_idx)} test={len(test_idx)} -> {cache}", flush=True)
    return train_idx, test_idx


def load_xy(
    eval_dir: Path,
    dataset: str,
    ckpt: int,
    cache_dir: Path,
    benchmark_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str] | None:
    train_p, test_p, whole_p = find_feature_pair(eval_dir, dataset, ckpt)
    if train_p is not None and test_p is not None:
        x_train, y_train = load_npz_xy(train_p)
        x_test, y_test = load_npz_xy(test_p)
        return x_train, y_train, x_test, y_test, str(train_p)
    if dataset in GROUP_SPLIT and whole_p is not None:
        features, labels = load_npz_xy(whole_p)
        train_idx, test_idx = group_indices(dataset, cache_dir, benchmark_root)
        if max(int(train_idx.max()), int(test_idx.max())) >= len(labels):
            raise ValueError(f"{dataset} group indices out of range for {whole_p}")
        return features[train_idx], labels[train_idx], features[test_idx], labels[test_idx], str(whole_p)
    if dataset in WHOLE_STRATIFIED_SPLIT and whole_p is not None:
        features, labels = load_npz_xy(whole_p)
        train_idx, test_idx = stratified_indices(dataset, labels, cache_dir)
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError(f"{dataset} stratified split empty for {whole_p}")
        if max(int(train_idx.max()), int(test_idx.max())) >= len(labels):
            raise ValueError(f"{dataset} stratified indices out of range for {whole_p}")
        return features[train_idx], labels[train_idx], features[test_idx], labels[test_idx], str(whole_p)
    return None


def existing_keys(csv_path: Path) -> set[tuple]:
    if not csv_path.is_file():
        return set()
    keys = set()
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            keys.add((row["model"], row["axis"], row["pool"], row["ckpt"], row["dataset"], row["k"], row["seed"]))
    return keys


def append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.is_file()
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})


def listed_ckpts(eval_dir: Path, dataset: str) -> list[int]:
    root = eval_dir / "bio_classification" / dataset
    if not root.is_dir():
        return []
    out = []
    for path in root.iterdir():
        if path.name.isdigit() and path.is_dir():
            out.append(int(path.name))
    return sorted(out)


def iter_jobs(extra_e15: dict[str, Path], extra_dscale_root: Path | None, extra_eval: list[tuple[str, str, Path]]):
    e15 = dict(E15_EVAL)
    e15.update(extra_e15)
    for nick, ed in e15.items():
        if not ed.is_dir():
            continue
        for dataset in DATASETS:
            ckpts = listed_ckpts(ed, dataset) or list(COMPUTE_CKPTS)
            for ckpt in ckpts:
                epoch = COMPUTE_CKPTS.index(ckpt) + 1 if ckpt in COMPUTE_CKPTS else 0
                yield {
                    "model": nick,
                    "axis": "compute",
                    "pool": "1M",
                    "samples": 1_048_771,
                    "ckpt": ckpt,
                    "epoch": epoch,
                    "params": PARAMS[nick],
                    "dataset": dataset,
                    "eval_dir": ed,
                }
    for nick in PARAMS:
        for label, (pool, _last, samples) in POOLS.items():
            ed = dscale_eval(nick, label, extra_dscale_root)
            if not ed.is_dir():
                continue
            for dataset in DATASETS:
                ckpts = listed_ckpts(ed, dataset) or e8_ckpts(label)
                for ckpt in ckpts:
                    yield {
                        "model": nick,
                        "axis": "data",
                        "pool": pool,
                        "samples": samples,
                        "ckpt": ckpt,
                        "epoch": epoch_of_ckpt(label, ckpt),
                        "params": PARAMS[nick],
                        "dataset": dataset,
                        "eval_dir": ed,
                    }
    for nick, axis, ed in extra_eval:
        if not ed.is_dir() or nick not in PARAMS:
            continue
        pool = "1M" if axis == "compute" else "unknown"
        samples = 1_048_771
        for dataset in DATASETS:
            for ckpt in listed_ckpts(ed, dataset):
                yield {
                    "model": nick,
                    "axis": axis,
                    "pool": pool,
                    "samples": samples,
                    "ckpt": ckpt,
                    "epoch": 0,
                    "params": PARAMS[nick],
                    "dataset": dataset,
                    "eval_dir": ed,
                }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, default=OUT_DEFAULT / "kshot_raw.csv")
    parser.add_argument("--skip-csv", action="append", default=[], help="existing CSVs whose keys are skipped")
    parser.add_argument("--ks", default="5,10")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--axes", nargs="*", default=None, choices=["compute", "data"])
    parser.add_argument("--eval-root", action="append", default=[], help="nick=path e15 overrides")
    parser.add_argument("--dscale-run-root", type=Path, default=None)
    parser.add_argument("--extra-eval", action="append", default=[], help="nick:axis:path")
    parser.add_argument("--benchmark-root", type=Path, default=Path("/mnt/huawei_deepcad/benchmark"))
    parser.add_argument("--code-root", type=Path, default=ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global ROOT
    ROOT = args.code_root
    extra_e15 = {}
    for item in args.eval_root:
        nick, path = item.split("=", 1)
        extra_e15[nick] = Path(path)
    extra_eval = []
    for item in args.extra_eval:
        nick, axis, path = item.split(":", 2)
        extra_eval.append((nick, axis, Path(path)))
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    keep_ds = set(args.datasets) if args.datasets else set(DATASETS)
    keep_models = set(args.models) if args.models else set(PARAMS)
    keep_axes = set(args.axes) if args.axes else {"compute", "data"}
    cache_dir = args.output_csv.parent
    done = existing_keys(args.output_csv)
    for extra in args.skip_csv:
        done |= existing_keys(Path(extra))
    n_run = n_skip = n_miss = 0
    for job in iter_jobs(extra_e15, args.dscale_run_root, extra_eval):
        if job["dataset"] not in keep_ds or job["model"] not in keep_models or job["axis"] not in keep_axes:
            continue
        loaded = None
        for k in ks:
            for seed in seeds:
                key = (
                    job["model"],
                    job["axis"],
                    str(job["pool"]),
                    str(job["ckpt"]),
                    job["dataset"],
                    str(k),
                    str(seed),
                )
                if key in done:
                    n_skip += 1
                    continue
                if loaded is None:
                    try:
                        loaded = load_xy(
                            job["eval_dir"],
                            job["dataset"],
                            job["ckpt"],
                            cache_dir,
                            args.benchmark_root,
                        )
                    except Exception as exc:
                        print(
                            f"[fail] {job['model']} {job['axis']} {job['dataset']} ck{job['ckpt']}: {exc}",
                            flush=True,
                        )
                        loaded = False
                        n_miss += 1
                        break
                    if loaded is None:
                        n_miss += 1
                        break
                if loaded is False:
                    break
                x_train, y_train, x_test, y_test, feat_path = loaded
                idx = kshot_indices(np.asarray(y_train).reshape(-1), k, seed)
                metrics = probe(x_train[idx], y_train[idx], x_test, y_test)
                full = full_macro_f1(job["eval_dir"], job["dataset"], job["ckpt"])
                row = {
                    **{
                        field: job[field]
                        for field in ("model", "axis", "pool", "samples", "ckpt", "epoch", "params", "dataset")
                    },
                    "k": k,
                    "seed": seed,
                    **metrics,
                    "full_macro_f1": "" if full is None else f"{full:.8f}",
                    "feature_train": feat_path,
                }
                append_row(args.output_csv, row)
                done.add(key)
                n_run += 1
                if n_run % 25 == 0:
                    print(
                        f"[progress] wrote={n_run} skip={n_skip} miss={n_miss} "
                        f"last={job['model']} {job['dataset']} ck{job['ckpt']} k={k} seed={seed} "
                        f"err={metrics['error_balanced']:.3f}",
                        flush=True,
                    )
    print(f"[done] wrote={n_run} skip={n_skip} miss_or_absent={n_miss} csv={args.output_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
