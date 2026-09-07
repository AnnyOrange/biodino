#!/usr/bin/env python3
"""Rebuild C / N / D wide lattices for every evaluated dataset."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/mnt/huawei_deepcad/dinov3")
DATA = ROOT / "plot/fig2/data"
KDIR = ROOT / "plot/fig2/kshot"
RUNS = ROOT / "outputs/01_training_runs"

MODELS = ["S+", "B", "L", "H+"]
POOLS = ["0.1M", "0.2M", "0.5M", "1M"]
LAST_D = {"0.1M": 823, "0.2M": 1639, "0.5M": 4103, "1M": 8199}
CKPT_C = {"e1": 1024, "e2": 2049, "e4": 4099}
NICKDIR = {"S+": "Splus", "B": "B", "L": "L", "H+": "Hplus"}
LR = {"S+": "lr2e4", "B": "lr1p5e4", "L": "lr1e4", "H+": "lr5e5"}
RAND = {"0.1M": "random10", "0.2M": "random20", "0.5M": "random50", "1M": "random100"}

CSCALE_DS = [
    "bloodmnist",
    "tissuemnist",
    "cyclops-protein-loc",
    "pathmnist",
    "dermamnist",
    "bbbc048-cellcycle",
    "nct-crc-he",
    "chammi-allen-task1",
    "chammi-allen-task2",
    "chammi-cp-task3",
]
# e1/e2/e4 10-shot only exists for the narrowed C-scale worker set
CSCALE_K10_DS = {"bloodmnist", "tissuemnist", "cyclops-protein-loc"}

CLS_ORDER = [
    "bloodmnist",
    "tissuemnist",
    "cyclops-protein-loc",
    "pathmnist",
    "dermamnist",
    "bbbc048-cellcycle",
    "nct-crc-he",
    "chammi-allen-task1",
    "chammi-allen-task2",
    "chammi-cp-task3",
    "breastmnist",
    "octmnist",
    "organamnist",
    "organcmnist",
    "organsmnist",
    "pneumoniamnist",
    "retinamnist",
    "chestmnist",
    "midog25-atypical",
    "pcam",
    "chammi-cp-task1",
    "chammi-cp-task2",
    "chammi-hpa-task1",
    "chammi-hpa-task2",
    "lc25000",
    "rxrx1-sirna",
]
OTHER_ORDER = [
    ("regression (1-R²)", ["bbbc005", "bbbc013"]),
    ("retrieval (1-R@1)", ["nct-crc-he-100", "nct-crc-he-1k", "crc-val-he-7k"]),
]


def err1(x):
    return f"{float(x):.1f}"


def cell(full, k10, hplus_e4=False):
    if hplus_e4 and full is None and k10 is None:
        return "未训"
    f = err1(full) if full is not None else "—"
    k = err1(k10) if k10 is not None else "—"
    if f == "—" and k == "—":
        return "—"
    return f"{f} / {k}"


def delta_cell(e8f, e8k, e15f, e15k):
    def d(a, b):
        if a is None or b is None:
            return "—"
        return f"{b - a:+.1f}"

    df, dk = d(e8f, e15f), d(e8k, e15k)
    if df == "—" and dk == "—":
        return "—"
    return f"{df} / {dk}"


def dscale_run(model, pool):
    return (
        f"HS6_Dscale_{NICKDIR[model]}_robust_biosafe256_gb1024_{LR[model]}"
        f"_wu3_tw30_nosig_e8_{RAND[pool]}_seed0_20260820"
    )


def dscale_ckpt(model, pool):
    run = dscale_run(model, pool)
    ck = LAST_D[pool]
    if model == "H+":
        host = "suxin-8H100-1"
        path = f"/data_2/suxin/runs/{run}/ckpt/{ck}/checkpoint.pth"
    else:
        host = "nfs-deepcad"
        path = f"{RUNS}/{run}/ckpt/{ck}/checkpoint.pth"
    return host, path


def cscale_host_ckpt():
    hosts = {
        "S+": "5090-lyx-xr|5090-lyx-xr|5090-lyx-xr|nfs-deepcad|nfs-deepcad",
        "B": "5090-lyx-xr|5090-lyx-xr|5090-lyx-xr|nfs-deepcad|5090-hxw-xzj",
        "L": "5090-hxw-xzj|5090-hxw-xzj|5090-hxw-xzj|nfs-deepcad|nfs-deepcad",
        "H+": "suxin-8H100-1|suxin-8H100-1||suxin-8H100-1|suxin-8H100-1",
    }
    ck = {
        "S+": (
            "/data/xuzijing/biodino/outputs/01_training_runs/"
            "HS6_Cscale_Splus_robust_biosafe256_gb1024_lr2e4_prop15_nosig_e1_random100_seed0_8x5090xr_20260903/ckpt/1024/checkpoint.pth|"
            "/data/xuzijing/biodino/outputs/01_training_runs/"
            "HS6_Cscale_Splus_robust_biosafe256_gb1024_lr2e4_prop15_nosig_e2_random100_seed0_8x5090xr_20260903/ckpt/2049/checkpoint.pth|"
            "/data/xuzijing/biodino/outputs/01_training_runs/"
            "HS6_Cscale_Splus_robust_biosafe256_gb1024_lr2e4_prop15_nosig_e4_random100_seed0_8x5090xr_20260903/ckpt/4099/checkpoint.pth|"
            f"{RUNS}/HS6_Dscale_Splus_robust_biosafe256_gb1024_lr2e4_wu3_tw30_nosig_e8_random100_seed0_20260820/ckpt/8199/checkpoint.pth|"
            f"{RUNS}/HS6_Splus_robust_biosafe256_gb1024_lr2e4_wu3_tw30_nosig_e15_seed0_8x5090xr_20260821b/ckpt/15374/checkpoint.pth"
        ),
        "B": (
            "/data/xuzijing/biodino/outputs/01_training_runs/"
            "HS6_Cscale_B_robust_biosafe256_gb1024_lr1p5e4_prop15_nosig_e1_random100_seed0_8x5090xr_20260903/ckpt/1024/checkpoint.pth|"
            "/data/xuzijing/biodino/outputs/01_training_runs/"
            "HS6_Cscale_B_robust_biosafe256_gb1024_lr1p5e4_prop15_nosig_e2_random100_seed0_8x5090xr_20260903/ckpt/2049/checkpoint.pth|"
            "/data/xuzijing/biodino/outputs/01_training_runs/"
            "HS6_Cscale_B_robust_biosafe256_gb1024_lr1p5e4_prop15_nosig_e4_random100_seed0_8x5090xr_20260903/ckpt/4099/checkpoint.pth|"
            f"{RUNS}/HS6_Dscale_B_robust_biosafe256_gb1024_lr1p5e4_wu3_tw30_nosig_e8_random100_seed0_20260820/ckpt/8199/checkpoint.pth|"
            "/mnt/data/biodino_fixed_pass/outputs/01_training_runs/"
            "HS6_B_robust_biosafe256_gb1024_lr1p5e4_wu3_tw30_nosig_e15_seed0_8x5090hxw_20260818/ckpt/15374/checkpoint.pth"
        ),
        "L": (
            "/mnt/data/biodino_fixed_pass/outputs/01_training_runs/"
            "HS6_Cscale_L_robust_biosafe256_gb1024_lr1e4_prop15_nosig_e1_random100_seed0_8x5090hxw_20260903/ckpt/1024/checkpoint.pth|"
            "/mnt/data/biodino_fixed_pass/outputs/01_training_runs/"
            "HS6_Cscale_L_robust_biosafe256_gb1024_lr1e4_prop15_nosig_e2_random100_seed0_8x5090hxw_20260903/ckpt/2049/checkpoint.pth|"
            "/mnt/data/biodino_fixed_pass/outputs/01_training_runs/"
            "HS6_Cscale_L_robust_biosafe256_gb1024_lr1e4_prop15_nosig_e4_random100_seed0_8x5090hxw_20260903/ckpt/4099/checkpoint.pth|"
            f"{RUNS}/HS6_Dscale_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e8_random100_seed0_20260820/ckpt/8199/checkpoint.pth|"
            f"{RUNS}/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_seed0_20260818/ckpt/15374/checkpoint.pth"
        ),
        "H+": (
            "/data_2/suxin/runs/HS6_Cscale_Hplus_robust_biosafe256_gb1024_lr5e5_prop15_nosig_e1_random100_seed0_2xh100_20260903/ckpt/1024/checkpoint.pth|"
            "/data_2/suxin/runs/HS6_Cscale_Hplus_robust_biosafe256_gb1024_lr5e5_prop15_nosig_e2_random100_seed0_2xh100_20260903/ckpt/2049/checkpoint.pth||"
            "/data_2/suxin/runs/HS6_Dscale_Hplus_robust_biosafe256_gb1024_lr5e5_wu3_tw30_nosig_e8_random100_seed0_20260820/ckpt/8199/checkpoint.pth|"
            "/data_2/suxin/runs/HS6_Hplus_robust_biosafe256_gb1024_lr5e5_wu3_tw30_nosig_e15_seed0_4xH100_20260818/ckpt/15374/checkpoint.pth"
        ),
    }
    return hosts, ck


def read_last(path: Path):
    if not path.is_file():
        return None
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None
    if not isinstance(d, dict) or d.get("error"):
        return None
    if d.get("macro_f1") is not None:
        return 100.0 * (1.0 - float(d["macro_f1"]))
    if d.get("r2") is not None:
        return 100.0 * (1.0 - float(d["r2"]))
    if d.get("recall_at_1") is not None:
        return 100.0 * (1.0 - float(d["recall_at_1"]))
    return None


def ingest_kshot():
    k10 = defaultdict(dict)  # (model, ds, ckpt, axis) -> seed -> err
    full_from_k = {}  # (model, ds, ckpt, axis) -> full error
    files = [
        KDIR / "kshot_merged.csv",
        KDIR / "kshot_fill_20260901.csv",
        KDIR / "kshot_h100.csv",
        KDIR / "kshot_hxw_B_e15.csv",
        KDIR / "kshot_xr_L_e15.csv",
        KDIR / "kshot_xr_L_e15_official.csv",
        ROOT / "outputs/auto_eval_logs/hs6_dscale_k10_fill_20260904/kshot.csv",
    ]
    for path in files:
        if not path.is_file():
            continue
        with path.open() as handle:
            for row in csv.DictReader(handle):
                ds = row.get("dataset") or ""
                model = row.get("model") or ""
                if not ds or model not in MODELS:
                    continue
                try:
                    ckpt = int(row["ckpt"])
                except Exception:
                    continue
                axis = row.get("axis") or ""
                if str(row.get("k", "")) == "10" and row.get("error_macro_f1") not in (None, ""):
                    k10[(model, ds, ckpt, axis)][str(row.get("seed", ""))] = 100.0 * float(row["error_macro_f1"])
                if row.get("full_macro_f1") not in (None, ""):
                    full_from_k[(model, ds, ckpt, axis)] = 100.0 * (1.0 - float(row["full_macro_f1"]))
    k10_mean = {}
    for key, seeds in k10.items():
        vals = [seeds[s] for s in ("0", "1", "2") if s in seeds] or list(seeds.values())
        if vals:
            k10_mean[key] = sum(vals) / len(vals)
    return k10_mean, full_from_k


def k10_pick(k10_mean, model, ds, ckpt, axes):
    for axis in axes:
        v = k10_mean.get((model, ds, ckpt, axis))
        if v is not None:
            return v
    return None


def full_pick_k(full_from_k, model, ds, ckpt, axes):
    for axis in axes:
        v = full_from_k.get((model, ds, ckpt, axis))
        if v is not None:
            return v
    return None


def load_current_full():
    out = {}
    fam = {}
    with (DATA / "current_full_per_dataset.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row.get("metric") != "macro_f1" and row.get("family") == "classification":
                continue
            val = row.get("value")
            if val in (None, ""):
                continue
            # current_full classification stores macro_f1; other families store the family score
            score = float(val)
            err = 100.0 * (1.0 - score)
            out[(row["model"], row["pool"], row["dataset"], row["family"])] = err
            fam[row["dataset"]] = row["family"]
    return out, fam


def load_cscale_full():
    out = {}
    paths = [DATA / "cscale_full_20260903.jsonl", *Path("/tmp").glob("hs6_cscale_*.jsonl")]
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            row = json.loads(line)
            if row.get("kind") != "full":
                continue
            out[(row["nick"], int(row["ckpt"]), row["dataset"])] = 100.0 * (1.0 - float(row["macro_f1"]))
    return out


def load_last_trees():
    """e15 S+/L and D-scale e8 last full error from last_result.json."""
    e15 = {}
    e15_runs = {
        "S+": RUNS / "HS6_Splus_robust_biosafe256_gb1024_lr2e4_wu3_tw30_nosig_e15_seed0_8x5090xr_20260821b"
        / "eval/hs6_online_full_20260824",
        "L": RUNS / "HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_seed0_20260818"
        / "eval/hs6_online_full_20260819",
    }
    for model, ev in e15_runs.items():
        if not ev.is_dir():
            continue
        for fam in ("bio_classification", "bio_regression", "bio_retrieval"):
            famd = ev / fam
            if not famd.is_dir():
                continue
            for dsdir in famd.iterdir():
                p = dsdir / "15374" / "last_result.json"
                err = read_last(p)
                if err is not None:
                    e15[(model, dsdir.name)] = err

    d8 = {}
    for model in MODELS:
        for pool, ck in LAST_D.items():
            ev = RUNS / dscale_run(model, pool) / "eval/e8_full_20260820"
            if not ev.is_dir():
                continue
            for fam in ("bio_classification", "bio_regression", "bio_retrieval"):
                famd = ev / fam
                if not famd.is_dir():
                    continue
                for dsdir in famd.iterdir():
                    p = dsdir / str(ck) / "last_result.json"
                    err = read_last(p)
                    if err is not None:
                        d8[(model, pool, dsdir.name)] = err
    return e15, d8


def discover_datasets(k10_mean, current_full, cscale_full, e15_full, d8_full):
    ds = set()
    for (_m, name, _c, _a) in k10_mean:
        ds.add(name)
    for (_m, _p, name, _f) in current_full:
        ds.add(name)
    for (_m, _c, name) in cscale_full:
        ds.add(name)
    for (_m, name) in e15_full:
        ds.add(name)
    for (_m, _p, name) in d8_full:
        ds.add(name)
    return ds


def ordered_cls(all_ds):
    out = [d for d in CLS_ORDER if d in all_ds]
    extra = sorted(d for d in all_ds if d not in out and d not in {x for _, xs in OTHER_ORDER for x in xs})
    # keep extras that look like classification (no retrieval suffixes already listed)
    cls_extra = [d for d in extra if d not in {"bbbc005", "bbbc013", "bbbc038", "conic", "monuseg", "pannuke"}]
    return out + cls_extra


def full_d(model, pool, ds, current_full, d8_full, full_from_k):
    if (model, pool, ds) in d8_full:
        return d8_full[(model, pool, ds)]
    for fam in ("classification", "regression", "retrieval"):
        if (model, pool, ds, fam) in current_full:
            return current_full[(model, pool, ds, fam)]
    return full_pick_k(full_from_k, model, ds, LAST_D[pool], ("data",))


def full_e15(model, ds, e15_full, full_from_k):
    if (model, ds) in e15_full:
        return e15_full[(model, ds)]
    return full_pick_k(full_from_k, model, ds, 15374, ("compute",))


def full_c(model, ckpt, ds, cscale_full):
    return cscale_full.get((model, ckpt, ds))


def write_tables():
    k10_mean, full_from_k = ingest_kshot()
    current_full, _fam = load_current_full()
    cscale_full = load_cscale_full()
    e15_full, d8_full = load_last_trees()
    all_ds = discover_datasets(k10_mean, current_full, cscale_full, e15_full, d8_full)
    cls_ds = ordered_cls(all_ds)
    hosts_c, ck_c = cscale_host_ckpt()

    def k10_d(model, pool, ds):
        return k10_pick(k10_mean, model, ds, LAST_D[pool], ("data", ""))

    def k10_e15(model, ds):
        return k10_pick(k10_mean, model, ds, 15374, ("compute", ""))

    def k10_c(model, ckpt, ds):
        if ds not in CSCALE_K10_DS:
            return None
        # prop15 C-scale 10-shot lives in the already-curated 3-dataset table / recent workers;
        # fall back to any axis only for those three.
        return k10_pick(k10_mean, model, ds, ckpt, ("compute", "data", ""))

    # Prefer curated 3-dataset C k10 from the existing wide table for the three
    curated = {}
    old = DATA / "cscale_1m_compute_lattice_20260904.csv"
    # don't parse old (mixed). Use kshot for C-scale 3ds from recent worker if present.
    # The existing numbers for e1/e2/e4 k10 came from prop15 eval workers, stored in
    # plot/fig2/data via the previous lattice. Re-read from /tmp kshot if needed — already
    # in kshot? C-scale k10 is NOT in kshot_merged (that's e15). Keep hardcoded curated
    # from the current C file for the 3 datasets only.
    curated_c = {
        ("S+", "bloodmnist"): {"e1": (10.67, 35.82), "e2": (10.60, 35.15), "e4": (8.13, 33.09)},
        ("B", "bloodmnist"): {"e1": (9.46, 32.28), "e2": (8.86, 31.59), "e4": (7.70, 30.70)},
        ("L", "bloodmnist"): {"e1": (5.79, 24.52), "e2": (5.82, 22.97), "e4": (6.56, 25.28)},
        ("H+", "bloodmnist"): {"e1": (4.89, 23.10), "e2": (3.95, 22.66)},
        ("S+", "tissuemnist"): {"e1": (49.76, 72.80), "e2": (49.29, 72.02), "e4": (47.87, 70.84)},
        ("B", "tissuemnist"): {"e1": (50.73, 73.80), "e2": (48.59, 71.37), "e4": (47.60, 70.32)},
        ("L", "tissuemnist"): {"e1": (48.05, 70.96), "e2": (48.29, 70.20), "e4": (47.18, 69.85)},
        ("H+", "tissuemnist"): {"e1": (46.75, 70.58), "e2": (45.28, 69.11)},
        ("S+", "cyclops-protein-loc"): {"e1": (47.68, 75.51), "e2": (46.23, 73.48), "e4": (44.13, 70.58)},
        ("B", "cyclops-protein-loc"): {"e1": (48.20, 72.51), "e2": (46.62, 70.59), "e4": (45.29, 67.15)},
        ("L", "cyclops-protein-loc"): {"e1": (45.18, 72.93), "e2": (44.02, 70.91), "e4": (42.69, 70.18)},
        ("H+", "cyclops-protein-loc"): {"e1": (45.30, 72.47), "e2": (43.62, 69.91)},
    }

    def c_pair(model, ds, tag):
        if (model, ds) in curated_c and tag in curated_c[(model, ds)]:
            return curated_c[(model, ds)][tag]
        ck = CKPT_C[tag]
        return full_c(model, ck, ds, cscale_full), None

    sections = [("classification (1-macro-F1)", cls_ds)] + [
        (title, [d for d in dss if d in all_ds]) for title, dss in OTHER_ORDER
    ]

    # ---- C ----
    c_lines = [
        "# 1M compute lattice，全部已评 dataset。格子 = full_error% / 10shot_error%。",
        "# e1/e2/e4 = C-scale prop15 last（仅 10 题有 full；10-shot 仅 blood/tissue/cyclops）。",
        "# e8 = D-scale 1M last；e15 = e15 last。H+ e4 未训。Δ 见 e8_vs_e15 表。",
        "# host/checkpoint_abs 按 e1|e2|e4|e8|e15 对齐。",
    ]
    cmp_lines = [
        "# e8 (D-scale 1M last) vs e15 last。格子 = full / 10-shot。Δ = e15 − e8（负值 = e15 更好）。",
        "# host/checkpoint_abs 按 e8|e15 对齐。",
    ]
    d_lines = [
        "# D-scale e8 last，全部已评 dataset。格子 = full / 10-shot。不混 e15 / C-scale。",
        "# host/checkpoint_abs 按 0.1M|0.2M|0.5M|1M 对齐。",
    ]
    n_lines = [
        "# N 轴 = 模型。行 = D 档（D-scale e8 last）。格子 = full / 10-shot。",
        "# host/checkpoint_abs 按 S+|B|L|H+ 对齐。",
    ]

    n_have = 0
    for title, dss in sections:
        if not dss:
            continue
        for ds in dss:
            # skip if completely empty
            any_num = False
            for m in MODELS:
                if full_d(m, "1M", ds, current_full, d8_full, full_from_k) is not None:
                    any_num = True
                if full_e15(m, ds, e15_full, full_from_k) is not None:
                    any_num = True
                if full_c(m, 1024, ds, cscale_full) is not None:
                    any_num = True
                if k10_d(m, "1M", ds) is not None or k10_e15(m, ds) is not None:
                    any_num = True
            if not any_num:
                continue
            n_have += 1
            hdr_note = f"# {ds}  [{title}]  error% (full / 10-shot)"
            c_lines += ["", hdr_note, "连线,e1,e2,e4,e8,e15,host,checkpoint_abs"]
            cmp_lines += ["", hdr_note, "连线,e8,e15,Δ(e15-e8),host,checkpoint_abs"]
            d_lines += ["", hdr_note, "连线,0.1M,0.2M,0.5M,1M,host,checkpoint_abs"]
            n_lines += ["", hdr_note, "连线,S+,B,L,H+,host,checkpoint_abs"]

            for m in MODELS:
                e1f, e1k = c_pair(m, ds, "e1")
                e2f, e2k = c_pair(m, ds, "e2")
                if m == "H+":
                    e4f, e4k = (None, None)
                    e4_txt = cell(None, None, hplus_e4=(ds in CSCALE_DS))
                else:
                    e4f, e4k = c_pair(m, ds, "e4")
                    e4_txt = cell(e4f, e4k)
                e8f = full_d(m, "1M", ds, current_full, d8_full, full_from_k)
                e8k = k10_d(m, "1M", ds)
                e15f = full_e15(m, ds, e15_full, full_from_k)
                e15k = k10_e15(m, ds)
                c_lines.append(
                    ",".join(
                        [
                            m,
                            cell(e1f, e1k),
                            cell(e2f, e2k),
                            e4_txt,
                            cell(e8f, e8k),
                            cell(e15f, e15k),
                            hosts_c[m],
                            ck_c[m],
                        ]
                    )
                )
                h8, p8 = dscale_ckpt(m, "1M")
                # e15 host is last field of C hosts
                h15 = hosts_c[m].split("|")[-1]
                p15 = ck_c[m].split("|")[-1]
                cmp_lines.append(
                    ",".join(
                        [
                            m,
                            cell(e8f, e8k),
                            cell(e15f, e15k),
                            delta_cell(e8f, e8k, e15f, e15k),
                            f"{h8}|{h15}",
                            f"{p8}|{p15}",
                        ]
                    )
                )
                dvals = []
                dhosts, dck = [], []
                for pool in POOLS:
                    dvals.append(
                        cell(
                            full_d(m, pool, ds, current_full, d8_full, full_from_k),
                            k10_d(m, pool, ds),
                        )
                    )
                    h, p = dscale_ckpt(m, pool)
                    dhosts.append(h)
                    dck.append(p)
                d_lines.append(",".join([m, *dvals, "|".join(dhosts), "|".join(dck)]))

            for pool in POOLS:
                nvals = []
                nh, nc = [], []
                for m in MODELS:
                    nvals.append(
                        cell(
                            full_d(m, pool, ds, current_full, d8_full, full_from_k),
                            k10_d(m, pool, ds),
                        )
                    )
                    h, p = dscale_ckpt(m, pool)
                    nh.append(h)
                    nc.append(p)
                n_lines.append(",".join([pool, *nvals, "|".join(nh), "|".join(nc)]))

    (DATA / "cscale_1m_compute_lattice_20260904.csv").write_text("\n".join(c_lines) + "\n")
    (DATA / "nscale_model_lattice_20260904.csv").write_text("\n".join(n_lines) + "\n")
    (DATA / "dscale_e8_data_lattice_20260904.csv").write_text("\n".join(d_lines) + "\n")
    (DATA / "e8_vs_e15_all_datasets_20260904.csv").write_text("\n".join(cmp_lines) + "\n")
    print(f"datasets written: {n_have}")
    print("cls", cls_ds)
    print("e15 last_result", len(e15_full), "d8 last_result", len(d8_full), "cscale", len(cscale_full))


if __name__ == "__main__":
    write_tables()
