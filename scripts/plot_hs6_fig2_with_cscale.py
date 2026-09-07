#!/usr/bin/env python3
"""Redraw Fig.2 D/N/C with whatever C-scale last-ckpt points exist today.

Loads the existing plot stack from scripts/__pycache__ (source py files were
deleted on NFS). Injects C-scale e1/e2/e4 last points as extra 1M runs.
Does not copy checkpoints.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/mnt/huawei_deepcad/dinov3")
PYC = ROOT / "scripts" / "__pycache__"
DATA = ROOT / "plot/fig2/data"
TOKENS = 514
GBS = 1024


def load_pyc(modname: str, pyc: Path):
    spec = importlib.util.spec_from_file_location(modname, pyc)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


p7 = load_pyc("plot_hs6_scalingvit_7pp", PYC / "plot_hs6_scalingvit_7pp.cpython-311.pyc")
ov = load_pyc("plot_hs6_scalingvit_overlay", PYC / "plot_hs6_scalingvit_overlay.cpython-311.pyc")
load_pyc("plot_hs6_fig2_clone", PYC / "plot_hs6_fig2_clone.cpython-311.pyc")
fm = load_pyc("plot_hs6_scalingvit_fig2match", PYC / "plot_hs6_scalingvit_fig2match.cpython-311.pyc")

from ppt_friendly_svg import configure_matplotlib, sanitize_svg_file

configure_matplotlib()
_save_fig2_clone = fm.save_fig2_clone


def _save_ppt_svg(stem, pack, title, note, ylabel, anchor="last"):
    path = _save_fig2_clone(stem, pack, title, note, ylabel, anchor)
    svg = Path(path).with_suffix(".svg")
    if svg.is_file():
        sanitize_svg_file(svg)
    return path


fm.save_fig2_clone = _save_ppt_svg

STEM_K10 = {
    "bbbc048-cellcycle": "bbbc048_k10_fig2",
    "bloodmnist": "bloodmnist_k10_fig2",
    "dermamnist": "dermamnist_k10_fig2",
    "nct-crc-he": "nct_k10_fig2",
    "pathmnist": "pathmnist_k10_fig2",
    "tissuemnist": "tissuemnist_k10_fig2",
    "organamnist": "organamnist_k10_fig2",
    "organcmnist": "organcmnist_k10_fig2",
    "organsmnist": "organsmnist_k10_fig2",
    "octmnist": "octmnist_k10_fig2",
    "pneumoniamnist": "pneumoniamnist_k10_fig2",
    "retinamnist": "retinamnist_k10_fig2",
    "breastmnist": "breastmnist_k10_fig2",
    "pcam": "pcam_k10_fig2",
    "chammi-allen-task1": "chammi_allen_t1_k10_fig2",
    "chammi-allen-task2": "chammi_allen_t2_k10_fig2",
    "chammi-cp-task3": "chammi_cp_t3_k10_fig2",
    "cyclops-protein-loc": "cyclops_k10_fig2",
}
STEM_FULL = {
    "nct-crc-he": "nct_fullin_fig2",
    "chammi-allen-task1": "chammi_allen_t1_fullin_fig2",
    "bbbc048-cellcycle": "bbbc048_fullin_fig2",
    "bloodmnist": "bloodmnist_fullin_fig2",
    "dermamnist": "dermamnist_fullin_fig2",
    "pathmnist": "pathmnist_fullin_fig2",
    "tissuemnist": "tissuemnist_fullin_fig2",
    "chammi-allen-task2": "chammi_allen_t2_fullin_fig2",
    "chammi-cp-task3": "chammi_cp_t3_fullin_fig2",
    "cyclops-protein-loc": "cyclops_fullin_fig2",
}


def flops(nick: str, ckpt: int) -> float:
    return 6.0 * float(p7.PARAMS[nick]) * (int(ckpt) * GBS) * TOKENS


def load_full_points() -> dict[tuple[str, str, int], float]:
    out = {}
    for path in sorted(DATA.glob("cscale_full_*.jsonl")) + sorted(Path("/tmp").glob("hs6_cscale_*.jsonl")):
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            row = json.loads(line)
            if row.get("kind") != "full":
                continue
            err = 100.0 * (1.0 - float(row["macro_f1"]))
            out[(row["nick"], row["dataset"], int(row["ckpt"]))] = err
    return out


def load_k10_points() -> dict[tuple[str, str, int], float]:
    need = {("10", "0"), ("10", "1"), ("10", "2")}
    have: dict[tuple[str, str, int], dict[tuple[str, str], float]] = defaultdict(dict)
    files = [
        Path("/tmp/kshot_S.csv"),
        Path("/tmp/kshot_B.csv"),
        Path("/tmp/kshot_L.csv"),
        Path("/tmp/kshot_H.csv"),
        DATA / "cscale_kshot.csv",
    ]
    for path in files:
        if not path.is_file() or path.stat().st_size < 20:
            continue
        with path.open() as handle:
            for row in csv.DictReader(handle):
                nick = row.get("model") or row.get("nick")
                ds = row.get("dataset")
                ckpt = row.get("ckpt")
                k = str(row.get("k", ""))
                seed = str(row.get("seed", ""))
                if not nick or not ds or not ckpt:
                    continue
                if "error_macro_f1" in row and row["error_macro_f1"] not in ("", None):
                    err = 100.0 * float(row["error_macro_f1"])
                else:
                    err = 100.0 * (1.0 - float(row["macro_f1"]))
                have[(nick, ds, int(ckpt))][(k, seed)] = err
    out = {}
    for key, got in have.items():
        if need <= set(got):
            out[key] = sum(got[s] for s in need) / 3.0
    return out


def inject(pack: dict, dataset: str, points: dict[tuple[str, str, int], float]) -> dict:
    runs = dict(pack.get("runs") or {})
    added = 0
    for (nick, ds, ckpt), err in points.items():
        if ds != dataset or nick not in p7.PARAMS:
            continue
        epochs = {1024: 1, 2049: 2, 4099: 4}.get(ckpt)
        if epochs is None:
            continue
        key = (nick, "1M", f"cscale-e{epochs}")
        runs[key] = [(flops(nick, ckpt), float(err))]
        added += 1
    pack = dict(pack)
    pack["runs"] = runs
    # N = best last-of-run on C (annealed), including new duration points.
    best: dict[str, float] = {}
    for (nick, _pool, _tag), pts in runs.items():
        if not pts:
            continue
        last_err = pts[-1][1]
        best[nick] = last_err if nick not in best else min(best[nick], last_err)
    n = dict(pack.get("n") or {})
    for nick, err in best.items():
        sample = n.get(nick) or pack.get("n", {}).get(nick)
        if sample:
            x0 = sample[0][0] if isinstance(sample, list) and sample else p7.PARAMS[nick]
        else:
            x0 = p7.PARAMS[nick]
        n[nick] = [(x0, err)]
    pack["n"] = n
    pack["_cscale_added"] = added
    return pack


def note_for(kind: str, added: int) -> str:
    return (
        "Color = model (blue S+ → red H+), marker area = unique-D, circles, "
        "as in Scaling-ViT Fig.2. C solid = annealed last ckpt of each run, "
        f"plus C-scale cosine e1/e2/e4 last where json exists today ({kind} +{added} pts). "
        "D = D-scale 8-epoch best. N = C-best among those last points."
    )


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    full_pts = load_full_points()
    k10_pts = load_k10_points()
    print(
        f"cscale full cells={len(full_pts)} k10 complete cells={len(k10_pts)}",
        flush=True,
    )
    c_mean, d_mean, d_cloud = ov.load_k10()
    ylabel_k10 = "10-shot error (%)"
    ylabel_full = "linear error (%)"
    written = []
    for ds in ov.K10_ORDER:
        pack = ov.pack_k10(ds, c_mean, d_mean)
        pack = fm.attach_runs_k10(ds, pack, c_mean, d_cloud)
        pack = inject(pack, ds, k10_pts)
        stem = STEM_K10.get(ds)
        if not stem:
            continue
        title = ov.LABEL.get(ds, ds)
        path = fm.save_fig2_clone(
            stem, pack, title, note_for("k10", pack["_cscale_added"]), ylabel_k10, anchor="last"
        )
        written.append(path)
        print(f"k10 {ds} +{pack['_cscale_added']} -> {path}", flush=True)
    for ds in ov.FULL_LINEAR:
        pack = ov.pack_full_linear(ds)
        pack = fm.attach_runs_full(ds, pack)
        pack = inject(pack, ds, full_pts)
        stem = STEM_FULL.get(ds)
        if not stem:
            continue
        title = ov.LABEL.get(ds, ds)
        path = fm.save_fig2_clone(
            stem, pack, title, note_for("full", pack["_cscale_added"]), ylabel_full, anchor="last"
        )
        written.append(path)
        print(f"full {ds} +{pack['_cscale_added']} -> {path}", flush=True)
    print(f"wrote {len(written)} figures under {fm.OUT}", flush=True)


if __name__ == "__main__":
    main()
