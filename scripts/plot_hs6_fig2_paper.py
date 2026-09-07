#!/usr/bin/env python3
"""HS6 Fig.2 from plot/fig2/FIG2_PLOT_SPEC.md.

Two 1M-duration line variants (same points, same E fit):
  lineA: 1M chain includes D-scale e8 last
  lineB: 1M chain is C-scale e1/e2/e4 + e15 only
"""
from __future__ import annotations

import csv
import importlib.machinery
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np

from ppt_friendly_svg import configure_matplotlib, sanitize_svg_file

ROOT = Path("/mnt/huawei_deepcad/dinov3")
PYC = ROOT / "scripts" / "__pycache__"
DATA = ROOT / "plot/fig2/data"
OUT = ROOT / "plot/fig2/scalingvit_fig2match"
TOKENS = 514
GBS = 1024
C0 = 1e18
_SUP = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
INK, GRID, FAINT = "#2C3338", "#E4E8EB", "#D5DBE5"
COLOR = {"S+": "#465ECF", "B": "#88ABFD", "L": "#E97A5F", "H+": "#B40426"}
NICKS = ("S+", "B", "L", "H+")
POOLS = ("0.1M", "0.2M", "0.5M", "1M")
POOL_AREA = {"0.1M": 26.0, "0.2M": 42.0, "0.5M": 68.0, "1M": 104.0}
PARAMS = {"S+": 21e6, "B": 86e6, "L": 300e6, "H+": 840e6}
CKPT_EPOCH = {1024: 1, 2049: 2, 4099: 4}
CHAIN_A = ("cscale-e1", "cscale-e2", "cscale-e4", "e8", "e15")
CHAIN_B = ("cscale-e1", "cscale-e2", "cscale-e4", "e15")
STEM = {
    "bbbc048-cellcycle": "bbbc048",
    "bloodmnist": "bloodmnist",
    "dermamnist": "dermamnist",
    "nct-crc-he": "nct",
    "pathmnist": "pathmnist",
    "tissuemnist": "tissuemnist",
    "chammi-allen-task1": "chammi_allen_t1",
    "chammi-allen-task2": "chammi_allen_t2",
    "chammi-cp-task3": "chammi_cp_t3",
    "cyclops-protein-loc": "cyclops",
}


def load_pyc(name: str, pyc: Path):
    loader = importlib.machinery.SourcelessFileLoader(name, str(pyc))
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


p7 = load_pyc("plot_hs6_scalingvit_7pp", PYC / "plot_hs6_scalingvit_7pp.cpython-311.pyc")
ov = load_pyc("plot_hs6_scalingvit_overlay", PYC / "plot_hs6_scalingvit_overlay.cpython-311.pyc")
load_pyc("plot_hs6_fig2_clone", PYC / "plot_hs6_fig2_clone.cpython-311.pyc")
fm = load_pyc("plot_hs6_scalingvit_fig2match", PYC / "plot_hs6_scalingvit_fig2match.cpython-311.pyc")


def flops(nick: str, ckpt: int) -> float:
    return 6.0 * float(PARAMS[nick]) * (int(ckpt) * GBS) * TOKENS


def _to_sup(text: str) -> str:
    out = []
    for ch in str(text).replace("−", "-"):
        out.append("·" if ch == "." else ch.translate(_SUP))
    return "".join(out)


def _pow10_tick(x, _pos=None) -> str:
    if x is None or not np.isfinite(x) or x <= 0:
        return ""
    exp = int(round(np.log10(x)))
    if abs(np.log10(x) - exp) > 1e-6:
        return f"{x:g}"
    return "10" + _to_sup(exp)


def _plain_e_fit(a: float, b: float, c: float, d: float) -> str:
    return f"E={c:.2f}+{a:.2f}(C/10{_to_sup(18)}+{d:.2f}){_to_sup(f'-{b:.2f}')}"


def last_of_run(pts) -> tuple[float, float] | None:
    if not pts:
        return None
    return float(pts[-1][0]), float(pts[-1][1])


def annealed_rows(pack: dict) -> list[tuple[str, str, str, float, float]]:
    rows = []
    for (nick, pool, tag), pts in (pack.get("runs") or {}).items():
        last = last_of_run(pts)
        if last is None:
            continue
        rows.append((nick, pool, tag, last[0], last[1]))
    return rows


def pareto_front(cs: np.ndarray, es: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(cs)
    best = np.inf
    px, py = [], []
    for i in order:
        if es[i] < best - 1e-12:
            best = float(es[i])
            px.append(float(cs[i]))
            py.append(float(es[i]))
    return np.asarray(px), np.asarray(py)


def power_E(C, a, b, c, d):
    return c + a * np.power(np.asarray(C, float) / C0 + d, -b)


def fit_saturating_power_law(cs, es):
    """FIG2_PLOT_SPEC §3.4. Returns (a,b,c,d) or None."""
    cs = np.asarray(cs, float)
    es = np.asarray(es, float)
    px, py = pareto_front(cs, es)
    if len(px) < 4:
        return None
    e_min = float(py.min())
    c_grid = np.linspace(0.0, 0.99 * e_min, 25)
    d_grid = np.concatenate([[0.0], np.logspace(-3, 1, 20)])
    best = None
    for c in c_grid:
        y = py - c
        if np.any(y <= 1e-9):
            continue
        logy = np.log(y)
        for d in d_grid:
            x = px / C0 + d
            if np.any(x <= 1e-18):
                continue
            logx = np.log(x)
            b, loga = np.polyfit(logx, logy, 1)
            b = -float(b)
            a = float(np.exp(loga))
            if a <= 0 or b <= 0:
                continue
            pred = power_E(px, a, b, c, d)
            mse = float(np.mean((pred - py) ** 2))
            cand = (mse, b, a, c, d)
            if best is None or cand < best:
                best = cand
    if best is None:
        return None
    _mse, b, a, c, d = best
    return a, b, c, d


def load_cscale_full() -> dict[tuple[str, str, int], float]:
    out = {}
    for path in [DATA / "cscale_full_20260903.jsonl", *Path("/tmp").glob("hs6_cscale_*.jsonl")]:
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            row = json.loads(line)
            if row.get("kind") != "full":
                continue
            out[(row["nick"], row["dataset"], int(row["ckpt"]))] = 100.0 * (1.0 - float(row["macro_f1"]))
    return out


def _k10_from_csv(path: Path, have: dict) -> None:
    if not path.is_file() or path.stat().st_size < 20:
        return
    with path.open() as handle:
        for row in csv.DictReader(handle):
            nick, ds, ckpt = row.get("model"), row.get("dataset"), row.get("ckpt")
            if not nick or not ds or not ckpt:
                continue
            if str(row.get("k", "")) != "10":
                continue
            seed = str(row.get("seed", ""))
            if row.get("error_balanced") not in (None, ""):
                err = 100.0 * float(row["error_balanced"])
            elif row.get("error_macro_f1") not in (None, ""):
                err = 100.0 * float(row["error_macro_f1"])
            else:
                err = 100.0 * (1.0 - float(row["macro_f1"]))
            have[(nick, ds, int(ckpt))][seed] = err


def load_k10_cscale() -> dict[tuple[str, str, int], float]:
    have: dict[tuple[str, str, int], dict[str, float]] = defaultdict(dict)
    for path in (
        Path("/tmp/kshot_S.csv"),
        Path("/tmp/kshot_B.csv"),
        Path("/tmp/kshot_L.csv"),
        Path("/tmp/kshot_H.csv"),
        ROOT / "plot/fig2/kshot/kshot_merged.csv",
    ):
        _k10_from_csv(path, have)
    out = {}
    for key, seeds in have.items():
        vals = [seeds[s] for s in ("0", "1", "2") if s in seeds]
        if len(vals) >= 1:
            out[key] = float(np.mean(vals))
    return out


def inject_cscale(pack: dict, dataset: str, points: dict[tuple[str, str, int], float]) -> dict:
    runs = dict(pack.get("runs") or {})
    for (nick, ds, ckpt), err in points.items():
        if ds != dataset or nick not in PARAMS or ckpt not in CKPT_EPOCH:
            continue
        epochs = CKPT_EPOCH[ckpt]
        runs[(nick, "1M", f"cscale-e{epochs}")] = [(flops(nick, ckpt), float(err))]
    pack = dict(pack)
    pack["runs"] = runs
    return pack


def style(ax) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK, labelsize=8.5)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def tight_ylim(ax, ys: list[float], pad: float = 0.10) -> None:
    ys = [y for y in ys if y is not None and np.isfinite(y)]
    if not ys:
        return
    lo, hi = min(ys), max(ys)
    span = max(hi - lo, 1.0)
    ax.set_ylim(lo - pad * span, hi + pad * span)


def filter_rows(rows, pools: set[str] | None = None, drop_tags: set[str] | None = None):
    out = []
    for row in rows:
        nick, pool, tag, x, y = row
        if pools is not None and pool not in pools:
            continue
        if drop_tags and tag in drop_tags:
            continue
        out.append(row)
    return out


def draw_c(ax, rows, ylabel: str, title: str, chain_tags: tuple[str, ...], fit_rows=None) -> None:
    style(ax)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_pow10_tick))
    ax.set_xlabel("pretrain FLOPs", fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=6)
    chain: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    ys = []
    for nick, pool, tag, x, y in rows:
        ys.append(y)
        ax.scatter(
            [x],
            [y],
            s=POOL_AREA[pool],
            c=COLOR[nick],
            marker="o",
            zorder=4,
            edgecolors="white",
            linewidths=0.45,
        )
        if pool == "1M" and tag in chain_tags:
            chain[nick][tag] = (x, y)
    order = list(chain_tags)
    for nick, by_tag in chain.items():
        for left, right in zip(order, order[1:]):
            if left not in by_tag or right not in by_tag:
                continue
            x0, y0 = by_tag[left]
            x1, y1 = by_tag[right]
            ax.plot(
                [x0, x1],
                [y0, y1],
                color=COLOR[nick],
                lw=0.9,
                alpha=0.28,
                zorder=2,
                solid_capstyle="round",
            )
    fit_src = list(fit_rows) if fit_rows is not None else list(rows)
    if len(fit_src) >= 4:
        xs = np.array([r[3] for r in fit_src], dtype=float)
        es = np.array([r[4] for r in fit_src], dtype=float)
        fit = fit_saturating_power_law(xs, es)
        if fit is not None:
            a, b, c, d = fit
            grid = np.logspace(np.log10(xs.min() * 0.85), np.log10(xs.max() * 1.15), 200)
            ax.plot(grid, power_E(grid, a, b, c, d), color=INK, ls="--", lw=1.65, zorder=3)
            ax.axhline(c, color=INK, lw=0.6, alpha=0.45, zorder=1)
            ax.text(
                0.03,
                0.04,
                _plain_e_fit(a, b, c, d),
                transform=ax.transAxes,
                fontsize=7.5,
                color=INK,
            )
    tight_ylim(ax, ys)


def draw_n(ax, cloud: list[tuple[str, str, float]], best: dict[str, tuple[str, float]]) -> None:
    style(ax)
    ax.set_title("N", loc="left", fontsize=11, color=INK, pad=6)
    ax.set_xlabel("model size", fontsize=9, color=INK)
    ax.set_ylabel("10-shot error [%]", fontsize=9, color=INK)
    ax.set_xscale("log")
    ys = []
    for nick, pool, y in cloud:
        ys.append(y)
        ax.scatter(
            [PARAMS[nick]],
            [y],
            s=POOL_AREA[pool],
            c=FAINT,
            marker="o",
            zorder=2,
            linewidths=0,
            alpha=0.55,
        )
    xs, bv = [], []
    for nick in NICKS:
        if nick not in best:
            continue
        pool, y = best[nick]
        ys.append(y)
        ax.scatter(
            [PARAMS[nick]],
            [y],
            s=POOL_AREA[pool],
            c=COLOR[nick],
            marker="o",
            zorder=4,
            edgecolors="white",
            linewidths=0.5,
        )
        xs.append(PARAMS[nick])
        bv.append(y)
    if len(xs) >= 2:
        ax.plot(xs, bv, color=INK, ls="--", lw=1.0, zorder=3)
    ax.set_xticks([PARAMS[n] for n in NICKS])
    ax.set_xticklabels(list(NICKS))
    tight_ylim(ax, ys)


def draw_d(ax, cloud: list[tuple[str, str, float]], best: dict[str, tuple[str, float]]) -> None:
    style(ax)
    ax.set_title("D", loc="left", fontsize=11, color=INK, pad=6)
    ax.set_xlabel("unique images", fontsize=9, color=INK)
    ax.set_ylabel("10-shot error [%]", fontsize=9, color=INK)
    ax.set_xscale("log")
    ys = []
    for nick, pool, y in cloud:
        ys.append(y)
        ax.scatter(
            [p7.POOL_N[pool]],
            [y],
            s=POOL_AREA[pool],
            c=FAINT,
            marker="o",
            zorder=2,
            linewidths=0,
            alpha=0.55,
        )
    xs, bv = [], []
    for pool in POOLS:
        if pool not in best:
            continue
        nick, y = best[pool]
        ys.append(y)
        ax.scatter(
            [p7.POOL_N[pool]],
            [y],
            s=POOL_AREA[pool],
            c=COLOR[nick],
            marker="o",
            zorder=4,
            edgecolors="white",
            linewidths=0.5,
        )
        xs.append(p7.POOL_N[pool])
        bv.append(y)
    if len(xs) >= 2:
        ax.plot(xs, bv, color=INK, ls="--", lw=1.0, zorder=3)
    ax.set_xticks([p7.POOL_N[p] for p in POOLS])
    ax.set_xticklabels(list(POOLS))
    tight_ylim(ax, ys)


def split_nd(dataset: str, k10: dict, d_cloud: dict):
    """N grey = all 10-shot ckpts; D grey = D-scale 8-epoch only."""
    n_cloud = []
    n_best: dict[str, tuple[str, float]] = {}
    for (nick, pool, tag), pts in (k10.get("runs") or {}).items():
        for _x, y in pts:
            y = float(y)
            n_cloud.append((nick, pool, y))
            prev = n_best.get(nick)
            if prev is None or y < prev[1]:
                n_best[nick] = (pool, y)
    d_cloud_nd = []
    d_best: dict[str, tuple[str, float]] = {}
    for nick in NICKS:
        rows = d_cloud.get((dataset, nick)) or []
        for pool, _ckpt, err in rows:
            y = float(err)
            d_cloud_nd.append((nick, pool, y))
            prev = d_best.get(pool)
            if prev is None or y < prev[1]:
                d_best[pool] = (nick, y)
            prevn = n_best.get(nick)
            if prevn is None or y < prevn[1]:
                n_best[nick] = (pool, y)
            n_cloud.append((nick, pool, y))
    return n_cloud, n_best, d_cloud_nd, d_best


def legends(fig) -> None:
    models = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLOR[n],
            markeredgecolor="white",
            markersize=8,
            label=n,
        )
        for n in NICKS
    ]
    sizes = []
    for pool, area in POOL_AREA.items():
        sizes.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#8A9096",
                markeredgecolor="white",
                markersize=max(4.5, np.sqrt(area) * 0.42),
                label=pool,
            )
        )
    fig.legend(handles=models, loc="upper left", bbox_to_anchor=(0.055, 0.99), ncol=4, frameon=False, fontsize=9)
    fig.legend(handles=sizes, loc="upper left", bbox_to_anchor=(0.40, 0.99), ncol=4, frameon=False, fontsize=8.5)


def save_one(
    stem: str,
    title: str,
    note: str,
    full,
    k10,
    n_pack,
    d_pack,
    chain,
    suffix: str,
    c_pools: set[str] | None = None,
    drop_fit_tags: set[str] | None = None,
) -> Path:
    configure_matplotlib()
    plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(15.4, 6.45))
    gs = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1.18, 1.18, 0.82],
        height_ratios=[1, 1],
        wspace=0.30,
        hspace=0.42,
        left=0.055,
        right=0.985,
        top=0.82,
        bottom=0.11,
    )
    ax_full = fig.add_subplot(gs[:, 0])
    ax_k10 = fig.add_subplot(gs[:, 1])
    ax_n = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 2])
    full_rows = filter_rows(annealed_rows(full), c_pools)
    k10_rows = filter_rows(annealed_rows(k10), c_pools)
    full_fit = filter_rows(full_rows, drop_tags=drop_fit_tags)
    k10_fit = filter_rows(k10_rows, drop_tags=drop_fit_tags)
    draw_c(ax_full, full_rows, "linear error [%]", "C   full linear", chain, fit_rows=full_fit)
    draw_c(ax_k10, k10_rows, "10-shot error [%]", "C   10-shot", chain, fit_rows=k10_fit)
    draw_n(ax_n, n_pack[0], n_pack[1])
    draw_d(ax_d, d_pack[0], d_pack[1])
    fig.suptitle(title, fontsize=13, color=INK, x=0.055, ha="left", y=0.995)
    fig.text(0.055, 0.895, note, fontsize=7.4, color="#5B656C", ha="left", va="top")
    legends(fig)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{stem}_fig2_{suffix}.png"
    fig.savefig(path, dpi=170)
    svg = path.with_suffix(".svg")
    fig.savefig(svg)
    sanitize_svg_file(svg)
    plt.close(fig)
    return path


def main() -> None:
    full_pts = load_cscale_full()
    k10_pts = load_k10_cscale()
    c_mean, d_mean, d_cloud = ov.load_k10()
    print(f"cscale full={len(full_pts)} k10={len(k10_pts)}", flush=True)
    note_a = (
        "lineA: 1M duration chain = C-scale e1/e2/e4 + D-scale e8 + e15. "
        "Marker area = unique-D. C = annealed last only. Black dashed = saturating power-law on Pareto. "
        "N/D = 10-shot; grey = all ckpts / D-scale 8-epoch."
    )
    note_b = (
        "lineB: 1M duration chain = C-scale e1/e2/e4 + e15 only; D-scale 1M e8 is the same-size solid point, not on the line. "
        "Otherwise identical to lineA (same points, same E fit)."
    )
    note_1m = (
        "MAIN C: unique-D fixed at 1M (only duration and model size vary). "
        "lineB chain e1→e2→e4→e15. Old C-scale e2 (50% warmup) is plotted but excluded from the Pareto fit. "
        "This is the Chinchilla Approach-1 compute law; allD is the appendix."
    )
    for ds, stem in STEM.items():
        full = inject_cscale(fm.attach_runs_full(ds, ov.pack_full_linear(ds)), ds, full_pts)
        k10 = inject_cscale(fm.attach_runs_k10(ds, ov.pack_k10(ds, c_mean, d_mean), c_mean, d_cloud), ds, k10_pts)
        n_cloud, n_best, d_cloud_nd, d_best = split_nd(ds, k10, d_cloud)
        title = ov.LABEL.get(ds, ds)
        packs = ((n_cloud, n_best), (d_cloud_nd, d_best))
        pa = save_one(stem, title, note_a, full, k10, packs[0], packs[1], CHAIN_A, "lineA")
        pb = save_one(stem, title, note_b, full, k10, packs[0], packs[1], CHAIN_B, "lineB")
        p1 = save_one(
            stem,
            title,
            note_1m,
            full,
            k10,
            packs[0],
            packs[1],
            CHAIN_B,
            "lineB_1Monly",
            c_pools={"1M"},
            drop_fit_tags={"cscale-e2"},
        )
        print(
            f"{ds}  C-full {len(annealed_rows(full))}  C-k10 {len(annealed_rows(k10))}  -> {pa.name} / {pb.name} / {p1.name}",
            flush=True,
        )


if __name__ == "__main__":
    main()
