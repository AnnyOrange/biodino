#!/usr/bin/env python3
"""Build a draft Fig. 3 representation figure for the Biodino H+ checkpoint.

The script creates deterministic test-only masks for Panel A/B, extracts frozen
features with the same DINOv3 eval encoder used by the benchmark harness, then
plots:
  A. A biological embedding landscape colored by modality and biology identity.
  B. Cross-modal biological correspondence using paired histology samples.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image, ImageFile, ImageOps

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BENCHMARK_ROOT = Path("/mnt/huawei_deepcad/benchmark")
DEFAULT_OUT_DIR = REPO_ROOT / "outputs/04_figures/fig3_representation_20260812"
DEFAULT_CKPT = (
    REPO_ROOT
    / "outputs/01_training_runs/hplus_s6_e15_nosigreg_alpha1_20260812/ckpt/100/checkpoint.pth"
)
DEFAULT_CONFIG = REPO_ROOT / "outputs/01_training_runs/hplus_s6_e15_nosigreg_alpha1_20260812/config.yaml"

TILE_FRACTIONS = (
    (0.50, 0.50),
    (0.32, 0.34),
    (0.68, 0.34),
    (0.34, 0.68),
    (0.68, 0.68),
    (0.50, 0.27),
    (0.27, 0.50),
    (0.73, 0.50),
)

PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#8c564b",
    "#e6ab02",
    "#17becf",
    "#7f7f7f",
    "#66a61e",
    "#e7298a",
    "#a6761d",
    "#1b9e77",
]


def stable_int(text: str) -> int:
    value = 2166136261
    for ch in str(text):
        value ^= ord(ch)
        value = (value * 16777619) & 0xFFFFFFFF
    return int(value)


def compact_label(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    text = str(value).strip()
    return text if text and text.lower() != "nan" else default


def normalize_feature_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No feature rows were built.")
    df = df.drop_duplicates("sample_id", keep="first").reset_index(drop=True)
    df["feature_label"] = pd.factorize(df["biology_id"].astype(str))[0].astype(int)
    return df


def resolve_cyto_path(extracted_root: Path, row: pd.Series) -> Path:
    rel = compact_label(row.get("path")).lstrip("/")
    if rel.startswith("cytoimagenet/"):
        rel = rel.split("/", 1)[1]
    return extracted_root / rel / compact_label(row.get("filename"))


def round_robin_sample(group: pd.DataFrame, key: str, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pools: dict[str, list[int]] = {}
    for label, sub in group.groupby(key, dropna=False):
        idx = sub.index.to_numpy(copy=True)
        rng.shuffle(idx)
        pools[compact_label(label)] = idx.tolist()
    selected: list[int] = []
    labels = sorted(pools, key=lambda x: (-len(pools[x]), x))
    while len(selected) < n and labels:
        next_labels: list[str] = []
        for label in labels:
            if pools[label] and len(selected) < n:
                selected.append(pools[label].pop())
            if pools[label]:
                next_labels.append(label)
        labels = next_labels
    return group.loc[selected]


def make_cyto_panel_a_rows(root: Path, target: int, seed: int) -> list[dict[str, Any]]:
    cyto_root = root / "Representation/CytoImageNet"
    meta_path = cyto_root / "metadata/metadata.csv"
    extracted_root = cyto_root / "extracted"
    if not meta_path.exists() or not extracted_root.exists():
        raise FileNotFoundError(f"CytoImageNet metadata/extracted root missing under {cyto_root}")

    usecols = [
        "database",
        "organism",
        "cell_type",
        "microscopy",
        "path",
        "filename",
        "idx",
        "label",
        "category",
        "dset",
    ]
    df = pd.read_csv(meta_path, usecols=usecols)
    df = df[df["dset"].astype(str).str.lower().eq("val")].copy()
    for col in ("organism", "cell_type", "microscopy"):
        df[col] = df[col].map(compact_label)
    df = df[
        df["organism"].ne("unknown")
        & df["cell_type"].ne("unknown")
        & df["microscopy"].ne("unknown")
    ].copy()
    df["image_path"] = [str(resolve_cyto_path(extracted_root, row)) for _, row in df.iterrows()]
    df = df[df["image_path"].map(lambda x: Path(x).exists())].copy()
    if df.empty:
        raise RuntimeError("CytoImageNet validation mask is empty after path checks.")

    modality_counts = df["microscopy"].value_counts()
    modalities = modality_counts[modality_counts >= 20].index.tolist()
    if not modalities:
        modalities = modality_counts.index.tolist()
    per_modality = max(1, math.ceil(target / max(1, len(modalities))))

    sampled: list[pd.DataFrame] = []
    for i, modality in enumerate(modalities):
        sub = df[df["microscopy"].eq(modality)]
        take = min(per_modality, len(sub))
        sampled.append(round_robin_sample(sub, "cell_type", take, seed + i))
    sample_df = pd.concat(sampled, axis=0)
    if len(sample_df) > target:
        sample_df = round_robin_sample(sample_df, "microscopy", target, seed + 97)
    sample_df = sample_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for _, row in sample_df.iterrows():
        cell_type = compact_label(row["cell_type"])
        modality = compact_label(row["microscopy"])
        rows.append(
            {
                "sample_id": f"cyto:{compact_label(row['idx'])}",
                "dataset": "CytoImageNet",
                "split": "val",
                "sample_type": "image",
                "image_path": row["image_path"],
                "tile_idx": "",
                "tile_page": "",
                "tile_frac_x": "",
                "tile_frac_y": "",
                "entity_id": compact_label(row["idx"]),
                "biology_id": cell_type,
                "biology_group": "cell_type",
                "modality": modality,
                "organism": compact_label(row["organism"]),
                "tissue": "",
                "cell_type": cell_type,
                "stain": "",
                "source_label": compact_label(row["label"]),
                "source_category": compact_label(row["category"]),
                "source_database": compact_label(row["database"]),
            }
        )
    return rows


def parse_acrobat_name(path: Path) -> tuple[str, str]:
    stem = path.stem
    if not stem.endswith("_val"):
        raise ValueError(f"Unexpected ACROBAT name: {path.name}")
    body = stem[: -len("_val")]
    patient, stain = body.rsplit("_", 1)
    return patient, stain


def choose_acrobat_page(path: Path, min_size: int = 512) -> int:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        chosen = len(tif.pages) - 1
        for i, page in enumerate(tif.pages):
            shape = page.shape
            h, w = int(shape[0]), int(shape[1])
            if min(h, w) >= min_size:
                chosen = i
        return chosen


def list_acrobat_slides(root: Path) -> dict[str, dict[str, Path]]:
    valid_root = root / "Representation/ACROBAT/extracted/valid"
    if not valid_root.exists():
        return {}
    slides: dict[str, dict[str, Path]] = defaultdict(dict)
    for path in sorted(valid_root.glob("*.tif")):
        try:
            patient, stain = parse_acrobat_name(path)
        except ValueError:
            continue
        slides[patient][stain] = path
    return slides


def make_acrobat_tile_rows(
    root: Path,
    tiles_per_patient: int,
    max_patients: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    slides = list_acrobat_slides(root)
    patients = sorted([p for p, stains in slides.items() if "HE" in stains and len(stains) >= 2], key=lambda x: int(x))
    if max_patients > 0:
        patients = patients[:max_patients]

    sample_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    patient_stain_by_key: dict[tuple[str, str, int], str] = {}
    for patient in patients:
        stains = ["HE"] + sorted(s for s in slides[patient] if s != "HE")
        for stain in stains:
            path = slides[patient][stain]
            page = choose_acrobat_page(path)
            for tile_idx in range(tiles_per_patient):
                frac_x, frac_y = TILE_FRACTIONS[tile_idx % len(TILE_FRACTIONS)]
                sample_id = f"acrobat:{patient}:{stain}:t{tile_idx}"
                patient_stain_by_key[(patient, stain, tile_idx)] = sample_id
                sample_rows.append(
                    {
                        "sample_id": sample_id,
                        "dataset": "ACROBAT",
                        "split": "valid",
                        "sample_type": "acrobat_tile",
                        "image_path": str(path),
                        "tile_idx": tile_idx,
                        "tile_page": page,
                        "tile_frac_x": frac_x,
                        "tile_frac_y": frac_y,
                        "entity_id": patient,
                        "biology_id": "breast tissue",
                        "biology_group": "patient/tissue",
                        "modality": f"histology:{stain}",
                        "organism": "human",
                        "tissue": "breast",
                        "cell_type": "",
                        "stain": stain,
                        "source_label": patient,
                        "source_category": "paired slide",
                        "source_database": "ACROBAT",
                    }
                )

    for patient in patients:
        other_stains = sorted(s for s in slides[patient] if s != "HE")
        if not other_stains:
            continue
        target_stain = other_stains[stable_int(patient) % len(other_stains)]
        for tile_idx in range(tiles_per_patient):
            he_id = patient_stain_by_key[(patient, "HE", tile_idx)]
            ihc_id = patient_stain_by_key[(patient, target_stain, tile_idx)]
            pair_rows.append(
                {
                    "pair_id": f"acrobat_pos:{patient}:{target_stain}:t{tile_idx}",
                    "source": "ACROBAT",
                    "entity_id": patient,
                    "pair_type": "same_biology_cross_modality",
                    "anchor_id": he_id,
                    "other_id": ihc_id,
                    "anchor_modality": "histology:HE",
                    "other_modality": f"histology:{target_stain}",
                    "biology_id": "breast tissue",
                    "is_positive": 1,
                }
            )

    if patients:
        for i, patient in enumerate(patients):
            next_patient = patients[(i + 1) % len(patients)]
            for tile_idx in range(tiles_per_patient):
                pair_rows.append(
                    {
                        "pair_id": f"acrobat_neg_same_mod:{patient}:HE:{next_patient}:t{tile_idx}",
                        "source": "ACROBAT",
                        "entity_id": patient,
                        "pair_type": "different_biology_same_modality",
                        "anchor_id": patient_stain_by_key[(patient, "HE", tile_idx)],
                        "other_id": patient_stain_by_key[(next_patient, "HE", tile_idx)],
                        "anchor_modality": "histology:HE",
                        "other_modality": "histology:HE",
                        "biology_id": "breast tissue",
                        "is_positive": 0,
                    }
                )
    return sample_rows, pair_rows


def parse_cima_stain(path: Path) -> str:
    stem = path.stem.lower().replace("_", "-")
    if "pro-spc" in stem or "prospc" in stem:
        return "proSPC"
    for key, label in (
        ("ki67", "Ki67"),
        ("cd31", "CD31"),
        ("cc10", "Cc10"),
        ("cneu", "CNEU"),
        ("er", "ER"),
        ("pr", "PR"),
        ("he", "He"),
    ):
        if re.search(rf"(^|-){key}($|-)", stem):
            return label
    return "unknown"


def cima_biology_from_folder(folder: str) -> tuple[str, str]:
    if folder.startswith("lung-lesion"):
        return "lung lesion", "lung"
    if folder.startswith("lung-lobes"):
        return "lung lobe", "lung"
    if folder.startswith("mammary-gland"):
        return "mammary gland", "mammary gland"
    return folder.replace("-", " "), ""


def make_cima_rows(
    root: Path,
    scale: str,
    max_panel_a: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cima_root = root / "Representation/CIMA/extracted"
    if not cima_root.exists():
        return [], [], []
    image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    paths = sorted(
        p
        for p in cima_root.glob(f"*/{scale}/*")
        if p.is_file() and p.suffix.lower() in image_exts
    )
    if not paths:
        return [], [], []

    sample_by_path: dict[Path, dict[str, Any]] = {}
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        lesion = path.parts[-3]
        biology_id, tissue = cima_biology_from_folder(lesion)
        stain = parse_cima_stain(path)
        sample_id = f"cima:{path.relative_to(cima_root).with_suffix('')}"
        row = {
            "sample_id": sample_id,
            "dataset": "CIMA",
            "split": "full",
            "sample_type": "image",
            "image_path": str(path),
            "tile_idx": "",
            "tile_page": "",
            "tile_frac_x": "",
            "tile_frac_y": "",
            "entity_id": lesion,
            "biology_id": biology_id,
            "biology_group": "tissue_structure",
            "modality": f"histology:{stain}",
            "organism": "mouse",
            "tissue": tissue,
            "cell_type": "",
            "stain": stain,
            "source_label": lesion,
            "source_category": "same lesion multi-stain",
            "source_database": "CIMA",
        }
        sample_by_path[path] = row
        groups[lesion].append(path)

    all_samples = list(sample_by_path.values())
    panel_a_rows = all_samples
    if max_panel_a > 0 and len(panel_a_rows) > max_panel_a:
        panel_a_rows = (
            pd.DataFrame(panel_a_rows)
            .sample(n=max_panel_a, random_state=seed)
            .to_dict(orient="records")
        )

    pair_rows: list[dict[str, Any]] = []
    stain_to_lesions: dict[str, list[str]] = defaultdict(list)
    for lesion, group_paths in groups.items():
        for path in group_paths:
            stain_to_lesions[parse_cima_stain(path)].append(lesion)

    for lesion, group_paths in groups.items():
        by_stain = {parse_cima_stain(p): sample_by_path[p]["sample_id"] for p in group_paths}
        if "He" not in by_stain:
            continue
        for stain, sample_id in sorted(by_stain.items()):
            if stain == "He":
                continue
            pair_rows.append(
                {
                    "pair_id": f"cima_pos:{lesion}:He:{stain}",
                    "source": "CIMA",
                    "entity_id": lesion,
                    "pair_type": "same_biology_cross_modality",
                    "anchor_id": by_stain["He"],
                    "other_id": sample_id,
                    "anchor_modality": "histology:He",
                    "other_modality": f"histology:{stain}",
                    "biology_id": "lung lesion",
                    "is_positive": 1,
                }
            )
            candidates = sorted(set(stain_to_lesions[stain]) - {lesion})
            if candidates:
                neg_lesion = candidates[stable_int(lesion + stain) % len(candidates)]
                neg_sample = next(
                    sample_by_path[p]["sample_id"]
                    for p in groups[neg_lesion]
                    if parse_cima_stain(p) == stain
                )
                pair_rows.append(
                    {
                        "pair_id": f"cima_neg_same_mod:{lesion}:{stain}:{neg_lesion}",
                        "source": "CIMA",
                        "entity_id": lesion,
                        "pair_type": "different_biology_same_modality",
                        "anchor_id": sample_id,
                        "other_id": neg_sample,
                        "anchor_modality": f"histology:{stain}",
                        "other_modality": f"histology:{stain}",
                        "biology_id": "lung lesion",
                        "is_positive": 0,
                    }
                )
    return panel_a_rows, all_samples, pair_rows


def make_cyto_panel_b_pairs(panel_a_df: pd.DataFrame, n_pairs: int, seed: int) -> list[dict[str, Any]]:
    cyto = panel_a_df[panel_a_df["dataset"].astype(str).eq("CytoImageNet")].copy()
    cyto = cyto[cyto["cell_type"].map(compact_label).ne("unknown")].copy()
    cyto = cyto[cyto["modality"].map(compact_label).ne("unknown")].copy()
    cell_mod_counts = cyto.groupby("cell_type")["modality"].nunique()
    cross_cells = set(cell_mod_counts[cell_mod_counts >= 2].index.astype(str))
    cyto = cyto[cyto["cell_type"].astype(str).isin(cross_cells)].copy().reset_index(drop=True)
    if cyto.empty:
        raise RuntimeError("No CytoImageNet cell types have at least two modalities in Panel A.")

    rng = np.random.default_rng(seed)
    by_cell_mod: dict[tuple[str, str], pd.DataFrame] = {}
    for key, group in cyto.groupby(["cell_type", "modality"]):
        by_cell_mod[(str(key[0]), str(key[1]))] = group.sample(frac=1, random_state=seed).reset_index(drop=True)

    positive_anchors = cyto.sample(frac=1, random_state=seed + 1).to_dict(orient="records")
    rows: list[dict[str, Any]] = []
    for j in range(n_pairs):
        anchor = positive_anchors[j % len(positive_anchors)]
        cell = str(anchor["cell_type"])
        modality = str(anchor["modality"])
        choices = sorted(k for k in by_cell_mod if k[0] == cell and k[1] != modality)
        if not choices:
            continue
        other_key = choices[(j + stable_int(anchor["sample_id"])) % len(choices)]
        pool = by_cell_mod[other_key]
        other = pool.iloc[(j + stable_int(cell + modality)) % len(pool)].to_dict()
        rows.append(
            {
                "pair_id": f"cyto_pos:{j:04d}:{anchor['sample_id']}::{other['sample_id']}",
                "source": "CytoImageNet",
                "entity_id": cell,
                "pair_type": "same_biology_cross_modality",
                "anchor_id": anchor["sample_id"],
                "other_id": other["sample_id"],
                "anchor_modality": modality,
                "other_modality": str(other["modality"]),
                "biology_id": cell,
                "is_positive": 1,
            }
        )

    modality_cell_counts = cyto.groupby("modality")["cell_type"].nunique()
    negative_modalities = set(modality_cell_counts[modality_cell_counts >= 2].index.astype(str))
    neg_anchors = cyto[cyto["modality"].astype(str).isin(negative_modalities)].copy()
    neg_anchors = neg_anchors.sample(frac=1, random_state=seed + 2).to_dict(orient="records")
    if not neg_anchors:
        raise RuntimeError("No CytoImageNet modality has at least two cell types for same-modality negatives.")
    by_modality = {
        str(key): group.sample(frac=1, random_state=seed + stable_int(key) % 997).reset_index(drop=True)
        for key, group in cyto.groupby("modality")
    }
    for j in range(n_pairs):
        anchor = neg_anchors[j % len(neg_anchors)]
        modality = str(anchor["modality"])
        cell = str(anchor["cell_type"])
        pool = by_modality[modality]
        pool = pool[pool["cell_type"].astype(str).ne(cell)].reset_index(drop=True)
        if pool.empty:
            continue
        other = pool.iloc[(j + stable_int(anchor["sample_id"])) % len(pool)].to_dict()
        rows.append(
            {
                "pair_id": f"cyto_neg_same_mod:{j:04d}:{anchor['sample_id']}::{other['sample_id']}",
                "source": "CytoImageNet",
                "entity_id": cell,
                "pair_type": "different_biology_same_modality",
                "anchor_id": anchor["sample_id"],
                "other_id": other["sample_id"],
                "anchor_modality": modality,
                "other_modality": str(other["modality"]),
                "biology_id": cell,
                "is_positive": 0,
            }
        )

    rng.shuffle(rows)
    return rows


def build_manifests(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    benchmark_root = Path(args.benchmark_root)
    panel_a_rows = make_cyto_panel_a_rows(
        benchmark_root,
        target=args.panel_a_cyto_target,
        seed=args.seed,
    )
    cima_panel_a, cima_samples, cima_pairs = make_cima_rows(
        benchmark_root,
        scale=args.cima_scale,
        max_panel_a=args.panel_a_cima_max,
        seed=args.seed + 11,
    )
    panel_a_rows.extend(cima_panel_a)
    panel_a_df = pd.DataFrame(panel_a_rows).drop_duplicates("sample_id", keep="first").reset_index(drop=True)

    pair_rows: list[dict[str, Any]] = []
    acrobat_samples: list[dict[str, Any]] = []
    if args.panel_b_mode in {"cyto", "cyto+histology"}:
        pair_rows.extend(make_cyto_panel_b_pairs(panel_a_df, n_pairs=args.panel_b_pairs, seed=args.seed + 23))
    if args.panel_b_mode in {"histology", "cyto+histology"}:
        acrobat_samples, acrobat_pairs = make_acrobat_tile_rows(
            benchmark_root,
            tiles_per_patient=args.acrobat_tiles_per_patient,
            max_patients=args.acrobat_max_patients,
        )
        pair_rows.extend(acrobat_pairs + cima_pairs)
    pair_df = pd.DataFrame(pair_rows)
    if pair_df.empty:
        raise RuntimeError("Panel B pair manifest is empty.")

    pair_sample_ids = set(pair_df["anchor_id"]).union(set(pair_df["other_id"]))
    sample_pool = panel_a_rows + acrobat_samples + cima_samples
    feature_rows = [row for row in sample_pool if row["sample_id"] in pair_sample_ids or row["sample_id"] in set(panel_a_df["sample_id"])]
    feature_df = normalize_feature_rows(feature_rows)

    return panel_a_df, pair_df, feature_df


def save_manifests(
    out_dir: Path,
    panel_a_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_a_df.to_csv(out_dir / "panel_a_test_mask.csv", index=False)
    pair_df.to_csv(out_dir / "panel_b_test_pairs.csv", index=False)
    feature_df.to_csv(out_dir / "feature_manifest.csv", index=False)


def array_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {3, 4}:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype == np.uint16:
        lo, hi = np.percentile(arr, [1.0, 99.5])
        arr = (np.clip((arr.astype(np.float32) - lo) / max(hi - lo, 1.0), 0, 1) * 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        finite = np.isfinite(arr)
        lo = float(np.percentile(arr[finite], 1.0)) if finite.any() else 0.0
        hi = float(np.percentile(arr[finite], 99.5)) if finite.any() else 1.0
        arr = (np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def crop_tissue_tile(arr: np.ndarray, row: dict[str, Any], crop_size: int = 384) -> Image.Image:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        gray = arr.astype(np.float32)
    else:
        gray = arr[..., :3].astype(np.float32).mean(axis=-1)
    h, w = gray.shape[:2]
    crop = min(crop_size, h, w)
    frac_x = float(row.get("tile_frac_x") or 0.5)
    frac_y = float(row.get("tile_frac_y") or 0.5)
    cx = int(np.clip(round(frac_x * w), crop // 2, max(crop // 2, w - crop // 2)))
    cy = int(np.clip(round(frac_y * h), crop // 2, max(crop // 2, h - crop // 2)))

    def crop_bounds(center: int, length: int) -> tuple[int, int]:
        lo = int(np.clip(center - crop // 2, 0, max(0, length - crop)))
        return lo, lo + crop

    y0, y1 = crop_bounds(cy, h)
    x0, x1 = crop_bounds(cx, w)
    tissue_fraction = float((gray[y0:y1, x0:x1] < 238).mean())
    if tissue_fraction < 0.03:
        mask = gray < 238
        ys, xs = np.where(mask)
        if len(xs) > 0:
            pick = stable_int(compact_label(row.get("sample_id"))) % len(xs)
            cx = int(np.clip(xs[pick], crop // 2, max(crop // 2, w - crop // 2)))
            cy = int(np.clip(ys[pick], crop // 2, max(crop // 2, h - crop // 2)))
            y0, y1 = crop_bounds(cy, h)
            x0, x1 = crop_bounds(cx, w)
    return array_to_pil(arr[y0:y1, x0:x1])


class FigureImageDataset:
    def __init__(self, manifest: pd.DataFrame):
        self.rows = manifest.to_dict(orient="records")
        self._wsi_cache: dict[tuple[str, int], np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        label = int(row["feature_label"])
        sample_id = compact_label(row["sample_id"])
        sample_type = compact_label(row["sample_type"])
        if sample_type == "acrobat_tile":
            image = self._read_acrobat_tile(row)
        else:
            path = Path(compact_label(row["image_path"]))
            with Image.open(path) as img:
                image = ImageOps.exif_transpose(img).convert("RGB")
        return image, label, sample_id

    def _read_acrobat_tile(self, row: dict[str, Any]) -> Image.Image:
        import tifffile

        path = compact_label(row["image_path"])
        page_idx = int(row.get("tile_page") or 0)
        key = (path, page_idx)
        if key not in self._wsi_cache:
            with tifffile.TiffFile(path) as tif:
                self._wsi_cache[key] = tif.pages[page_idx].asarray()
        return crop_tissue_tile(self._wsi_cache[key], row)


def extract_hplus_features(
    manifest: pd.DataFrame,
    out_dir: Path,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[str]]:
    feature_path = out_dir / "features_hplus.npz"
    if feature_path.exists() and not args.overwrite_features:
        pack = np.load(feature_path, allow_pickle=True)
        return pack["features"].astype(np.float32), [str(x) for x in pack["paths"]]

    checkpoint = Path(args.checkpoint)
    train_config = Path(args.train_config)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint is missing: {checkpoint}")
    if not train_config.exists():
        raise FileNotFoundError(f"Train config is missing: {train_config}")

    import torch

    from dinov3.eval.bio_frozen_eval.encoder import (
        Dinov3CkptEncoder,
        extract_features,
        parse_autocast_dtype,
    )

    encoder = Dinov3CkptEncoder(
        checkpoint=checkpoint,
        train_config=train_config,
        device=args.device,
        n_last_blocks=args.n_last_blocks,
        use_avgpool=not args.no_avgpool,
        autocast_dtype=parse_autocast_dtype(args.autocast_dtype),
        image_size=args.image_size,
        resize_size=args.resize_size,
        channel_policy=args.channel_policy,
        channel_tta_samples=args.channel_tta_samples,
        channel_policy_seed=args.seed,
    )
    dataset = FigureImageDataset(manifest)
    features, _ = extract_features(
        dataset,
        encoder,
        output_path=feature_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        overwrite=True,
        model_name="biodino_hplus_s6_alpha1",
        save_features=True,
        save_paths=True,
    )
    torch.cuda.empty_cache()
    pack = np.load(feature_path, allow_pickle=True)
    return features.astype(np.float32), [str(x) for x in pack["paths"]]


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(denom, 1e-12, None)


def nearest_neighbor_accuracy(features: np.ndarray, labels: np.ndarray) -> float | None:
    valid = pd.Series(labels).map(compact_label).to_numpy()
    counts = Counter(valid.tolist())
    keep = np.asarray([counts[v] >= 2 for v in valid], dtype=bool)
    if keep.sum() < 4 or len(set(valid[keep].tolist())) < 2:
        return None
    x = l2_normalize(features[keep])
    y = valid[keep]
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    pred = y[np.argmax(sim, axis=1)]
    return float(np.mean(pred == y))


def safe_silhouette(features: np.ndarray, labels: np.ndarray) -> float | None:
    try:
        from sklearn.metrics import silhouette_score

        valid = pd.Series(labels).map(compact_label).to_numpy()
        counts = Counter(valid.tolist())
        keep = np.asarray([counts[v] >= 2 for v in valid], dtype=bool)
        if keep.sum() < 10 or len(set(valid[keep].tolist())) < 2:
            return None
        return float(silhouette_score(features[keep], valid[keep], metric="cosine"))
    except Exception as exc:
        warnings.warn(f"silhouette_score failed: {type(exc).__name__}: {exc}")
        return None


def compute_embedding(features: np.ndarray, seed: int) -> np.ndarray:
    try:
        import umap

        reducer = umap.UMAP(
            n_neighbors=30,
            min_dist=0.08,
            metric="cosine",
            random_state=seed,
        )
        return reducer.fit_transform(features)
    except Exception as exc:
        warnings.warn(f"UMAP failed, falling back to PCA: {type(exc).__name__}: {exc}")
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=seed).fit_transform(features)


def compute_pair_results(pair_df: pd.DataFrame, features: np.ndarray, paths: list[str]) -> pd.DataFrame:
    id_to_index = {sample_id: i for i, sample_id in enumerate(paths)}
    x = l2_normalize(features)
    records: list[dict[str, Any]] = []
    for _, row in pair_df.iterrows():
        a = compact_label(row["anchor_id"])
        b = compact_label(row["other_id"])
        if a not in id_to_index or b not in id_to_index:
            continue
        sim = float(np.dot(x[id_to_index[a]], x[id_to_index[b]]))
        rec = row.to_dict()
        rec["cosine_similarity"] = sim
        records.append(rec)
    if not records:
        raise RuntimeError("No Panel B pairs could be scored from extracted features.")
    return pd.DataFrame(records)


def bootstrap_alignment(pair_results: pd.DataFrame, seed: int, rounds: int = 1000) -> dict[str, float]:
    pos = pair_results[pair_results["pair_type"].eq("same_biology_cross_modality")]
    neg = pair_results[pair_results["pair_type"].eq("different_biology_same_modality")]
    score = float(pos["cosine_similarity"].mean() - neg["cosine_similarity"].mean())
    entities = sorted(set(pair_results["entity_id"].astype(str)))
    if len(entities) < 2:
        return {"score": score, "ci_low": score, "ci_high": score}
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    by_entity = {e: pair_results[pair_results["entity_id"].astype(str).eq(e)] for e in entities}
    for _ in range(rounds):
        draw = rng.choice(entities, size=len(entities), replace=True)
        boot = pd.concat([by_entity[e] for e in draw], axis=0)
        bpos = boot[boot["pair_type"].eq("same_biology_cross_modality")]
        bneg = boot[boot["pair_type"].eq("different_biology_same_modality")]
        if len(bpos) and len(bneg):
            vals.append(float(bpos["cosine_similarity"].mean() - bneg["cosine_similarity"].mean()))
    if not vals:
        return {"score": score, "ci_low": score, "ci_high": score}
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"score": score, "ci_low": float(lo), "ci_high": float(hi)}


def remap_top_labels(labels: pd.Series, max_labels: int, other: str = "other") -> tuple[list[str], list[str]]:
    labels = labels.astype(str)
    order = labels.value_counts().index.tolist()
    keep = set(order[:max_labels])
    mapped = [x if x in keep else other for x in labels]
    legend_order = [x for x in order[:max_labels] if x in set(mapped)]
    if other in mapped:
        legend_order.append(other)
    return mapped, legend_order


def scatter_with_labels(
    ax: plt.Axes,
    xy: np.ndarray,
    labels: pd.Series,
    title: str,
    max_labels: int,
    point_size: float = 12.0,
) -> None:
    mapped, order = remap_top_labels(labels, max_labels=max_labels)
    color_map = {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(order)}
    colors = [color_map[label] for label in mapped]
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=point_size, alpha=0.82, linewidths=0)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color_map[label], label=label, markersize=5)
        for label in order
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.02),
        frameon=False,
        fontsize=7,
        handletextpad=0.3,
        borderaxespad=0.0,
    )


def write_metrics(
    out_dir: Path,
    panel_a_df: pd.DataFrame,
    panel_a_features: np.ndarray,
    pair_results: pd.DataFrame,
    alignment: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "model": "Biodino H+ S6 nosigreg alpha=1.0",
        "checkpoint": str(Path(args.checkpoint)),
        "train_config": str(Path(args.train_config)),
        "panel_a": {
            "n": int(len(panel_a_df)),
            "dataset_counts": panel_a_df["dataset"].value_counts().to_dict(),
            "modality_counts": panel_a_df["modality"].value_counts().to_dict(),
            "biology_counts": panel_a_df["biology_id"].value_counts().to_dict(),
            "modality_nn_acc": nearest_neighbor_accuracy(panel_a_features, panel_a_df["modality"].to_numpy()),
            "biology_nn_acc": nearest_neighbor_accuracy(panel_a_features, panel_a_df["biology_id"].to_numpy()),
            "modality_silhouette_cosine": safe_silhouette(panel_a_features, panel_a_df["modality"].to_numpy()),
            "biology_silhouette_cosine": safe_silhouette(panel_a_features, panel_a_df["biology_id"].to_numpy()),
        },
        "panel_b": {
            "n_pairs": int(len(pair_results)),
            "pair_type_counts": pair_results["pair_type"].value_counts().to_dict(),
            "source_counts": pair_results["source"].value_counts().to_dict(),
            "mean_similarity": pair_results.groupby("pair_type")["cosine_similarity"].mean().to_dict(),
            "biological_alignment_score": alignment,
        },
    }
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def plot_figure(
    out_dir: Path,
    panel_a_df: pd.DataFrame,
    panel_a_xy: np.ndarray,
    pair_results: pd.DataFrame,
    metrics: dict[str, Any],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": "#f8f4eb",
            "figure.facecolor": "#f8f4eb",
            "axes.edgecolor": "#2b2b2b",
            "axes.labelcolor": "#2b2b2b",
            "xtick.color": "#2b2b2b",
            "ytick.color": "#2b2b2b",
        }
    )

    fig = plt.figure(figsize=(14.8, 9.2), dpi=220)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.82], wspace=0.43, hspace=0.38)
    ax_a1 = fig.add_subplot(gs[0, 0])
    ax_a2 = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_m = fig.add_subplot(gs[1, 1])

    scatter_with_labels(
        ax_a1,
        panel_a_xy,
        panel_a_df["modality"],
        "A  Biological embedding landscape | colored by modality",
        max_labels=9,
    )
    scatter_with_labels(
        ax_a2,
        panel_a_xy,
        panel_a_df["biology_id"],
        "A  Same embedding | colored by biology identity",
        max_labels=10,
    )

    pos = pair_results[pair_results["pair_type"].eq("same_biology_cross_modality")]["cosine_similarity"].to_numpy()
    neg = pair_results[pair_results["pair_type"].eq("different_biology_same_modality")]["cosine_similarity"].to_numpy()
    violin = ax_b.violinplot([pos, neg], positions=[1, 2], widths=0.72, showmeans=False, showextrema=False)
    for body, color in zip(violin["bodies"], ["#1b9e77", "#d95f02"]):
        body.set_facecolor(color)
        body.set_alpha(0.25)
        body.set_edgecolor("none")
    rng = np.random.default_rng(123)
    for xpos, values, color in [(1, pos, "#1b9e77"), (2, neg, "#d95f02")]:
        if len(values) > 1200:
            values = rng.choice(values, size=1200, replace=False)
        jitter = rng.normal(0, 0.055, size=len(values))
        ax_b.scatter(np.full(len(values), xpos) + jitter, values, s=7, alpha=0.18, color=color, linewidths=0)
        ax_b.plot([xpos - 0.24, xpos + 0.24], [np.mean(values), np.mean(values)], color=color, lw=3)
    ax_b.set_xticks([1, 2])
    ax_b.set_xticklabels(["same biology\ncross modality", "different biology\nsame modality"], fontsize=9)
    ax_b.set_ylabel("cosine similarity", fontsize=10)
    ax_b.set_title("B  Cross-modal biological correspondence", loc="left", fontsize=12, fontweight="bold", pad=8)
    ax_b.grid(axis="y", color="#d4c8b2", linewidth=0.8, alpha=0.8)
    score = metrics["panel_b"]["biological_alignment_score"]
    ax_b.text(
        0.03,
        0.96,
        f"Alignment score = {score['score']:.3f}\n95% bootstrap CI [{score['ci_low']:.3f}, {score['ci_high']:.3f}]",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", fc="#fff9ed", ec="#c7b58b", lw=1.0),
    )

    ax_m.axis("off")
    panel_a = metrics["panel_a"]
    panel_b = metrics["panel_b"]
    lines = [
        "Draft readout",
        "",
        f"Encoder: Biodino H+ S6 nosigreg alpha=1.0",
        f"Panel A samples: {panel_a['n']} test/validation images",
        "  " + ", ".join(f"{k}={v}" for k, v in panel_a["dataset_counts"].items()),
        f"Panel A modality NN acc: {panel_a['modality_nn_acc']:.3f}" if panel_a["modality_nn_acc"] is not None else "Panel A modality NN acc: n/a",
        f"Panel A biology NN acc: {panel_a['biology_nn_acc']:.3f}" if panel_a["biology_nn_acc"] is not None else "Panel A biology NN acc: n/a",
        "",
        f"Panel B pairs: {panel_b['n_pairs']} scored pairs",
        "  " + ", ".join(f"{k}={v}" for k, v in panel_b["pair_type_counts"].items()),
        f"Mean same-biology cross-modality sim: {panel_b['mean_similarity'].get('same_biology_cross_modality', float('nan')):.3f}",
        f"Mean different-biology same-modality sim: {panel_b['mean_similarity'].get('different_biology_same_modality', float('nan')):.3f}",
        "",
        "Interpretation target:",
        "Low modality separation with retained biology structure in A;",
        "positive alignment over same-modality negatives in B.",
    ]
    ax_m.text(
        0.02,
        0.97,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=10,
        linespacing=1.35,
        family="DejaVu Sans Mono",
        color="#2b2b2b",
    )

    fig.suptitle(
        "Fig. 3 draft - Biodino learns a universal biological representation",
        x=0.03,
        y=0.985,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color="#1f2528",
    )
    fig.text(
        0.03,
        0.018,
        "Panel A uses only held-out CytoImageNet validation plus withheld tissue examples. "
        "Panel B uses ACROBAT validation patient-paired H&E/IHC tiles and CIMA same-lesion multi-stain pairs.",
        fontsize=8,
        color="#4a4a4a",
    )
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out_dir / f"fig3_representation_draft.{ext}", bbox_inches="tight")
    plt.close(fig)


def make_readme(out_dir: Path, metrics: dict[str, Any]) -> None:
    text = f"""# Fig. 3 Representation Draft

Generated by `scripts/make_fig3_representation_draft.py`.

## Files

- `panel_a_test_mask.csv`: deterministic test/validation-only image mask for Panel A.
- `panel_b_test_pairs.csv`: positive and same-modality negative pairs for Panel B.
- `feature_manifest.csv`: unique samples that were encoded by the frozen H+ encoder.
- `features_hplus.npz`: cached L2-normalized H+ features.
- `panel_a_umap.csv`: two-dimensional embedding used by the figure.
- `panel_b_pair_scores.csv`: pairwise cosine similarities.
- `metrics.json`: numeric readouts for the draft figure.
- `fig3_representation_draft.png/.svg/.pdf`: rendered figure draft.

## Current Readout

- Panel A samples: {metrics['panel_a']['n']}
- Panel B pairs: {metrics['panel_b']['n_pairs']}
- Biological alignment score: {metrics['panel_b']['biological_alignment_score']['score']:.4f}

## Notes

This is a draft for the H+ S6 nosigreg alpha=1.0 checkpoint. Add DINOv2/ImageNet
baselines by reusing the same `feature_manifest.csv` and `panel_b_test_pairs.csv`,
so the mask/pair protocol stays fixed across encoders.
"""
    (out_dir / "README.md").write_text(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--panel-a-cyto-target", type=int, default=720)
    parser.add_argument("--panel-a-cima-max", type=int, default=120)
    parser.add_argument("--panel-b-mode", default="cyto", choices=["cyto", "histology", "cyto+histology"])
    parser.add_argument("--panel-b-pairs", type=int, default=600)
    parser.add_argument("--cima-scale", default="scale-25pc")
    parser.add_argument("--acrobat-tiles-per-patient", type=int, default=6)
    parser.add_argument("--acrobat-max-patients", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-last-blocks", type=int, default=1)
    parser.add_argument("--no-avgpool", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--autocast-dtype", default="bf16", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--channel-policy", default="auto")
    parser.add_argument("--channel-tta-samples", type=int, default=8)
    parser.add_argument("--overwrite-features", action="store_true")
    parser.add_argument("--build-manifests-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[fig3] building deterministic Panel A/B masks", flush=True)
    panel_a_df, pair_df, feature_df = build_manifests(args)
    save_manifests(out_dir, panel_a_df, pair_df, feature_df)
    print(
        f"[fig3] panel A n={len(panel_a_df)}; panel B pairs={len(pair_df)}; unique samples={len(feature_df)}",
        flush=True,
    )
    if args.build_manifests_only:
        return

    print("[fig3] extracting H+ features", flush=True)
    features, paths = extract_hplus_features(feature_df, out_dir, args)
    id_to_index = {sample_id: i for i, sample_id in enumerate(paths)}
    panel_a_indices = [id_to_index[sid] for sid in panel_a_df["sample_id"] if sid in id_to_index]
    panel_a_features = features[panel_a_indices]
    panel_a_df = panel_a_df[panel_a_df["sample_id"].isin(id_to_index)].copy().reset_index(drop=True)

    print("[fig3] computing UMAP and Panel B scores", flush=True)
    panel_a_xy = compute_embedding(panel_a_features, seed=args.seed)
    panel_a_umap = panel_a_df[["sample_id", "dataset", "modality", "biology_id", "organism", "tissue", "cell_type"]].copy()
    panel_a_umap["umap_1"] = panel_a_xy[:, 0]
    panel_a_umap["umap_2"] = panel_a_xy[:, 1]
    panel_a_umap.to_csv(out_dir / "panel_a_umap.csv", index=False)

    pair_results = compute_pair_results(pair_df, features, paths)
    pair_results.to_csv(out_dir / "panel_b_pair_scores.csv", index=False)
    alignment = bootstrap_alignment(pair_results, seed=args.seed)
    metrics = write_metrics(out_dir, panel_a_df, panel_a_features, pair_results, alignment, args)

    print("[fig3] plotting draft figure", flush=True)
    plot_figure(out_dir, panel_a_df, panel_a_xy, pair_results, metrics)
    make_readme(out_dir, metrics)
    print(f"[fig3] done: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
