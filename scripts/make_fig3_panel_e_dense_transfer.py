#!/usr/bin/env python3
"""Build evidence-backed dense-transfer assets for Fig. 3 Panel E.

The panel uses frozen final-layer patch tokens with a learned 1x1 segmentation
head. It exports a deterministic median-performance TissueNet example and
paired test-set metrics for BioDINO H+ versus the original DINOv3 H+.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUT_DIR = REPO_ROOT / "outputs/04_figures/fig3_representation_20260812"
DEFAULT_DENSE_ROOT = REPO_ROOT / "outputs/02_eval_runs/fig3_hplus_external_dense_protocol_20260813_v2"
DEFAULT_TISSUENET_ROOT = Path("/mnt/huawei_deepcad/benchmark/segmentation/tissuenet/extracted")
DATASETS = ("conic", "monuseg", "pannuke", "livecell", "bbbc038", "tissuenet")
DISPLAY_NAMES = {
    "conic": "CoNIC",
    "monuseg": "MoNuSeg",
    "pannuke": "PanNuke",
    "livecell": "LIVECell",
    "bbbc038": "BBBC038",
    "tissuenet": "TissueNet",
}
MODELS = {
    "dinov3_hplus_official": "Original DINOv3 H+",
    "biodino_hplus": "BioDINO H+",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-root", type=Path, default=DEFAULT_DENSE_ROOT)
    parser.add_argument("--tissuenet-root", type=Path, default=DEFAULT_TISSUENET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def result_path(root: Path, model: str, dataset: str) -> Path:
    return root / model / "linear_probe" / dataset / model / "results.json"


def head_path(root: Path, model: str, dataset: str) -> Path:
    return root / model / "linear_probe" / dataset / model / "best_head.pth"


def cache_path(root: Path, model: str, dataset: str) -> Path:
    return root / model / "cache" / dataset / model / "test.npz"


def load_head(path: Path, in_channels: int, device: str):
    import torch

    from dinov3.eval.bio_segmentation.linear_probe import LinearSegHead

    head = LinearSegHead(in_channels, num_classes=2, dropout=0.1).to(device)
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    head.load_state_dict(state)
    return head.eval()


def predict(head, features: np.ndarray, out_size: tuple[int, int], device: str) -> np.ndarray:
    import torch

    with torch.inference_mode():
        x = torch.from_numpy(np.asarray(features)).to(device=device, dtype=torch.float32)
        logits = head(x, out_size=out_size)
        return logits.argmax(dim=1).cpu().numpy().astype(np.uint8)


def foreground_dice(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    valid = target != 255
    pred_fg = (pred == 1) & valid
    target_fg = (target == 1) & valid
    axes = tuple(range(1, pred.ndim))
    intersection = np.logical_and(pred_fg, target_fg).sum(axis=axes)
    denominator = pred_fg.sum(axis=axes) + target_fg.sum(axis=axes)
    return (2.0 * intersection + 1e-8) / (denominator + 1e-8)


def select_median_tissuenet_example(cache: Path, head, device: str) -> tuple[int, float, np.ndarray]:
    scores: list[float] = []
    with np.load(cache, allow_pickle=False) as data:
        n_chunks = int(data["num_chunks"])
        out_size = (int(data["orig_H"]), int(data["orig_W"]))
        for chunk_index in range(n_chunks):
            features = data[f"features_{chunk_index:04d}"]
            target = data[f"sem_masks_{chunk_index:04d}"]
            pred = predict(head, features, out_size, device)
            scores.extend(foreground_dice(pred, target).tolist())

        score_array = np.asarray(scores, dtype=np.float64)
        median = float(np.median(score_array))
        sample_index = int(np.argmin(np.abs(score_array - median)))
        pred, target = load_cached_sample(data, head, sample_index, out_size, device)
    return sample_index, float(score_array[sample_index]), np.stack([pred, target])


def load_cached_sample(data, head, sample_index: int, out_size: tuple[int, int], device: str) -> tuple[np.ndarray, np.ndarray]:
    offset = 0
    n_chunks = int(data["num_chunks"])
    for chunk_index in range(n_chunks):
        features = data[f"features_{chunk_index:04d}"]
        if sample_index < offset + len(features):
            local_index = sample_index - offset
            target = data[f"sem_masks_{chunk_index:04d}"][local_index]
            pred = predict(head, features[local_index : local_index + 1], out_size, device)[0]
            return pred, target
        offset += len(features)
    raise IndexError(f"Sample {sample_index} is outside cache with {offset} samples")


def find_tissuenet_test_npz(root: Path) -> Path:
    candidates = sorted(root.rglob("*test*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No TissueNet test NPZ found under {root}")
    return candidates[0]


def load_tissuenet_image(path: Path, sample_index: int) -> np.ndarray:
    from dinov3.eval.bio_segmentation.preprocessing import apply_preprocessing_single_channel

    with np.load(path, mmap_mode="r", allow_pickle=False) as data:
        sample = data["X"][sample_index]
    nuclear = apply_preprocessing_single_channel(sample[..., 0].astype(np.float32), mode="percentile")
    whole_cell = apply_preprocessing_single_channel(sample[..., 1].astype(np.float32), mode="percentile")
    return np.stack([nuclear, whole_cell, nuclear], axis=-1)


def blend_mask(image: np.ndarray, mask: np.ndarray, color: tuple[float, float, float], alpha: float) -> np.ndarray:
    out = image.copy()
    selected = mask.astype(bool)
    out[selected] = (1.0 - alpha) * out[selected] + alpha * np.asarray(color, dtype=np.float32)
    return np.clip(out, 0.0, 1.0)


def ground_truth_overlay(image: np.ndarray, target: np.ndarray) -> np.ndarray:
    return blend_mask(image * 0.72, target == 1, (0.04, 0.82, 0.70), 0.62)


def prediction_error_overlay(image: np.ndarray, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    valid = target != 255
    true_positive = (pred == 1) & (target == 1) & valid
    false_positive = (pred == 1) & (target == 0) & valid
    false_negative = (pred == 0) & (target == 1) & valid
    out = image * 0.58
    out = blend_mask(out, true_positive, (0.04, 0.77, 0.67), 0.66)
    out = blend_mask(out, false_positive, (0.95, 0.49, 0.17), 0.78)
    out = blend_mask(out, false_negative, (0.82, 0.16, 0.43), 0.82)
    return out


def save_rgb(array: np.ndarray, path: Path) -> None:
    image = Image.fromarray(np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
    image.save(path)


def collect_metrics(root: Path, output_path: Path) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    excluded: list[str] = []
    for dataset in DATASETS:
        paths = {model: result_path(root, model, dataset) for model in MODELS}
        missing = [model for model, path in paths.items() if not path.is_file()]
        if missing:
            excluded.append(f"{dataset}: missing {', '.join(missing)}")
            continue
        for model, display_name in MODELS.items():
            result = json.loads(paths[model].read_text())
            test = result["test"]
            meta = result.get("_meta", {})
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": DISPLAY_NAMES[dataset],
                    "model": model,
                    "model_label": display_name,
                    "mDice": float(test["mDice"]),
                    "mIoU": float(test["mIoU"]),
                    "probe_epochs": int(meta.get("probe_epochs", 20)),
                    "seed": int(meta.get("seed", 0)),
                    "results_json": str(paths[model]),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No paired dense-probe results found under {root}")
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows, excluded


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bio_model = "biodino_hplus"
    official_model = "dinov3_hplus_official"
    dataset = "tissuenet"
    bio_cache = cache_path(args.dense_root, bio_model, dataset)
    official_cache = cache_path(args.dense_root, official_model, dataset)
    for required in (
        bio_cache,
        official_cache,
        head_path(args.dense_root, bio_model, dataset),
        head_path(args.dense_root, official_model, dataset),
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    with np.load(bio_cache, allow_pickle=False) as data:
        in_channels = int(data["embed_dim"])
        out_size = (int(data["orig_H"]), int(data["orig_W"]))
    bio_head = load_head(head_path(args.dense_root, bio_model, dataset), in_channels, args.device)
    sample_index, bio_dice, bio_pair = select_median_tissuenet_example(bio_cache, bio_head, args.device)
    bio_pred, target = bio_pair[0], bio_pair[1]

    official_head = load_head(head_path(args.dense_root, official_model, dataset), in_channels, args.device)
    with np.load(official_cache, allow_pickle=False) as data:
        official_pred, official_target = load_cached_sample(
            data, official_head, sample_index, out_size, args.device
        )
    if not np.array_equal(target, official_target):
        raise ValueError("Paired TissueNet caches do not contain the same target at the selected index")
    official_dice = float(foreground_dice(official_pred[None], target[None])[0])

    tissuenet_npz = find_tissuenet_test_npz(args.tissuenet_root)
    image = load_tissuenet_image(tissuenet_npz, sample_index)
    if image.shape[:2] != target.shape:
        image = np.asarray(
            Image.fromarray(np.rint(image * 255.0).astype(np.uint8)).resize(
                (target.shape[1], target.shape[0]), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0

    qualitative_paths = {
        "input": args.output_dir / "panel_e_dense_tissuenet_input.png",
        "ground_truth": args.output_dir / "panel_e_dense_tissuenet_ground_truth.png",
        "official_prediction": args.output_dir / "panel_e_dense_tissuenet_official_prediction.png",
        "biodino_prediction": args.output_dir / "panel_e_dense_tissuenet_biodino_prediction.png",
    }
    save_rgb(image, qualitative_paths["input"])
    save_rgb(ground_truth_overlay(image, target), qualitative_paths["ground_truth"])
    save_rgb(
        prediction_error_overlay(image, official_pred, target),
        qualitative_paths["official_prediction"],
    )
    save_rgb(
        prediction_error_overlay(image, bio_pred, target),
        qualitative_paths["biodino_prediction"],
    )

    metrics_path = args.output_dir / "panel_e_dense_transfer_metrics.csv"
    rows, excluded = collect_metrics(args.dense_root, metrics_path)
    metadata = {
        "claim": "Frozen BioDINO patch tokens support cell and nucleus segmentation with a linear head.",
        "protocol": {
            "backbone": "frozen",
            "features": "final-layer patch tokens",
            "head": "BatchNorm plus 1x1 convolution",
            "probe_epochs": 20,
            "seed": 0,
            "encoder_input_size": 224,
            "selection_metric": "validation mIoU",
            "reported_split": "test",
        },
        "qualitative": {
            "dataset": "TissueNet",
            "target": "nucleus",
            "sample_index": sample_index,
            "selection_rule": "lowest-index test sample closest to the BioDINO median per-image foreground Dice",
            "biodino_foreground_dice": bio_dice,
            "official_foreground_dice": official_dice,
            "source_npz": str(tissuenet_npz),
            "assets": {key: str(path) for key, path in qualitative_paths.items()},
            "overlay_colors": {
                "true_positive": "teal",
                "false_positive": "orange",
                "false_negative": "magenta",
            },
        },
        "metrics_csv": str(metrics_path),
        "paired_datasets": sorted({str(row["dataset"]) for row in rows}),
        "excluded": excluded,
        "dense_root": str(args.dense_root),
    }
    metadata_path = args.output_dir / "panel_e_dense_transfer_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"[fig3-panel-e] sample={sample_index} BioDINO Dice={bio_dice:.4f} "
        f"original Dice={official_dice:.4f}; wrote {metadata_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
