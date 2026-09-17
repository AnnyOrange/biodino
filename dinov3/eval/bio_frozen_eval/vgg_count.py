"""Official VGG synthetic-count pairing and seeded published N=32 draws."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from .candidate_datasets import digest, file_digest, image_digests, validate_records
from .probes import run_regression_probe_split

FOLDER = "ood/ood_regression/datasets/VGG_Cell_Counting"
OFFICIAL_ARCHIVE_SHA256 = "d254630889aab3ab166e2e8757fe7d256e5e81f89774e1d7d9fdae5ff4b668bd"
DATA_URL = "https://www.robots.ox.ac.uk/~vgg/research/counting/cells.zip"
PAPER_URL = "https://www.robots.ox.ac.uk/~vgg/publications/2010/Lempitsky10b/lempitsky10b.pdf"


def dot_count(path):
    with Image.open(path) as image:
        points = np.asarray(image)
    if points.shape != (256, 256, 3) or points.dtype != np.uint8:
        raise ValueError(f"Unexpected official dot-map dimensions/dtype: {path}")
    if np.any(points[:, :, 1:]) or not np.isin(points[:, :, 0], [0, 255]).all():
        raise ValueError(f"Expected binary red-channel single-pixel annotations: {path}")
    # The official MATLAB example uses double(gtDensities(:,:,1))/255.
    return int(np.count_nonzero(points[:, :, 0]))


def published_draws(seed=0, n_train=32, repetitions=5):
    if n_train not in (1, 2, 4, 8, 16, 32) or repetitions != 5:
        raise ValueError("Published experiment uses N in {1,2,4,8,16,32} and five draws")
    draws = []
    for repetition in range(repetitions):
        order = np.random.default_rng(seed + repetition).permutation(100) + 1
        draws.append({"repetition": repetition,
                      "train": [f"{i:03d}" for i in order[:n_train]],
                      "val": [f"{i:03d}" for i in order[n_train:2*n_train]],
                      "unused_development": [f"{i:03d}" for i in order[2*n_train:]],
                      "test": [f"{i:03d}" for i in range(101, 201)]})
    return draws


def build_vgg_manifest(benchmark_root, seed=0, n_train=32):
    folder = Path(benchmark_root) / FOLDER
    root, archive = folder / "extracted", folder / "archives/cells.zip"
    if file_digest(archive) != OFFICIAL_ARCHIVE_SHA256:
        raise ValueError("Official VGG archive identity mismatch")
    expected = {f"{i:03d}{kind}.png" for i in range(1, 201) for kind in ("cell", "dots")}
    if {p.name for p in root.iterdir() if p.is_file()} != expected:
        raise ValueError("Expected exactly200 image/dot pairs, IDs001..200")
    records = []
    with zipfile.ZipFile(archive) as source:
        if set(source.namelist()) != expected or source.testzip() is not None:
            raise ValueError("Official archive member count or CRC failure")
        for i in range(1, 201):
            image, annotation = root / f"{i:03d}cell.png", root / f"{i:03d}dots.png"
            for path in (image, annotation):
                if source.read(path.name) != path.read_bytes():
                    raise ValueError(f"Extracted file differs official archive: {path}")
            with Image.open(image) as pixels:
                if pixels.size != (256, 256) or pixels.mode != "RGB":
                    raise ValueError(f"Unexpected official input image dimensions/mode: {image}")
                pixels.load()
            encoded_sha, pixel_sha = image_digests(image)
            records.append({"sample_id": f"{i:03d}", "path": image.name,
                            "annotation_path": annotation.name,
                            "target": dot_count(annotation), "group": f"synthetic-source:{i:03d}",
                            "official_pool": "development" if i <= 100 else "test",
                            "source_split": "development" if i <= 100 else "test",
                            "image_sha256": encoded_sha, "image_pixel_sha256": pixel_sha,
                            "annotation_sha256": file_digest(annotation)})
    draws, stats = published_draws(seed, n_train), []
    for draw in draws:
        assignments = {sample: split for split in ("train", "val", "test") for sample in draw[split]}
        stats.append(validate_records([{**row, "split": assignments[row["sample_id"]]}
                                       for row in records if row["sample_id"] in assignments]))
    manifest = {"dataset": "vgg-synthetic-cell-count", "task": "regression", "version": 1,
                "status": "PASS", "seed": seed, "n_train": n_train, "repetitions": 5,
                "official_protocol": "OFFICIAL: first100 development, second100 test; Ntrain/Nvalidation, five random draws",
                "draw_identity": "PROPOSED_BY_US: NumPy default_rng(seed+repetition), seeds0..4 at base seed0; official exact random indices unavailable",
                "grouping": "synthetic source image; no crops or transformed copies across partitions",
                "target_definition": "Raw full256x256 image count=sum(red annotation channel/255), no smoothing/cropping",
                "data_url": DATA_URL, "paper_url": PAPER_URL,
                "archive_sha256": OFFICIAL_ARCHIVE_SHA256, "archive_crc_verified": True,
                "payload_matches_archive": True, "records": records, "draws": draws,
                "folds": [{"fold": d["repetition"], "draw_seed": seed+d["repetition"],
                           **{split: d[split] for split in ("train", "val", "test")}} for d in draws],
                "draw_stats": stats, "expected_pairs": 200, "verified_pairs": len(records),
                "primary_metric": "mae", "aggregation": "Mean and population SD of five per-draw test MAEs; same100 test images, not five independent test cohorts",
                "probe": "Existing run_regression_probe_split: training-only StandardScaler + sklearn Ridge, raw targets",
                "selected_using_1tb": False, "frozen": False,
                "limitations": ["Global-feature count regression is the paper's counting-by-regression baseline task, not its density/MESA method.",
                                "Full-FOV count benchmark; not comparable with cropped/density-smoothed boundary counts.",
                                "Official example code is research-only/noncommercial with citation; standalone data archive has no explicit separate license. Do not assume CC or commercial permission."]}
    manifest["manifest_sha256"] = digest(manifest)
    return manifest


class VGGCountDataset:
    def __init__(self, benchmark_root, manifest, split, repetition=0):
        if split not in ("development", "train", "val", "test"):
            raise ValueError("Expected development/train/val/test")
        self.root = Path(benchmark_root) / FOLDER / "extracted"
        by_id = {row["sample_id"]: row for row in manifest["records"]}
        if split in ("development", "test"):
            self.records = [row for row in manifest["records"] if row["source_split"] == split]
        else:
            if not 0 <= repetition < manifest["repetitions"]:
                raise ValueError("Invalid draw repetition")
            self.records = [by_id[sample] for sample in manifest["draws"][repetition][split]]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(self.root / row["path"]) as image:
            rgb = image.convert("RGB")
        return rgb, float(row["target"]), row["sample_id"]


def ridge_draw_scores(features, manifest, alpha, split="val"):
    if split not in ("val", "test"):
        raise ValueError("Score validation or frozen test only")
    features = np.asarray(features)
    if len(features) != 200 or not np.isfinite(features).all():
        raise ValueError("Features must contain exactly200 finite source-image rows in manifest order")
    by_id = {row["sample_id"]: i for i, row in enumerate(manifest["records"])}
    targets = np.asarray([row["target"] for row in manifest["records"]], dtype=float)
    scores = []
    for draw in manifest["draws"]:
        train = [by_id[s] for s in draw["train"]]
        evaluation = [by_id[s] for s in draw[split]]
        score = run_regression_probe_split(features[train], targets[train],
                                           features[evaluation], targets[evaluation], alpha=float(alpha))
        scores.append({"repetition": draw["repetition"], "alpha": float(alpha),
                       "split": split, **score.to_dict()})
    maes = np.asarray([row["mae"] for row in scores])
    return {"mean_mae": float(maes.mean()), "std_mae": float(maes.std(ddof=0)), "draw_scores": scores}


def ridge_development_sweep(features, manifest, alphas=(0.1, 1., 10., 100., 1000.)):
    sweep = [{"alpha": float(alpha), **ridge_draw_scores(features, manifest, alpha)} for alpha in alphas]
    if not sweep:
        raise ValueError("Empty regularization sweep")
    winner = min(sweep, key=lambda row: (row["mean_mae"], row["alpha"]))
    return {"sweep": sweep, "selected_alpha": winner["alpha"], "selection_metric": "mean_val_mae", "test_accessed": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    manifest = build_vgg_manifest(args.benchmark_root, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": manifest["status"], "pairs": manifest["verified_pairs"], "manifest_sha256": manifest["manifest_sha256"]}))


if __name__ == "__main__":
    main()
