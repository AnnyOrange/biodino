#!/usr/bin/env python3
"""Calibrate abstaining BioCLIP species/acquisition routing on recovered truth.

The packed training set lost its source-domain fields.  This script uses the
small provenance subset that can be recovered as a source-grouped calibration
and holdout set.  Predictions are made once per original image (not per crop),
and class-specific confidence thresholds abstain when calibration evidence is
insufficient.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F


ORGANISM_PROMPTS = {
    "human": (
        "a microscopy image of human cells",
        "a microscopy image of human tissue",
        "Homo sapiens under a microscope",
    ),
    "mouse": (
        "a microscopy image of mouse cells",
        "a microscopy image of mouse tissue",
        "Mus musculus under a microscope",
    ),
    "rat": (
        "a microscopy image of rat cells",
        "a microscopy image of rat tissue",
        "Rattus norvegicus under a microscope",
    ),
    "zebrafish": (
        "a microscopy image of zebrafish cells",
        "a microscopy image of zebrafish tissue",
        "Danio rerio under a microscope",
    ),
    "insect": (
        "a microscopy image of insect cells",
        "a microscopy image of insect tissue",
        "an insect under a microscope",
    ),
    "yeast": (
        "a microscopy image of yeast cells",
        "Saccharomyces cerevisiae under a microscope",
        "budding yeast microscopy",
    ),
    "bacteria": (
        "a microscopy image of bacterial cells",
        "bacteria under a microscope",
        "micrograph of a bacterial culture",
    ),
    "plant": (
        "a microscopy image of plant cells",
        "a microscopy image of plant tissue",
        "plant cells under a microscope",
    ),
}

ACQUISITION_PROMPTS = {
    "fluorescence_microscopy": (
        "a fluorescence microscopy image",
        "a confocal fluorescence microscopy image",
        "fluorescently labeled cells under a microscope",
    ),
    "label_free_microscopy": (
        "a brightfield microscopy image",
        "a phase contrast microscopy image",
        "unstained cells under a light microscope",
    ),
    "histopathology": (
        "a histopathology microscopy image",
        "an H and E stained tissue section",
        "a brightfield pathology slide",
    ),
    "electron_microscopy": (
        "an electron microscopy image",
        "a transmission electron micrograph",
        "a scanning electron micrograph",
    ),
    "imaging_mass_cytometry": (
        "an imaging mass cytometry image",
        "a multiplexed ion beam tissue image",
        "a mass cytometry tissue image",
    ),
    "light_sheet_fluorescence": (
        "a light sheet fluorescence microscopy image",
        "a light sheet micrograph of a whole organism",
        "a three dimensional light sheet microscopy image",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--truth-overlay", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--open-clip-root",
        type=Path,
        default=Path("/mnt/huawei_deepcad/benchmark_model/_vendor/external_gapfill_py311"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout-fraction", type=float, default=0.5)
    parser.add_argument("--target-calibration-precision", type=float, default=0.8)
    parser.add_argument("--min-calibration-accepted", type=int, default=3)
    parser.add_argument("--min-holdout-precision", type=float, default=0.7)
    parser.add_argument("--min-holdout-accepted", type=int, default=3)
    parser.add_argument("--min-routed-classes", type=int, default=2)
    parser.add_argument("--enforce-gates", action="store_true")
    return parser.parse_args()


def canonical_organism(value: str) -> str:
    normalized = str(value).strip().lower()
    if not normalized or normalized in {"unknown", "unresolved", "mixed sample"}:
        return ""
    mappings = (
        (("homo sapiens", "human"), "human"),
        (("mus musculus", "mouse"), "mouse"),
        (("rattus norvegicus", "rat"), "rat"),
        (("danio rerio", "zebrafish"), "zebrafish"),
        (("drosophila", "tribolium", "insect"), "insect"),
        (("saccharomyces", "yeast"), "yeast"),
        (("staphylococcus", "escherichia", "bacter"), "bacteria"),
        (("arabidopsis", "plant"), "plant"),
    )
    for needles, label in mappings:
        if any(needle in normalized for needle in needles):
            return label
    return ""


def canonical_acquisition(value: str) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "fluorescence_microscopy": "fluorescence_microscopy",
        "confocal_microscopy": "fluorescence_microscopy",
        "label_free_microscopy": "label_free_microscopy",
        "brightfield_microscopy": "label_free_microscopy",
        "phase_contrast_microscopy": "label_free_microscopy",
        "histopathology": "histopathology",
        "electron_microscopy": "electron_microscopy",
        "imaging_mass_cytometry": "imaging_mass_cytometry",
        "light_sheet_fluorescence": "light_sheet_fluorescence",
    }
    return aliases.get(normalized, "")


def aggregate_by_source(features: np.ndarray, source_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_ids, inverse = np.unique(source_ids, return_inverse=True)
    sums = np.zeros((len(unique_ids), features.shape[1]), dtype=np.float32)
    np.add.at(sums, inverse, features.astype(np.float32, copy=False))
    counts = np.bincount(inverse, minlength=len(unique_ids)).astype(np.float32)
    means = sums / counts[:, None]
    norms = np.linalg.norm(means, axis=1, keepdims=True)
    means /= np.maximum(norms, np.finfo(np.float32).eps)
    return unique_ids, inverse, means


def grouped_truth(
    values: np.ndarray,
    inverse: np.ndarray,
    canonicalize: Callable[[str], str],
) -> np.ndarray:
    grouped: list[set[str]] = [set() for _ in range(int(inverse.max()) + 1)]
    for row, value in zip(inverse, values):
        label = canonicalize(str(value))
        if label:
            grouped[int(row)].add(label)
    conflicts = [sorted(labels) for labels in grouped if len(labels) > 1]
    if conflicts:
        raise ValueError(f"Conflicting recovered labels within a source image: {conflicts[:3]}")
    return np.asarray([next(iter(labels), "") for labels in grouped])


def stratified_split(labels: np.ndarray, holdout_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    calibration: list[int] = []
    holdout: list[int] = []
    for label in sorted(set(labels.tolist()) - {""}):
        indices = np.flatnonzero(labels == label)
        indices = rng.permutation(indices)
        if len(indices) < 2:
            calibration.extend(indices.tolist())
            continue
        n_holdout = min(len(indices) - 1, max(1, int(round(len(indices) * holdout_fraction))))
        holdout.extend(indices[:n_holdout].tolist())
        calibration.extend(indices[n_holdout:].tolist())
    return np.asarray(sorted(calibration), dtype=np.int64), np.asarray(sorted(holdout), dtype=np.int64)


def load_text_prototypes(
    model_path: Path,
    open_clip_root: Path,
    prompt_groups: dict[str, tuple[str, ...]],
    device: str,
) -> tuple[list[str], np.ndarray]:
    sys.path.insert(0, str(open_clip_root))
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained=None)
    state = torch.load(model_path / "open_clip_pytorch_model.bin", map_location="cpu")
    missing, unexpected = model.load_state_dict(state.get("state_dict", state), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"BioCLIP checkpoint mismatch: missing={len(missing)} unexpected={len(unexpected)}"
        )
    model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    labels: list[str] = []
    prototypes: list[torch.Tensor] = []
    with torch.inference_mode():
        for label, prompts in prompt_groups.items():
            tokens = tokenizer(list(prompts)).to(device)
            encoded = F.normalize(model.encode_text(tokens).float(), dim=-1)
            prototype = F.normalize(encoded.mean(dim=0), dim=0)
            labels.append(label)
            prototypes.append(prototype.cpu())
    del model
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return labels, torch.stack(prototypes).numpy()


def predictions(features: np.ndarray, prototypes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = features @ prototypes.T
    order = np.argsort(scores, axis=1)
    best = order[:, -1]
    top = scores[np.arange(len(scores)), best]
    second = scores[np.arange(len(scores)), order[:, -2]]
    return best, top, top - second


def fit_class_thresholds(
    predicted: np.ndarray,
    margins: np.ndarray,
    truth: np.ndarray,
    calibration_indices: np.ndarray,
    labels: list[str],
    target_precision: float,
    min_accepted: int,
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for class_index, label in enumerate(labels):
        eligible = calibration_indices[predicted[calibration_indices] == class_index]
        eligible = eligible[truth[eligible] != ""]
        if len(eligible) < min_accepted:
            thresholds[label] = float("inf")
            continue
        ranked = eligible[np.argsort(-margins[eligible])]
        correct = (truth[ranked] == label).astype(np.int64)
        precision = np.cumsum(correct) / np.arange(1, len(ranked) + 1)
        candidates = np.flatnonzero(
            (precision >= target_precision)
            & (np.arange(1, len(ranked) + 1) >= min_accepted)
        )
        if len(candidates) == 0:
            thresholds[label] = float("inf")
            continue
        # The longest precision-valid prefix maximizes calibrated coverage.
        last = int(candidates[-1])
        thresholds[label] = float(margins[ranked[last]])
    return thresholds


def accepted_mask(
    predicted: np.ndarray,
    margins: np.ndarray,
    labels: list[str],
    thresholds: dict[str, float],
) -> np.ndarray:
    cutoffs = np.asarray([thresholds[label] for label in labels], dtype=np.float32)
    return margins >= cutoffs[predicted]


def evaluate_routing(
    name: str,
    predicted: np.ndarray,
    margins: np.ndarray,
    truth: np.ndarray,
    indices: np.ndarray,
    labels: list[str],
    thresholds: dict[str, float],
    *,
    min_holdout_precision: float,
    min_holdout_accepted: int,
    min_routed_classes: int,
) -> dict[str, object]:
    known = indices[truth[indices] != ""]
    accepted = accepted_mask(predicted, margins, labels, thresholds)
    selected = known[accepted[known]]
    predicted_labels = np.asarray(labels)[predicted]
    raw_accuracy = float(np.mean(predicted_labels[known] == truth[known])) if len(known) else 0.0
    selective_precision = (
        float(np.mean(predicted_labels[selected] == truth[selected])) if len(selected) else 0.0
    )
    per_class = {}
    for label in labels:
        class_selected = selected[predicted_labels[selected] == label]
        if len(class_selected):
            per_class[label] = {
                "accepted": int(len(class_selected)),
                "precision": float(np.mean(truth[class_selected] == label)),
            }
    routed_classes = sum(value["accepted"] > 0 for value in per_class.values())
    gate_pass = (
        len(selected) >= min_holdout_accepted
        and selective_precision >= min_holdout_precision
        and routed_classes >= min_routed_classes
    )
    return {
        "name": name,
        "known": int(len(known)),
        "raw_accuracy": raw_accuracy,
        "accepted": int(len(selected)),
        "coverage": float(len(selected) / len(known)) if len(known) else 0.0,
        "selective_precision": selective_precision,
        "routed_classes": int(routed_classes),
        "per_predicted_class": per_class,
        "gate_pass": bool(gate_pass),
    }


def task_routing(
    *,
    name: str,
    features: np.ndarray,
    truth: np.ndarray,
    prompt_groups: dict[str, tuple[str, ...]],
    model_path: Path,
    open_clip_root: Path,
    device: str,
    calibration_indices: np.ndarray,
    holdout_indices: np.ndarray,
    target_precision: float,
    min_calibration_accepted: int,
    min_holdout_precision: float,
    min_holdout_accepted: int,
    min_routed_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    labels, prototypes = load_text_prototypes(model_path, open_clip_root, prompt_groups, device)
    predicted, top_scores, margins = predictions(features, prototypes)
    thresholds = fit_class_thresholds(
        predicted,
        margins,
        truth,
        calibration_indices,
        labels,
        target_precision,
        min_calibration_accepted,
    )
    calibration_report = evaluate_routing(
        "calibration",
        predicted,
        margins,
        truth,
        calibration_indices,
        labels,
        thresholds,
        min_holdout_precision=target_precision,
        min_holdout_accepted=min_calibration_accepted,
        min_routed_classes=min_routed_classes,
    )
    holdout_report = evaluate_routing(
        "holdout",
        predicted,
        margins,
        truth,
        holdout_indices,
        labels,
        thresholds,
        min_holdout_precision=min_holdout_precision,
        min_holdout_accepted=min_holdout_accepted,
        min_routed_classes=min_routed_classes,
    )
    accepted = accepted_mask(predicted, margins, labels, thresholds)
    predicted_labels = np.asarray(labels)[predicted]
    report = {
        "labels": labels,
        "thresholds": {
            label: (value if np.isfinite(value) else None) for label, value in thresholds.items()
        },
        "truth_counts": dict(Counter(value for value in truth if value)),
        "predicted_counts": dict(Counter(predicted_labels.tolist())),
        "accepted_counts": dict(Counter(predicted_labels[accepted].tolist())),
        "calibration": calibration_report,
        "holdout": holdout_report,
        "gate_pass": bool(holdout_report["gate_pass"]),
    }
    return predicted_labels, top_scores, margins, report


def main() -> None:
    args = parse_args()
    if not 0 < args.holdout_fraction < 1:
        raise ValueError("--holdout-fraction must be in (0, 1)")
    if not 0 < args.target_calibration_precision <= 1:
        raise ValueError("--target-calibration-precision must be in (0, 1]")
    if not 0 < args.min_holdout_precision <= 1:
        raise ValueError("--min-holdout-precision must be in (0, 1]")
    if args.min_calibration_accepted <= 0 or args.min_holdout_accepted <= 0:
        raise ValueError("Minimum accepted counts must be positive")

    with np.load(args.bank, allow_pickle=False) as bank:
        keys = np.asarray(bank["keys"]).astype(str)
        features = np.asarray(bank["features"], dtype=np.float32)
    with np.load(args.truth_overlay, allow_pickle=False) as overlay:
        truth_keys = np.asarray(overlay["keys"]).astype(str)
        source_ids = np.asarray(overlay["source_id"], dtype=np.int64)
        raw_organisms = np.asarray(overlay["organism"]).astype(str)
        raw_acquisitions = np.asarray(overlay["acquisition_family"]).astype(str)
        recovered = np.asarray(overlay["recovered"], dtype=np.bool_)
    if not np.array_equal(keys, truth_keys):
        raise ValueError("Feature bank and truth overlay keys differ")
    if features.ndim != 2 or len(features) != len(keys):
        raise ValueError(f"Invalid feature bank shape: {features.shape} for {len(keys)} keys")

    unique_ids, inverse, grouped_features = aggregate_by_source(features, source_ids)
    organism_truth = grouped_truth(raw_organisms, inverse, canonical_organism)
    acquisition_truth = grouped_truth(raw_acquisitions, inverse, canonical_acquisition)
    organism_cal, organism_holdout = stratified_split(
        organism_truth, args.holdout_fraction, args.seed
    )
    acquisition_cal, acquisition_holdout = stratified_split(
        acquisition_truth, args.holdout_fraction, args.seed + 1
    )

    common = {
        "features": grouped_features,
        "model_path": args.model_path,
        "open_clip_root": args.open_clip_root,
        "device": args.device,
        "target_precision": args.target_calibration_precision,
        "min_calibration_accepted": args.min_calibration_accepted,
        "min_holdout_precision": args.min_holdout_precision,
        "min_holdout_accepted": args.min_holdout_accepted,
        "min_routed_classes": args.min_routed_classes,
    }
    organism_prediction, organism_score, organism_margin, organism_report = task_routing(
        name="organism",
        truth=organism_truth,
        prompt_groups=ORGANISM_PROMPTS,
        calibration_indices=organism_cal,
        holdout_indices=organism_holdout,
        **common,
    )
    acquisition_prediction, acquisition_score, acquisition_margin, acquisition_report = task_routing(
        name="acquisition",
        truth=acquisition_truth,
        prompt_groups=ACQUISITION_PROMPTS,
        calibration_indices=acquisition_cal,
        holdout_indices=acquisition_holdout,
        **common,
    )

    organism_gate = bool(organism_report["gate_pass"])
    acquisition_gate = bool(acquisition_report["gate_pass"])
    organism_thresholds = organism_report["thresholds"]
    acquisition_thresholds = acquisition_report["thresholds"]
    organism_accept = np.asarray(
        [
            organism_gate
            and organism_thresholds[label] is not None
            and margin >= float(organism_thresholds[label])
            for label, margin in zip(organism_prediction, organism_margin)
        ],
        dtype=np.bool_,
    )
    acquisition_accept = np.asarray(
        [
            acquisition_gate
            and acquisition_thresholds[label] is not None
            and margin >= float(acquisition_thresholds[label])
            for label, margin in zip(acquisition_prediction, acquisition_margin)
        ],
        dtype=np.bool_,
    )

    routed_organism = np.where(organism_accept, organism_prediction, "")
    routed_acquisition = np.where(acquisition_accept, acquisition_prediction, "")
    # Recovered truth is always preferred over a semantic prediction.
    routed_organism = np.where(organism_truth != "", organism_truth, routed_organism)
    routed_acquisition = np.where(acquisition_truth != "", acquisition_truth, routed_acquisition)
    sample_organism = routed_organism[inverse]
    sample_acquisition = routed_acquisition[inverse]
    sample_type = np.where(
        np.isin(sample_acquisition, ["histopathology", "imaging_mass_cytometry"]),
        "tissue",
        np.where(sample_acquisition != "", "cell", ""),
    )
    domain = np.asarray(
        [
            f"routed:{organism or 'unknown'}:{acquisition or 'unknown'}"
            if organism or acquisition
            else "unresolved"
            for organism, acquisition in zip(sample_organism, sample_acquisition)
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        keys=keys,
        source_id=source_ids,
        recovered=recovered,
        domain=domain,
        organism=sample_organism,
        acquisition_family=sample_acquisition,
        sample_type=sample_type,
        organism_prediction=organism_prediction[inverse],
        organism_score=organism_score[inverse].astype(np.float32),
        organism_margin=organism_margin[inverse].astype(np.float32),
        organism_accepted=organism_accept[inverse],
        acquisition_prediction=acquisition_prediction[inverse],
        acquisition_score=acquisition_score[inverse].astype(np.float32),
        acquisition_margin=acquisition_margin[inverse].astype(np.float32),
        acquisition_accepted=acquisition_accept[inverse],
    )
    report = {
        "bank": str(args.bank),
        "truth_overlay": str(args.truth_overlay),
        "output": str(args.output),
        "samples": int(len(keys)),
        "source_images": int(len(unique_ids)),
        "recovered_samples": int(recovered.sum()),
        "organism": organism_report,
        "acquisition": acquisition_report,
        "routed_sample_organisms": dict(Counter(value for value in sample_organism if value)),
        "routed_sample_acquisitions": dict(
            Counter(value for value in sample_acquisition if value)
        ),
        "known_organism_sample_fraction": float(np.mean(sample_organism != "")),
        "known_acquisition_sample_fraction": float(np.mean(sample_acquisition != "")),
        "gates_pass": bool(organism_gate and acquisition_gate),
        "gate_policy": {
            "target_calibration_precision": args.target_calibration_precision,
            "min_calibration_accepted": args.min_calibration_accepted,
            "min_holdout_precision": args.min_holdout_precision,
            "min_holdout_accepted": args.min_holdout_accepted,
            "min_routed_classes": args.min_routed_classes,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if args.enforce_gates and not report["gates_pass"]:
        raise SystemExit("BioCLIP semantic routing failed one or more holdout gates")


if __name__ == "__main__":
    main()
