"""Campaign-facing API; callers own checkpoint admission and sync gates."""
from collections import defaultdict
import time

import numpy as np

from .runner import evaluate_pair, extract_dinov3_descriptors


def evaluate_registration(backbone, pairs, layers, device, max_side=512, ratio=.9,
                          threshold_fraction=.005, seed=0, descriptor_cache=None):
    """Real dense DINO correspondence fit and specimen-macro heldout scoring.

    An externally supplied cache can be reused across ratio/threshold candidates.
    Its keys include backbone instance, layers and resize so model/config changes
    cannot accidentally reuse another checkpoint's descriptors. Ground-truth
    annotations are NEVER used in feature extraction, matching or affine fitting.
    """
    cache = descriptor_cache if descriptor_cache is not None else {}
    layer_key = layers if isinstance(layers, int) else tuple(layers)
    results = []
    for pair in pairs:
        started = time.monotonic()
        data = []
        for path in (pair.source_image, pair.target_image):
            key = (id(backbone), str(path), layer_key, max_side)
            if key not in cache:
                cache[key] = extract_dinov3_descriptors(backbone, path, layers,
                    device=device, max_side=max_side)[:2]
            data.append(cache[key])
        result = evaluate_pair(pair, data[0][0], data[1][0], data[0][1], data[1][1],
            ratio=ratio, threshold_fraction=threshold_fraction, seed=seed,
            matching_backend="torch", device=device)
        result["total_seconds"] = time.monotonic() - started
        results.append(result)
    if not results:
        raise ValueError("No registration pairs to score")
    grouped = defaultdict(list)
    for result in results:
        grouped[result["group"]].append(result["metrics"]["median_rtre"])
    group_scores = {group: float(np.mean(scores)) for group, scores in sorted(grouped.items())}
    metrics = {"mean_group_median_rtre": float(np.mean(list(group_scores.values()))),
               "mean_pair_median_rtre": float(np.mean([r["metrics"]["median_rtre"] for r in results])),
               "mean_pair_robustness": float(np.mean([r["metrics"]["robustness"] for r in results])),
               "failed_pair_fraction": float(np.mean([not r["success"] for r in results])),
               "pairs": len(results), "groups": len(group_scores)}
    return {"metrics": metrics, "group_scores": group_scores, "pair_results": results,
            "parameters": {"layers": layers, "max_side": max_side, "ratio": ratio,
                           "threshold_fraction": threshold_fraction, "seed": seed},
            "task": "native_affine_registration", "primary_metric": "mean_group_median_rtre",
            "direction": "lower_is_better"}
