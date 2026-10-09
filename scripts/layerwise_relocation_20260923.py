#!/usr/bin/env python3
"""Is an eroded early capability destroyed, or merely relocated to an earlier block?

For each checkpoint we extract, in ONE forward pass, the protocol readout of every
requested transformer block: concat(CLS_j, patch-mean_j) L2-normalised per block,
which is byte-equivalent to running the evaluator with --feature-layers [j].
Each block is then probed with the protocol estimator.

If max_j score(L, block j) reaches score(E, final block), the late model still holds
the capability and only the readout depth changed: retention needs no new training
objective, just a deeper readout.  If it does not, the capability is genuinely gone
from the late network and a training-time intervention is required.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/mnt/huawei_deepcad/dinov3")
sys.path.insert(0, str(REPO))

from dinov3.eval.bio_frozen_eval.encoder import Dinov3CkptEncoder, extract_features  # noqa: E402
from dinov3.eval.bio_frozen_eval.registry import build_dataset  # noqa: E402

CKPT_ROOT = REPO / "outputs/02_eval_inputs/hs6_l_5t_every_05m_20260907"
TRAIN_CFG = (REPO / "outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_"
             "nosig_e15_5tb_mix1m03_5tv107_8x5090zxr_20260907/config.yaml")
BENCH = Path("/mnt/huawei_deepcad/benchmark")


class AllLayerReadout(torch.nn.Module):
    """Protocol readout for every listed block, produced by a single backbone pass."""

    def __init__(self, backbone, layers, autocast_dtype):
        super().__init__()
        self.backbone = backbone
        self.layers = list(layers)
        self.autocast_dtype = autocast_dtype

    @torch.inference_mode()
    def forward(self, images, channel_ids=None, channel_valid_mask=None):
        with torch.autocast(images.device.type, enabled=True, dtype=self.autocast_dtype):
            tokens = self.backbone.get_intermediate_layers(
                images, n=self.layers, reshape=False, return_class_token=True,
                channel_ids=channel_ids, channel_valid_mask=channel_valid_mask)
        outs = []
        for patch, cls in tokens:                       # one entry per requested block
            block = torch.cat((cls, patch.mean(dim=1)), dim=-1).float()
            outs.append(torch.nn.functional.normalize(block, dim=1))
        return torch.cat(outs, dim=-1)                  # [B, n_layers * 2048]


def probe(task, xtr, ytr, xte, yte):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score, r2_score
    if task == "regression":
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)); m.fit(xtr, ytr.astype(float))
        return float(r2_score(yte.astype(float), m.predict(xte)))
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000,
                      class_weight="balanced", C=1.0, random_state=0))
    m.fit(xtr, ytr.astype(int))
    return float(balanced_accuracy_score(yte.astype(int), m.predict(xte)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--checkpoints", default="12687,20007,29279")
    ap.add_argument("--layers", default="3,7,11,15,17,19,21,23")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--max-train", type=int, default=30000)
    ap.add_argument("--cache", type=Path,
                    default=REPO / "outputs/02_eval_cache/hs6_l5_layerwise_20260923")
    ap.add_argument("--out", type=Path,
                    default=REPO / "outputs/00_reports/hs6_l5_layerwise_relocation_20260923")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True); a.cache.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in a.layers.split(",")]
    cks = [int(x) for x in a.checkpoints.split(",")]
    tag = a.out / f"layerwise_{a.device.replace(':','')}.json"
    rep = {}

    for ds in [d.strip() for d in a.datasets.split(",") if d.strip()]:
        t0 = time.time()
        banks = {}
        task = None
        for ck in cks:
            enc = Dinov3CkptEncoder(checkpoint=CKPT_ROOT / str(ck) / "checkpoint.pth",
                                    train_config=TRAIN_CFG, device=a.device, n_last_blocks=1,
                                    use_avgpool=True, autocast_dtype=torch.bfloat16,
                                    image_size=224, resize_size=256, channel_policy="auto",
                                    channel_tta_samples=8, channel_policy_seed=0)
            depth = len(enc.model.backbone.blocks)
            assert all(0 <= j < depth for j in layers), f"layers out of range for depth {depth}"
            enc.model = AllLayerReadout(enc.model.backbone, layers, torch.bfloat16).to(a.device).eval()
            for split in ("train", "test"):
                dset, task = build_dataset(ds, split, None, None, benchmark_root=BENCH)
                out = a.cache / f"{ds}_{ck}_{split}_L{'-'.join(map(str,layers))}.npz"
                f, y = extract_features(dset, enc, out, a.batch_size, a.num_workers, False,
                                        f"ck{ck}-{split}", save_features=True, save_paths=False)
                banks[(ck, split)] = (f, y)
            del enc
            torch.cuda.empty_cache()

        ytr, yte = banks[(cks[0], "train")][1], banks[(cks[0], "test")][1]
        n = len(ytr)
        idx = np.arange(n)
        if n > a.max_train:
            idx = np.random.default_rng(0).choice(n, a.max_train, replace=False); idx.sort()
        print(f"\n=== {ds} [{task}] n_train={n} (probed {len(idx)}) n_test={len(yte)} ===", flush=True)
        res = {}
        for ck in cks:
            ftr, fte = banks[(ck, "train")][0], banks[(ck, "test")][0]
            row = {}
            for li, j in enumerate(layers):
                sl = slice(li * 2048, (li + 1) * 2048)
                row[str(j)] = probe(task, ftr[idx, sl], ytr[idx], fte[:, sl], yte)
            res[str(ck)] = row
            print(f"  ck{ck}: " + "  ".join(f"L{j}={row[str(j)]:.4f}" for j in layers), flush=True)
        best = {c: max(res[str(c)].values()) for c in cks}
        final = {c: res[str(c)][str(layers[-1])] for c in cks}
        print(f"  final-block: " + "  ".join(f"ck{c}={final[c]:.4f}" for c in cks))
        print(f"  best-block : " + "  ".join(f"ck{c}={best[c]:.4f}" for c in cks))
        print(f"  RELOCATED? best(last ck) {best[cks[-1]]:.4f} vs final(first ck) {final[cks[0]]:.4f}"
              f"  -> {'YES' if best[cks[-1]] >= final[cks[0]] else 'NO'}", flush=True)
        rep[ds] = {"task": task, "layers": layers, "checkpoints": cks, "n_train": int(n),
                   "n_probed": int(len(idx)), "n_test": int(len(yte)), "scores": res,
                   "best_per_ck": {str(k): v for k, v in best.items()},
                   "final_per_ck": {str(k): v for k, v in final.items()},
                   "relocated": bool(best[cks[-1]] >= final[cks[0]]), "seconds": time.time() - t0}
        tag.write_text(json.dumps(rep, indent=1))
    print("\nwrote", tag)


if __name__ == "__main__":
    main()
