"""
CPU smoke test for the instance_seg engine (no GPU, no datasets).

Run:
    python -m dinov3.eval.bio_segmentation.instance_seg.smoke_test

Checks:
    1. DINOHoVerNet (frozen, stub backbone) forward → correct NP/HV/TP shapes,
       with both even-4 taps and an 8-layer tap set.
    2. One backward step through the HoVerNet loss updates the decoder.
    3. postproc splits TWO overlapping disks (one connected component) into two
       instances — the thing connected-components cannot do.
    4. accumulate_instance_metrics returns finite numbers.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..metrics import accumulate_instance_metrics
from .decoder import assign_buckets
from .losses import HoVerNetLoss
from .model import DINOHoVerNet
from .postproc import postprocess
from .targets import gen_instance_hv_map, make_targets


class StubBackbone(nn.Module):
    """Minimal stand-in exposing the DINOv3 feature-access API."""

    def __init__(self, embed_dim: int = 64, patch_size: int = 16, depth: int = 12):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.n_blocks = depth
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.ModuleList(nn.Linear(embed_dim, embed_dim) for _ in range(depth))
        self.norm = nn.LayerNorm(embed_dim)

    def get_intermediate_layers(self, x, n=1, reshape=True, return_class_token=False):
        feat = self.proj(x)  # [B, D, Hp, Wp]
        indices = range(self.n_blocks - n, self.n_blocks) if isinstance(n, int) else list(n)
        # Vary slightly per layer so taps are not identical tensors.
        return tuple(feat * (1.0 + 0.01 * k) for k, _ in enumerate(indices))


def _two_disks(h=96, w=96):
    """Two overlapping disks → single FG blob but two GT instances."""
    yy, xx = np.mgrid[0:h, 0:w]
    inst = np.zeros((h, w), dtype=np.int32)
    for inst_id, (cy, cx) in enumerate([(48, 36), (48, 60)], start=1):
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= 16 ** 2
        inst[disk] = inst_id
    return inst


def test_shapes():
    for layers, num_types in ([2, 5, 8, 11], 4), ([1, 3, 5, 6, 7, 9, 10, 11], 0):
        backbone = StubBackbone(embed_dim=64, patch_size=16, depth=12)
        model = DINOHoVerNet(backbone, layers=layers, num_types=num_types,
                             freeze_backbone=True, feature_size=8, embed_proj=32)
        img = torch.randn(2, 3, 64, 64)
        out = model(img)
        assert out["np"].shape == (2, 2, 64, 64), out["np"].shape
        assert out["hv"].shape == (2, 2, 64, 64), out["hv"].shape
        if num_types > 0:
            assert out["tp"].shape == (2, num_types, 64, 64), out["tp"].shape
        else:
            assert out["tp"] is None
    print("[ok] forward shapes (even-4 and 8-layer taps)")


def test_buckets():
    assert assign_buckets(4) == [[0], [1], [2], [3]]
    assert assign_buckets(8) == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert assign_buckets(2) == [[0], [1], [1], [1]]
    print("[ok] tap→bucket assignment")


def test_backward():
    backbone = StubBackbone(embed_dim=32, patch_size=16, depth=12)
    model = DINOHoVerNet(backbone, layers=[2, 5, 8, 11], num_types=3,
                         freeze_backbone=True, feature_size=8, embed_proj=16)
    img = torch.randn(1, 3, 64, 64)
    out = model(img)
    target = {
        "np": torch.randint(0, 2, (1, 64, 64)),
        "hv": torch.randn(1, 2, 64, 64).clamp(-1, 1),
        "tp": torch.randint(0, 3, (1, 64, 64)),
    }
    loss, comps = HoVerNetLoss(num_types=3)(out, target)
    loss.backward()
    grads = [p.grad for p in model.decoder.parameters() if p.grad is not None]
    assert grads, "no decoder gradients"
    assert np.isfinite(comps["total"]), comps
    # frozen backbone must have no gradients
    assert all(p.grad is None for p in model.backbone.parameters())
    print(f"[ok] backward step (loss={comps['total']:.3f}, components={list(comps)})")


def test_partial_unfreeze():
    backbone = StubBackbone(embed_dim=32, patch_size=16, depth=12)
    model = DINOHoVerNet(
        backbone,
        layers=[2, 5, 8, 11],
        num_types=0,
        freeze_backbone=False,
        trainable_backbone_blocks=2,
        feature_size=8,
        embed_proj=16,
    )
    assert model.backbone_mode == "last2"
    assert all(not p.requires_grad for p in backbone.proj.parameters())
    assert all(not p.requires_grad for block in backbone.blocks[:-2] for p in block.parameters())
    assert all(p.requires_grad for block in backbone.blocks[-2:] for p in block.parameters())
    assert all(p.requires_grad for p in backbone.norm.parameters())
    model.train()
    assert not backbone.blocks[-3].training
    assert backbone.blocks[-2].training and backbone.blocks[-1].training
    print("[ok] partial unfreeze selects last 2 blocks + final norm")


def test_postproc_splits():
    inst_gt = _two_disks()
    # FG is a single connected component:
    from scipy.ndimage import label as nd_label
    n_cc = nd_label(inst_gt > 0)[1]
    assert n_cc == 1, f"expected 1 connected component, got {n_cc}"

    hv = gen_instance_hv_map(inst_gt)                      # [2,H,W]
    fg = (inst_gt > 0).astype(np.float32)
    np_logits = np.stack([(1 - fg) * 10.0, fg * 10.0])    # confident logits
    pred_inst, pred_sem = postprocess(np_logits, hv, tp_logits=None)

    n_pred = len([i for i in np.unique(pred_inst) if i != 0])
    assert n_pred >= 2, f"watershed failed to split touching disks (got {n_pred})"

    m = accumulate_instance_metrics([pred_inst], [inst_gt])
    assert all(np.isfinite(v) for v in m.values()), m
    print(f"[ok] postproc split 1 blob → {n_pred} instances; bPQ={m['bPQ']:.3f} AJI={m['AJI']:.3f}")


def test_targets():
    inst = _two_disks()
    sem = (inst > 0).astype(np.int64) * 2  # pretend class 2
    sem[0, 0] = 255                         # padded pixel
    t = make_targets(inst, sem, ignore_index=255)
    assert t["np"].shape == (96, 96) and t["hv"].shape == (2, 96, 96)
    assert int(t["np"][0, 0]) == 255          # padding propagated to NP ignore
    assert float(t["hv"].abs().max()) <= 1.0 + 1e-5
    print("[ok] target generation (NP/HV/TP, ignore propagation)")


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    test_buckets()
    test_targets()
    test_shapes()
    test_backward()
    test_partial_unfreeze()
    test_postproc_splits()
    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
