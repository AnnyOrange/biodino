#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""CPU smoke test for the #1 dual-route stem + #3 content-derived channel id.

Run with the dinov3 conda env:
    /home/lxy/miniconda3/envs/dinov3/bin/python scripts/smoke_test_dualroute.py

Checks (no GPU / no distributed needed):
  1. token count == 1 + n_storage + H'*W'  (RGB-shaped; no channel explosion)
  2. forward + backward produce no NaN; stem params get gradients
  3. per-sample routing: 1ch / 3ch-correlated -> RGB; 3ch-rand / 5ch -> pool
  4. channel_valid_mask=None on a 3ch-correlated input == standard PatchEmbed
  5. checkpoint remap: standard patch_embed.proj -> rgb (exact) + pool (mean)
"""

import sys
import traceback

import torch

from dinov3.models.vision_transformer import DinoVisionTransformer
from dinov3.checkpointer.checkpointer import _remap_dualroute_patch_embed

P = 16
IMG = 32           # -> 2x2 = 4 patch tokens
HP = WP = IMG // P
N_PATCH = HP * WP
N_STORAGE = 4
D = 64             # tiny trunk; stem logic is size-independent
FAILED = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}")
    if not cond:
        FAILED.append(name)


def build_model():
    m = DinoVisionTransformer(
        img_size=IMG,
        patch_size=P,
        embed_dim=D,
        depth=2,
        num_heads=2,
        n_storage_tokens=N_STORAGE,
        stem_type="dualroute",
    )
    m.init_weights()
    m.eval()
    return m


def make_mixed_batch():
    """4 samples padded to C=5 with a valid mask: [1ch, 3ch-corr, 3ch-rand, 5ch]."""
    torch.manual_seed(0)
    C = 5
    x = torch.zeros(4, C, IMG, IMG)
    valid = torch.zeros(4, C, dtype=torch.bool)

    # sample 0: 1 channel (grayscale)
    x[0, 0] = torch.randn(IMG, IMG)
    valid[0, 0] = True
    # sample 1: 3 highly-correlated channels (joint colour)
    base = torch.randn(IMG, IMG)
    for c in range(3):
        x[1, c] = base + 0.01 * torch.randn(IMG, IMG)
    valid[1, :3] = True
    # sample 2: 3 independent channels
    x[2, :3] = torch.randn(3, IMG, IMG)
    valid[2, :3] = True
    # sample 3: 5 independent channels
    x[3, :5] = torch.randn(5, IMG, IMG)
    valid[3, :5] = True

    channel_ids = torch.arange(C).unsqueeze(0).expand(4, C).clone()
    return x, channel_ids, valid


def test_forward_backward(model):
    print("[1/5] forward + token shape")
    x, cids, vmask = make_mixed_batch()
    x.requires_grad_(False)
    masks = torch.zeros(4, N_PATCH, dtype=torch.bool)
    masks[0, 0] = True  # exercise the masked (iBOT-like) path too
    out = model.forward_features(x, masks=masks, channel_ids=cids, channel_valid_mask=vmask)
    patch = out["x_norm_patchtokens"]
    cls = out["x_norm_clstoken"]
    check("patch token count == H'*W'", patch.shape[1] == N_PATCH)
    check("cls token dim == D", cls.shape[-1] == D)
    check("no NaN in patch tokens", not torch.isnan(patch).any())
    check("no NaN in cls token", not torch.isnan(cls).any())

    print("[2/5] backward + stem grads")
    loss = patch.float().pow(2).mean() + cls.float().pow(2).mean()
    loss.backward()
    stem = model.patch_embed
    grads = {
        "rgb.proj.weight": stem.rgb.proj.weight.grad,
        "pool.proj.weight": stem.pool.proj.weight.grad,
        "pool.query": stem.pool.query.grad,
        "descriptor.mlp[0].weight": stem.pool.descriptor.mlp[0].weight.grad,
    }
    for n, g in grads.items():
        check(f"grad exists & finite: {n}", g is not None and torch.isfinite(g).all())


def test_routing(model):
    print("[3/5] per-sample routing")
    x, _, vmask = make_mixed_batch()
    is_rgb, rgb_in = model.patch_embed._route_and_rgb_input(x, vmask)
    check("sample 0 (1ch) -> RGB", bool(is_rgb[0]))
    check("sample 1 (3ch corr) -> RGB", bool(is_rgb[1]))
    check("sample 2 (3ch rand) -> pool", not bool(is_rgb[2]))
    check("sample 3 (5ch) -> pool", not bool(is_rgb[3]))
    check("rgb_in is (N,3,H,W)", tuple(rgb_in.shape) == (4, 3, IMG, IMG))


def test_fsdp_safety(model):
    # The bug that caused the NCCL timeout: data-dependent routing meant a rank
    # whose batch was all-RGB never ran the pool stem -> its FSDP collectives
    # diverged. The fix runs BOTH stems every step. Proxy check (single-process):
    # in an all-RGB batch the pool params must STILL receive a (zero) gradient,
    # proving the backward traverses the pool subgraph -> reduce_scatter fires.
    print("[3b] FSDP-safety: all-RGB batch still yields pool grads")
    model.zero_grad(set_to_none=True)
    torch.manual_seed(3)
    x = torch.zeros(3, 3, IMG, IMG)
    valid = torch.zeros(3, 3, dtype=torch.bool)
    for i in range(3):  # all grayscale 1ch -> all route RGB
        x[i, 0] = torch.randn(IMG, IMG)
        valid[i, 0] = True
    cids = torch.arange(3).unsqueeze(0).expand(3, 3).clone()
    is_rgb, _ = model.patch_embed._route_and_rgb_input(x, valid)
    check("batch is all-RGB route", bool(is_rgb.all()))
    out = model.forward_features(x, channel_ids=cids, channel_valid_mask=valid)
    out["x_norm_patchtokens"].float().pow(2).mean().backward()
    pg = model.patch_embed.pool.proj.weight.grad
    qg = model.patch_embed.pool.query.grad
    dg = model.patch_embed.pool.descriptor.mlp[0].weight.grad
    check("pool.proj grad EXISTS in all-RGB batch", pg is not None and torch.isfinite(pg).all())
    check("pool.query grad EXISTS in all-RGB batch", qg is not None and torch.isfinite(qg).all())
    check("descriptor grad EXISTS in all-RGB batch", dg is not None and torch.isfinite(dg).all())


def test_none_mask_equiv(model):
    print("[4/5] valid_mask=None on 3ch-correlated == standard PatchEmbed")
    torch.manual_seed(1)
    base = torch.randn(1, 1, IMG, IMG)
    x3 = (base + 0.01 * torch.randn(1, 3, IMG, IMG)).clone()  # correlated 3ch
    with torch.no_grad():
        out_full = model.patch_embed(x3, channel_ids=None, channel_valid_mask=None)
        out_rgb = model.patch_embed.rgb(x3)
    check("routed to RGB & equals rgb stem", torch.allclose(out_full, out_rgb, atol=1e-6))


def test_ckpt_remap(model):
    print("[5/5] checkpoint remap (standard stem -> dual-route)")
    # Mirror real usage: init_fsdp_model_from_checkpoint receives the student
    # ModuleDict, so state-dict keys carry the "backbone." prefix.
    model_state = torch.nn.ModuleDict({"backbone": model}).state_dict()
    has_rgb = any(k.endswith("patch_embed.rgb.proj.weight") for k in model_state)
    has_pool = any(k.endswith("patch_embed.pool.proj.weight") for k in model_state)
    check("model has rgb.proj + pool.proj keys", has_rgb and has_pool)

    torch.manual_seed(2)
    w3 = torch.randn(D, 3, P, P)
    b = torch.randn(D)
    chkpt = {
        "backbone.patch_embed.proj.weight": w3.clone(),
        "backbone.patch_embed.proj.bias": b.clone(),
        "backbone.cls_token": torch.randn(1, 1, D),  # unrelated key, must survive
    }
    _remap_dualroute_patch_embed(chkpt, model_state)
    check("rgb.proj.weight = exact copy",
          "backbone.patch_embed.rgb.proj.weight" in chkpt
          and torch.equal(chkpt["backbone.patch_embed.rgb.proj.weight"], w3))
    check("rgb.proj.bias = exact copy",
          torch.equal(chkpt.get("backbone.patch_embed.rgb.proj.bias"), b))
    check("pool.proj.weight = channel-mean",
          "backbone.patch_embed.pool.proj.weight" in chkpt
          and torch.allclose(chkpt["backbone.patch_embed.pool.proj.weight"],
                             w3.mean(dim=1, keepdim=True)))
    check("pool.proj.weight shape [D,1,P,P]",
          tuple(chkpt["backbone.patch_embed.pool.proj.weight"].shape) == (D, 1, P, P))
    check("old patch_embed.proj.* removed",
          "backbone.patch_embed.proj.weight" not in chkpt
          and "backbone.patch_embed.proj.bias" not in chkpt)
    check("unrelated key survived", "backbone.cls_token" in chkpt)

    # no-op on a non-dual-route model_state
    chkpt2 = {"backbone.patch_embed.proj.weight": w3.clone()}
    _remap_dualroute_patch_embed(chkpt2, {"backbone.patch_embed.proj.weight": w3})
    check("no-op when model is not dual-route",
          "backbone.patch_embed.rgb.proj.weight" not in chkpt2)


def main():
    print("=== dual-route stem smoke test (CPU) ===")
    try:
        model = build_model()
        test_forward_backward(model)
        test_routing(model)
        test_fsdp_safety(model)
        test_none_mask_equiv(model)
        test_ckpt_remap(model)
    except Exception:
        traceback.print_exc()
        print("\nRESULT: ERROR (exception above)")
        return 2

    print()
    if FAILED:
        print(f"RESULT: {len(FAILED)} FAILED -> {FAILED}")
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
