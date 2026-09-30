#!/usr/bin/env python3
"""Fork a vanilla full-state checkpoint (model+optimizer+EMA) into a recovery-enabled run WITHOUT restarting the optimizer.

The vanilla consolidated ``checkpoint.pth`` lacks two groups of keys that an SSLMetaArch with
``gram.use_loss=true`` + ``recovery.enabled=true`` expects:
  gram_teacher.backbone.*      (frozen anchor)   <- copied from the checkpoint's own EMA teacher
                                                    (in-run anchor) or from --anchor-teacher export
  recovery_loss.{local,global_stream}.*          <- fresh zero buffers (warmup recalibrates them)
Optimizer state and iteration are copied untouched, so AdamW moments/steps stay continuous.
Also writes anchor/teacher_checkpoint.pth (the EMA teacher at the fork step, or the supplied anchor) for
``gram.ckpt`` / ``student.resume_from_teacher_chkpt``.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import torch

RECOVERY_BUFFERS = {"xx": lambda d: (d + 1, d + 1), "xr": lambda d: (d + 1, d), "delta": lambda d: (d + 1, d),
                    "mean": lambda d: (d,), "second": lambda d: (d,), "floor": lambda d: (d,), "monitor": lambda d: (d,), "dual": lambda d: (d,)}

def sha256(p: Path, chunk=1 << 26) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True, help="vanilla ckpt/<it>/checkpoint.pth")
    ap.add_argument("--out-root", type=Path, required=True, help="writes <out-root>/ckpt/<it>/checkpoint.pth and <out-root>/anchor/teacher_checkpoint.pth")
    ap.add_argument("--anchor-teacher", type=Path, default=None, help="optional teacher_checkpoint.pth to use as the frozen anchor instead of the fork-step EMA teacher")
    ap.add_argument("--no-hash", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    ck = torch.load(a.source, map_location="cpu", weights_only=False, mmap=True)
    assert set(ck.keys()) >= {"iteration", "model", "optimizer"}, ck.keys()
    it = int(ck["iteration"]); model = dict(ck["model"])
    print(f"loaded {a.source} iteration={it} model_keys={len(model)} optimizer_states={len(ck['optimizer']['state'])} ({time.time()-t0:.0f}s)")
    assert not any(k.startswith("gram_teacher.") or k.startswith("recovery_loss.") for k in model), "source already has recovery/anchor keys"

    teacher = {k[len("teacher."):]: v for k, v in model.items() if k.startswith("teacher.")}
    assert teacher, "no teacher.* keys in checkpoint"
    if a.anchor_teacher is not None:
        anc = torch.load(a.anchor_teacher, map_location="cpu", weights_only=False, mmap=True)
        anchor_sd = anc["teacher"] if "teacher" in anc else anc
        anchor_desc = str(a.anchor_teacher)
    else:
        anchor_sd = teacher; anchor_desc = f"EMA teacher at iteration {it} (in-run anchor)"
    backbone = {k: v for k, v in anchor_sd.items() if k.startswith("backbone.")}
    assert backbone, "anchor has no backbone.* keys"
    for k, v in backbone.items():
        model[f"gram_teacher.{k}"] = v.detach().clone()
    d = int(anchor_sd["backbone.cls_token"].shape[-1])
    n_rec = 0
    for stream in ("local", "global_stream"):
        for name, shape in RECOVERY_BUFFERS.items():
            model[f"recovery_loss.{stream}.{name}"] = torch.zeros(shape(d), dtype=torch.float32); n_rec += 1
        model[f"recovery_loss.{stream}.steps"] = torch.zeros((), dtype=torch.long); n_rec += 1
    print(f"injected gram_teacher.backbone.* ({len(backbone)} keys, from {anchor_desc}) and {n_rec} recovery buffers (dim {d}); model_keys now {len(model)}")

    out_ckpt = a.out_root / "ckpt" / str(it); out_ckpt.mkdir(parents=True, exist_ok=True)
    torch.save({"iteration": it, "model": model, "optimizer": ck["optimizer"]}, out_ckpt / "checkpoint.pth")
    anchor_dir = a.out_root / "anchor"; anchor_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"teacher": {k: v.detach().clone() for k, v in anchor_sd.items()}}, anchor_dir / "teacher_checkpoint.pth")
    # verify by reloading
    chk = torch.load(out_ckpt / "checkpoint.pth", map_location="cpu", weights_only=False, mmap=True)
    assert chk["iteration"] == it and len(chk["model"]) == len(model) and len(chk["optimizer"]["state"]) == len(ck["optimizer"]["state"])
    steps = [float(v["step"]) for v in chk["optimizer"]["state"].values() if isinstance(v, dict) and "step" in v]
    manifest = {"source": str(a.source), "source_sha256": None if a.no_hash else sha256(a.source), "iteration": it,
                "anchor": anchor_desc, "n_model_keys_source": len(ck["model"]), "n_model_keys_out": len(model),
                "n_gram_teacher_keys": len(backbone), "n_recovery_buffers": n_rec, "embed_dim": d,
                "optimizer_states": len(chk["optimizer"]["state"]), "adam_step_min_max": [min(steps), max(steps)] if steps else None,
                "out_checkpoint": str(out_ckpt / "checkpoint.pth"), "out_anchor": str(anchor_dir / "teacher_checkpoint.pth"),
                "note": "optimizer and iteration copied verbatim; recovery buffers zero (warmup recalibrates); gram_teacher = frozen anchor"}
    (a.out_root / "fork_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1)); print(f"done in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
