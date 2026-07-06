"""
Refinement runner: run ONE config (frozen/finetune, decoder size, epochs, batch)
for a single checkpoint across datasets, on a GPU fleet. Used to push for the
best result per dataset (decoder-size / epoch / fine-tune levers).

Example (fine-tune bio_12299 across datasets on deepcad A100s):
  ~/anaconda3/envs/dinov3/bin/python -m dinov3.eval.bio_segmentation.instance_seg.scripts._run_refine \
    --ckpt bio_12299:ckpt/12299/checkpoint.pth --datasets bbbc038 livecell monuseg tissuenet conic \
    --mode finetune --feature-size 96 --epochs 50 --batch-size 8 --backbone-lr 2e-5 \
    --gpus 1,2,3,4,5,6,7 --procs-per-gpu 1 --out-tag ft_fs96
"""

from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path

PY = sys.executable
ROOT = "/mnt/huawei_deepcad/dinov3"
SEG = "/mnt/huawei_deepcad/benchmark/segmentation"
CONFIG = "config.yaml"
# (crop, stride, max_eval_images) per dataset; epochs comes from --epochs.
DSCFG = {"tissuenet": (256, 256, 300), "conic": (256, 256, 300), "bbbc038": (256, 192, 200),
         "livecell": (256, 192, 120), "monuseg": (256, 192, 60), "pannuke": (256, 256, 300)}


def data_root(ds):
    return os.path.join(SEG, "LIVECell") if ds == "livecell" else os.path.join(SEG, ds, "extracted")


def make_cmd(ds, ckpt, out_dir, a):
    crop, stride, neval = DSCFG[ds]
    cmd = [PY, "-m", "dinov3.eval.bio_segmentation.instance_seg.train",
           "--dataset", ds, "--data-root", data_root(ds), "--checkpoint", ckpt,
           "--train-config", CONFIG, "--output-dir", out_dir, "--layers", "7", "15", "23", "31",
           "--feature-size", str(a.feature_size), "--embed-proj", str(a.embed_proj),
           "--epochs", str(a.epochs), "--batch-size", str(a.batch_size),
           "--crop-size", str(crop), "--stride", str(stride),
           "--eval-every", str(a.eval_every), "--max-eval-images", str(neval), "--skip-test-eval", "--num-workers", "5"]
    if a.mode == "finetune":
        cmd += ["--finetune", "--lr", str(a.lr), "--backbone-lr", str(a.backbone_lr)]
    else:
        cmd += ["--freeze-backbone", "--lr", str(a.lr)]
    cmd += ["--aug", a.aug, "--mosaic-prob", str(a.mosaic_prob)]
    if a.tta:
        cmd += ["--tta"]
    return cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="NAME:PATH")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--mode", choices=["frozen", "finetune"], default="frozen")
    p.add_argument("--feature-size", type=int, default=64)
    p.add_argument("--embed-proj", type=int, default=512)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone-lr", type=float, default=2e-5)
    p.add_argument("--gpus", default="1,2,3,4,5,6,7")
    p.add_argument("--procs-per-gpu", type=int, default=1)
    p.add_argument("--out-tag", required=True)
    p.add_argument("--aug", choices=["none", "strong"], default="none")
    p.add_argument("--mosaic-prob", type=float, default=0.3)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--eval-every", type=int, default=10)
    a = p.parse_args()

    os.chdir(ROOT)
    name, path = a.ckpt.split(":", 1)
    slots = [g for g in a.gpus.split(",") if g] * a.procs_per_gpu
    out_root = Path("outputs/instance_seg/refine") / a.out_tag
    out_root.mkdir(parents=True, exist_ok=True)
    pending = [ds for ds in a.datasets if not (out_root / ds / "results.json").exists()]
    print(f"[refine:{a.out_tag}] ckpt={name} mode={a.mode} fs={a.feature_size} ep={a.epochs} "
          f"bs={a.batch_size} | {len(pending)} jobs slots={slots}", flush=True)

    running, free, results = {}, list(range(len(slots))), {}
    while pending or running:
        while free and pending:
            sid = free.pop(0); gpu = slots[sid]; ds = pending.pop(0)
            od = str(out_root / ds); os.makedirs(od, exist_ok=True)
            logf = open(out_root / f"{ds}.log", "w")
            proc = subprocess.Popen(make_cmd(ds, path, od, a),
                                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu),
                                    stdout=logf, stderr=subprocess.STDOUT)
            running[sid] = (proc, ds, logf, gpu)
            print(f"[refine:{a.out_tag}] launch {ds} GPU{gpu} pid {proc.pid}", flush=True)
        time.sleep(15)
        for sid, (proc, ds, logf, gpu) in list(running.items()):
            if proc.poll() is None:
                continue
            logf.close(); del running[sid]; free.append(sid)
            rj = out_root / ds / "results.json"
            results[ds] = json.load(open(rj)).get("val") if rj.exists() else None
            print(f"[refine:{a.out_tag}] done {ds} GPU{gpu}: {'ok' if results[ds] else 'FAILED '+str(proc.returncode)}", flush=True)

    print(f"\n==== refine {a.out_tag} ====", flush=True)
    for ds in a.datasets:
        v = results.get(ds)
        if v:
            print(f"  {ds:<10} AJI={v.get('AJI',float('nan')):.4f} bPQ={v.get('bPQ',float('nan')):.4f} "
                  f"AP50={v.get('AP50',float('nan')):.4f} mPQ={v.get('mPQ',float('nan')):.4f}", flush=True)
    json.dump(results, open(out_root / "results_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
