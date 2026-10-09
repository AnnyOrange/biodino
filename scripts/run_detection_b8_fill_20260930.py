#!/usr/bin/env python3
"""Fill v4 detection-proxy gaps (B8, official CoNIC split) for the 1TB / 5TB GRAM report points.

User instruction 2026-09-30: check that CoNIC is tested with the official split everywhere and fill
what is missing.  Historical CoNIC detection written before commit 8529307 (2026-09-09) used the
legacy random patch-level 80/10/10 split (pos_weight 1.3915689; every test patch shares its source
image with training patches), so those values are excluded and the points are re-tested here with
the frozen v4 settings: center_probe, batch 8, 224 px, 5 epochs, lr 1e-3, seed 0, bf16,
conic-split-protocol official-baseline-fold0-nested-v1.  Code comes from the hash-verified MoNuSeg
evaluator snapshot (detection code identical to the v4 evaluator).  Sites that hold checkpoints
locally run the same script; their dataset files must match the NFS reference (relative path and
SHA-256 of every file the three detection datasets read).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

DATASETS = ("bbbc038", "conic", "livecell")
CONIC_SPLIT = "official-baseline-fold0-nested-v1"
OFFICIAL_CONIC_POS_WEIGHT = 1.3762682128


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def load_site(path) -> dict:
    site = json.loads(Path(path).read_text())
    sys.path.insert(0, site["snapshot"])
    os.environ["DINOV3_CODE_ROOT"] = site["snapshot"]
    return site


def membership(site: dict) -> dict:
    """Relative path + SHA-256 of every file each detection split reads."""
    import dinov3.eval.bio_detection.center_probe as cp
    benchmark = Path(site["benchmark"]).resolve()

    def rel(p):
        return str(Path(p).resolve().relative_to(benchmark))

    out = {}
    for dataset in DATASETS:
        for split in ("train", "val", "test"):
            ds = cp.build_center_dataset(dataset, benchmark, split, 224, 0, 0, conic_split_protocol=CONIC_SPLIT)
            base = getattr(ds, "dataset", ds)
            files = []
            if dataset == "livecell":
                files = [s[0] for s in base.samples] + [cp.get_livecell_paths(str(benchmark / "segmentation/LIVECell"), split)[0]]
                entry = {"samples": [[rel(p), w, h, len(c)] for p, w, h, c in base.samples]}
            elif dataset == "conic":
                entry = {"indices": [int(i) for i in base.indices]}
                files = [base.images.filename, base.labels.filename]
            else:
                pairs = []
                for img, mask in zip(base.img_paths, base.mask_paths):
                    masks = sorted(Path(mask).rglob("*")) if Path(mask).is_dir() else [Path(mask)]
                    pairs.append([rel(img), [rel(m) for m in masks if m.is_file()]])
                    files += [img] + [m for m in masks if m.is_file()]
                entry = {"pairs": pairs}
            entry["sha256"] = {rel(f): sha256(f) for f in sorted(set(map(str, files)))}
            out[f"{dataset}/{split}"] = entry
    return out


def reference(args) -> None:
    site = load_site(args.site)
    Path(args.out).write_text(json.dumps(membership(site), sort_keys=True) + "\n")
    print("reference", args.out, sha256(args.out), flush=True)


def prepare(args) -> None:
    site = load_site(args.site)
    output = Path(site["output"])
    if (output / "campaign_manifest.json").exists():
        raise FileExistsError("Existing campaign; refusing overwrite")
    source = json.loads((Path(site["snapshot"]) / "source_snapshot.json").read_text())
    for relative, digest in source["files"].items():
        if sha256(Path(site["snapshot"]) / relative) != digest:
            raise RuntimeError("snapshot copy differs: " + relative)
    ref = Path(site["reference"])
    if membership(site) != json.loads(ref.read_text()):
        raise RuntimeError("site detection data differ from the NFS reference")
    missing = [a["path"] for a in site["assets"] if not Path(a["path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"missing checkpoints: {missing[:3]}")
    tasks = [dict(key=f"{a['arm']}_ck{a['checkpoint_id']}__{d}", asset=a, dataset=d)
             for a in site["assets"] for d in a["datasets"]]
    save(output / "campaign_manifest.json", dict(
        protocol_id="bio-eval-union-v4-detection-proxy-b8", conic_split_protocol=CONIC_SPLIT,
        user_authorization="2026-09-30: CoNIC official split audit and gap fill",
        site=site["name"], source_snapshot_sha256=source["sha256"], source_snapshot_path=site["snapshot"],
        data_reference=dict(path=str(ref), sha256=sha256(ref), matched=True),
        runner_sha256=sha256(__file__), tasks=tasks, created_unix=time.time()))
    print(json.dumps(dict(output=str(output), tasks=len(tasks))), flush=True)


def validate(result: dict, dataset: str) -> None:
    for key, value in (("batch_size", 8), ("image_size", 224), ("epochs", 5), ("seed", 0), ("dataset", dataset)):
        if result.get(key) != value:
            raise ValueError(f"protocol mismatch {key}={result.get(key)}")
    if dataset == "conic":
        if result.get("conic_split_protocol") != CONIC_SPLIT or abs(result["pos_weight"] - OFFICIAL_CONIC_POS_WEIGHT) > 1e-5:
            raise ValueError("CoNIC not on the official split")
    if not math.isfinite(float(result["test_patch_f1"])):
        raise ValueError("nonfinite test_patch_f1")


def gpu_free_mib(gpu: int) -> int:
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).split()
    return int(rows[gpu])


def worker(args) -> None:
    site = load_site(args.site)
    output = Path(site["output"])
    manifest = json.loads((output / "campaign_manifest.json").read_text())
    for name in ("claims", "done", "failed"):
        (output / name).mkdir(exist_ok=True)
    active = {}
    while True:
        for gpu, (key, process, cell) in list(active.items()):
            if process.poll() is None:
                continue
            try:
                if process.returncode:
                    raise RuntimeError(f"exit {process.returncode}")
                result = json.loads((cell / "results_bio_detection.json").read_text())
                validate(result, key.rsplit("__", 1)[1])
                save(output / "done" / f"{key}.json", dict(status="VALID_COMPLETE", host=site["name"], gpu=gpu,
                     result_sha256=sha256(cell / "results_bio_detection.json"), time=time.time()))
                print(time.strftime("%FT%TZ", time.gmtime()), "DONE", key, flush=True)
            except Exception as error:
                save(output / "failed" / f"{key}.json", dict(error=str(error), host=site["name"], gpu=gpu, time=time.time()))
                print(time.strftime("%FT%TZ", time.gmtime()), "FAILED", key, error, flush=True)
            del active[gpu]
        pending = [t for t in manifest["tasks"] if not (output / "claims" / t["key"]).exists()]
        if not pending and not active:
            break
        for gpu in site["gpus"]:
            if gpu in active or not pending or gpu_free_mib(gpu) < site.get("min_free_mib", 12000):
                continue
            task = pending.pop(0)
            try:
                (output / "claims" / task["key"]).mkdir()
            except FileExistsError:
                continue
            asset, dataset = task["asset"], task["dataset"]
            cell = output / "cells" / task["key"]
            if cell.exists():
                shutil.rmtree(cell)
            cell.mkdir(parents=True)
            cmd = [sys.executable, "-B", "-m", "dinov3.eval.bio_detection.center_probe",
                   "--checkpoint", asset["path"], "--train-config", asset["config"],
                   "--benchmark-root", site["benchmark"], "--dataset", dataset, "--output-dir", str(cell),
                   "--batch-size", "8", "--num-workers", "2", "--image-size", "224", "--epochs", "5",
                   "--lr", "0.001", "--autocast-dtype", "bf16", "--channel-policy", "auto", "--seed", "0",
                   "--max-samples-per-split", "0", "--conic-split-protocol", CONIC_SPLIT]
            save(cell / "v4_invocation.json", dict(host=site["name"], gpu=gpu, command=cmd, asset=asset,
                 checkpoint_bytes=Path(asset["path"]).stat().st_size, start_unix=time.time()))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=site["snapshot"],
                       OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
            log = (cell / "run.log").open("w")
            active[gpu] = (task["key"], subprocess.Popen(cmd, cwd=site["snapshot"], env=env, stdout=log,
                                                         stderr=subprocess.STDOUT, start_new_session=True), cell)
            print(time.strftime("%FT%TZ", time.gmtime()), "START", task["key"], "gpu", gpu, flush=True)
            time.sleep(20)
        time.sleep(10)
    print("QUEUE_FINISHED", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("reference", "prepare", "worker"))
    p.add_argument("--site", required=True)
    p.add_argument("--out")
    a = p.parse_args()
    {"reference": reference, "prepare": prepare, "worker": worker}[a.mode](a)
