#!/usr/bin/env python3
"""Install or verify the isolated hs6 evaluation environment from a pinned spec."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "environments/hs6_protocol_v2.txt"
MODULES = {"scikit-learn": "sklearn", "Pillow": "PIL", "opencv-python": "cv2",
           "scikit-image": "skimage", "pycocotools": "pycocotools.coco",
           "PyYAML": "yaml"}
THREAD_ENV = {name: "1" for name in
              ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
for name, value in THREAD_ENV.items():
    os.environ.setdefault(name, value)


def requirements():
    return dict(line.split("==") for line in SPEC.read_text().splitlines()
                if line.strip() and not line.startswith("#"))


def fingerprint(require_cuda=False):
    if any(os.environ.get(name) != value for name, value in THREAD_ENV.items()):
        raise RuntimeError("Evaluation numerical thread variables must match the pinned specification")
    versions = {}
    for name, expected in requirements().items():
        found = importlib.metadata.version(name)
        if found.split("+")[0] != expected:
            raise RuntimeError(f"{name}: expected {expected}, found {found}")
        if not name.startswith("nvidia-"):
            importlib.import_module(MODULES.get(name, name))
        versions[name] = found
    import torch
    import torchvision

    a = torch.tensor([[0., 0., 2., 2.]])
    if torchvision.ops.nms(a, torch.ones(1), .5).tolist() != [0]:
        raise RuntimeError("Native torchvision CPU NMS failed")
    devices = []
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required on benchmark compute machines")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        a = torch.tensor([[0., 0., 2., 2.]], device=device)
        if torchvision.ops.nms(a, torch.ones(1, device=device), .5).tolist() != [0]:
            raise RuntimeError("Native torchvision CUDA NMS failed")
        x = torch.ones((16, 16), device=device, dtype=torch.bfloat16)
        if not torch.isfinite(x @ x).all():
            raise RuntimeError("BF16 CUDA matrix multiply failed")
        devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    cuda_packages = {name: importlib.metadata.version(name) for name in requirements()
                     if name.startswith("nvidia-") or name == "triton"}
    compatibility = {"python_minor": platform.python_version_tuple()[:2], "versions": versions,
                     "torch_cuda": torch.version.cuda, "cuda_packages": cuda_packages,
                     "thread_environment": {name: os.environ[name] for name in THREAD_ENV},
                     "torch_cpu_threads": torch.get_num_threads()}
    encoded = json.dumps(compatibility, sort_keys=True).encode()
    return {"hostname": platform.node(), "python": sys.executable,
            "python_version": platform.python_version(), "devices": devices,
            "cuda_available": torch.cuda.is_available(),
            "spec_sha256": hashlib.sha256(SPEC.read_bytes()).hexdigest(),
            "compatibility": compatibility, "environment_sha256": hashlib.sha256(encoded).hexdigest(),
            "runtime_pass": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", help="Create an isolated system-site-packages venv at this path")
    parser.add_argument("--output")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--environment-archive")
    parser.add_argument("--archive-sha256")
    args = parser.parse_args()
    if args.install:
        target = Path(args.install).resolve()
        if not (target / "bin/python").exists():
            subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(target)], check=True)
        if args.environment_archive:
            import tarfile

            checksum = hashlib.sha256()
            with Path(args.environment_archive).open("rb") as stream:
                for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                    checksum.update(block)
            if not args.archive_sha256 or checksum.hexdigest() != args.archive_sha256:
                raise RuntimeError("Offline environment archive checksum mismatch")
            site = subprocess.check_output([str(target / "bin/python"), "-c",
                                           "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip()
            with tarfile.open(args.environment_archive) as archive:
                archive.extractall(site, filter="data")
        else:
            subprocess.run([str(target / "bin/python"), "-m", "pip", "install", "-r", str(SPEC)], check=True)
        command = [str(target / "bin/python"), str(Path(__file__).resolve())]
        if args.output:
            command.extend(["--output", args.output])
        if args.require_cuda:
            command.append("--require-cuda")
        subprocess.run(command, check=True, env={**os.environ, **THREAD_ENV})
        return
    result = fingerprint(args.require_cuda)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
