#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from benchmark_eval.encoders import MODEL_REGISTRY, MODEL_ROOT


DINOV3_ROOT = Path(os.environ.get("DINOV3_CODE_ROOT", "/mnt/huawei_deepcad/dinov3"))
BENCH_SEG_ROOT = Path("/mnt/huawei_deepcad/benchmark/segmentation")
OUT_ROOT = Path("benchmark_runs/dense_probe")

DATASET_ROOTS = {
    "bbbc038": BENCH_SEG_ROOT / "bbbc038/extracted",
    "conic": BENCH_SEG_ROOT / "conic/extracted",
    "livecell": BENCH_SEG_ROOT / "LIVECell/LIVECell_dataset_2021",
    "monuseg": BENCH_SEG_ROOT / "monuseg/extracted",
    "pannuke": BENCH_SEG_ROOT / "pannuke/extracted",
    "tissuenet": BENCH_SEG_ROOT / "tissuenet/extracted",
}

DATASET_IMG_SIZE = {
    "bbbc038": 512,
    "conic": 256,
    "livecell": 512,
    "monuseg": 512,
    "pannuke": 256,
    "tissuenet": 256,
}

DATASET_CONFIGS = {
    "bbbc038": {"num_classes": 2, "class_names": ["background", "cell"]},
    "conic": {
        "num_classes": 7,
        "class_names": [
            "background",
            "neutrophil",
            "epithelial",
            "lymphocyte",
            "plasma_cell",
            "eosinophil",
            "connective",
        ],
    },
    "livecell": {"num_classes": 2, "class_names": ["background", "cell"]},
    "monuseg": {"num_classes": 2, "class_names": ["background", "nucleus"]},
    "pannuke": {
        "num_classes": 6,
        "class_names": ["background", "neoplastic", "inflammatory", "connective", "dead", "epithelial"],
    },
    "tissuenet": {"num_classes": 2, "class_names": ["background", "cell"]},
}


def configure_dataset_roots(benchmark_root: str | Path) -> None:
    """Point dense evaluators at a benchmark tree other than the host default."""
    segmentation_root = Path(benchmark_root) / "segmentation"
    DATASET_ROOTS.update(
        {
            "bbbc038": segmentation_root / "bbbc038/extracted",
            "conic": segmentation_root / "conic/extracted",
            "livecell": segmentation_root / "LIVECell/LIVECell_dataset_2021",
            "monuseg": segmentation_root / "monuseg/extracted",
            "pannuke": segmentation_root / "pannuke/extracted",
            "tissuenet": segmentation_root / "tissuenet/extracted",
        }
    )


def safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def add_dinov3_path() -> None:
    p = str(DINOV3_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)


def build_bioseg_dataset(dataset_name: str, split: str, img_size: int):
    add_dinov3_path()
    from dinov3.eval.bio_segmentation.datasets import DATASET_REGISTRY

    DatasetClass, get_paths_fn, loader_type = DATASET_REGISTRY[dataset_name]
    root = str(DATASET_ROOTS[dataset_name])
    size = (img_size, img_size)
    common = dict(size=size, augment=False, do_normalize=False)
    if loader_type == "file":
        img_paths, mask_paths = get_paths_fn(root, split=split)
        return DatasetClass(img_paths, mask_paths, **common)
    if loader_type == "coco":
        coco_json, image_root = get_paths_fn(root, split=split)
        return DatasetClass(coco_json, image_root, **common)
    if loader_type == "array":
        if dataset_name == "conic":
            images_npy, labels_npy, indices = get_paths_fn(root, split=split)
            return DatasetClass(images_npy, labels_npy, indices=indices, **common)
        if dataset_name == "pannuke":
            fold_dirs = get_paths_fn(root)
            split_map = {"train": [1, 2], "val": [3], "test": [3]}
            return DatasetClass(fold_dirs, split_folds=split_map[split], **common)
        if dataset_name == "tissuenet":
            npz_path = get_paths_fn(root, split=split)
            return DatasetClass(npz_path, **common)
    raise ValueError(f"Unsupported dataset {dataset_name} ({loader_type})")


def collate_bioseg(batch):
    imgs, sems, insts = [], [], []
    for item in batch:
        imgs.append(item[0])
        sems.append(item[1])
        insts.append(item[2] if len(item) > 2 else (item[1] > 0).long())
    return torch.stack(imgs), torch.stack(sems), torch.stack(insts)


class DenseFeatureExtractor:
    input_size = 224
    patch_size = 16

    def __init__(self, model_name: str, device: str, canonical: bool = False):
        self.model_name = model_name
        self.device = torch.device(device)
        self.canonical = canonical  # if True: feed dataset-canonical res (no downsize to model-native)
        self.spec = MODEL_REGISTRY[model_name]
        if self.spec.kind == "transformers":
            self._init_transformers()
        elif self.spec.kind == "timm":
            self._init_timm()
        elif self.spec.kind == "pe_openphenom":
            self._init_pe()
        elif self.spec.kind == "open_clip":
            self._init_bioclip()
        elif self.spec.kind == "conch":
            self._init_conch()
        elif self.spec.kind == "channelvit":
            self._init_jump_cp()
        elif self.spec.kind == "cytoself":
            self._init_cytoself()
        elif self.spec.kind == "keras_torch":
            self._init_cytoimagenet()
        else:
            raise NotImplementedError(f"Unsupported model kind: {self.spec.kind}")

    def _init_transformers(self):
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(
            self.spec.path, local_files_only=True, trust_remote_code=True
        )
        self.model, loading = AutoModel.from_pretrained(
            self.spec.path, local_files_only=True, trust_remote_code=True, output_loading_info=True
        )
        missing = set(loading.get('missing_keys',())) & set(dict(self.model.named_parameters()))
        if missing:raise RuntimeError(f'Published model parameters missing: {sorted(missing)}')
        self.model.to(self.device).eval()
        size = getattr(self.processor, "size", None) or {}
        self.input_size = int(size.get("height") or size.get("shortest_edge") or 224)
        self.patch_size = int(getattr(getattr(self.model, "config", object()), "patch_size", 16))
        mean = getattr(self.processor, "image_mean", [0.485, 0.456, 0.406])
        std = getattr(self.processor, "image_std", [0.229, 0.224, 0.225])
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def _init_timm(self):
        import json
        import timm
        from safetensors.torch import load_file

        cfg = json.loads((self.spec.path / "config.json").read_text())
        arch = cfg["architecture"]
        model_args = dict(cfg.get("model_args", {}))
        model_args.setdefault("num_classes", cfg.get("num_classes", 0))
        model_args.setdefault("global_pool", cfg.get("global_pool", "token"))
        if self.canonical:
            model_args.setdefault("dynamic_img_size", True)  # let timm ViT interpolate pos-embed at non-native res
        if self.spec.path.name.lower() == "virchow2":
            from timm.layers import SwiGLUPacked

            model_args.setdefault("mlp_layer", SwiGLUPacked)
            model_args.setdefault("act_layer", torch.nn.SiLU)
        if self.spec.path.name.lower() == "h-optimus-0":
            model_args.setdefault("img_size", 224)
            model_args.setdefault("init_values", 1e-5)
            model_args.setdefault("dynamic_img_size", False)
        self.model = timm.create_model(arch, pretrained=False, **model_args)
        weight_path = self.spec.path / "model.safetensors"
        state = load_file(str(weight_path)) if weight_path.exists() else torch.load(self.spec.path / "pytorch_model.bin", map_location="cpu")
        self._load_published(state)
        self.model.to(self.device).eval()
        pcfg = cfg.get("pretrained_cfg", {})
        self.input_size = int(pcfg.get("input_size", [3, 224, 224])[-1])
        ps = getattr(getattr(self.model, "patch_embed", None), "patch_size", 16)
        self.patch_size = int(ps[0] if isinstance(ps, tuple) else ps)
        self.mean = torch.tensor(pcfg.get("mean", [0.485, 0.456, 0.406])).view(1, 3, 1, 1)
        self.std = torch.tensor(pcfg.get("std", [0.229, 0.224, 0.225])).view(1, 3, 1, 1)

    def _init_pe(self):
        sys.path.insert(0, str(self.spec.path.parent))
        from PE_OpenPhenom.huggingface_mae import MAEConfig, MAEModel

        self.model = MAEModel(MAEConfig(mask_ratio=0.0))
        ckpt = torch.load(self.spec.path / "model.safetensors", map_location="cpu")
        self._load_published(ckpt.get("state_dict", ckpt))
        self.model.to(self.device).eval()
        self.input_size = 256
        self.patch_size = 16

    def _init_bioclip(self):
        import json
        import open_clip

        self.model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained=None)
        state = torch.load(self.spec.path / "open_clip_pytorch_model.bin", map_location="cpu")
        self._load_published(state.get("state_dict", state))
        self.model.to(self.device).eval()
        cfg = json.loads((self.spec.path / "open_clip_config.json").read_text())["preprocess_cfg"]
        self.input_size = int(cfg.get("size", 224))
        ps = getattr(self.model.visual, "patch_size", 16) or 16
        self.patch_size = int(ps[0] if isinstance(ps, tuple) else ps)
        self.mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)

    def _init_conch(self):
        sys.path.insert(0, str(MODEL_ROOT / "_vendor"))
        from conch.open_clip_custom import create_model_from_pretrained

        self.model, self.transform = create_model_from_pretrained(
            "conch_ViT-B-16", str(self.spec.path / "pytorch_model.bin")
        )
        self.model.to(self.device).eval()
        self.input_size = 224
        trunk = getattr(self.model.visual, "trunk", None)
        ps = getattr(getattr(trunk, "patch_embed", None), "patch_size", 16) if trunk is not None else 16
        self.patch_size = int(ps[0] if isinstance(ps, tuple) else ps)
        mean = getattr(self.model.visual, "image_mean", None) or [0.48145466, 0.4578275, 0.40821073]
        std = getattr(self.model.visual, "image_std", None) or [0.26862954, 0.26130258, 0.27577711]
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def _init_jump_cp(self):
        sys.path.insert(0, str(MODEL_ROOT / "_vendor"))
        from channelvit.backbone.hcs_channel_vit import hcs_channelvit_small

        self.in_chans = 5
        self.model = hcs_channelvit_small(patch_size=8, in_chans=self.in_chans, enable_sample=False)
        state = torch.load(self.spec.path / "cpjump_cellpaint_channelvit_small_p8_with_hcs_supervised.pth", map_location="cpu")
        self._load_published(state)
        self.model.to(self.device).eval()
        self.input_size = 224
        self.patch_size = 8
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)

    def _init_cytoself(self):
        from benchmark_eval.encoders import CytoselfEncoder

        enc = CytoselfEncoder(self.spec.path, str(self.device), 1)
        self.model = enc.model.to(self.device).eval()
        self.in_channels = enc.in_channels
        self.input_size = 100
        self.patch_size = 4

    def _load_published(self,state):
        result=self.model.load_state_dict(state,strict=False)
        missing=set(result.missing_keys)&set(dict(self.model.named_parameters()))
        if missing:raise RuntimeError(f'Published model parameters missing: {sorted(missing)}')

    def _init_cytoimagenet(self):
        os.environ.setdefault("KERAS_BACKEND", "torch")
        from keras.applications.efficientnet import EfficientNetB0

        self.model = EfficientNetB0(
            include_top=False,
            weights=None,
            pooling=None,
            input_shape=(224, 224, 3),
        )
        self.model.load_weights(self.spec.path / "efficientnetb0_weights-notop.h5")
        self.model.to(self.device).eval()
        self.input_size = 224
        self.patch_size = 32

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        if self.canonical:
            # Keep dataset-canonical resolution, but snap H,W to a multiple of patch_size so
            # patch14 models (virchow2, dinov2-L/14) accept it (e.g. 512 -> 518). No-op for patch16.
            ps = max(int(getattr(self, "patch_size", 16)), 1)
            h, w = int(x.shape[-2]), int(x.shape[-1])
            nh = max(ps, round(h / ps) * ps)
            nw = max(ps, round(w / ps) * ps)
            if nh != h or nw != w:
                x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
            return x
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        return x

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    @staticmethod
    def _tokens_to_map(tokens: torch.Tensor) -> torch.Tensor:
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[0]
        if isinstance(tokens, dict):
            tokens = tokens.get("x_norm_patchtokens") or tokens.get("last_hidden_state") or next(iter(tokens.values()))
        if tokens.ndim == 4:
            return tokens
        n = tokens.shape[1]
        s = int(math.sqrt(n))
        if s * s != n:
            # Drop prefix/register tokens by keeping the largest square suffix.
            s = int(math.sqrt(n - 1))
            if s * s == n - 1:
                tokens = tokens[:, 1:]
            else:
                s = int(math.sqrt(n))
                keep = s * s
                tokens = tokens[:, -keep:]
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], s, s).contiguous()

    @torch.inference_mode()
    def __call__(self, imgs_01: torch.Tensor) -> torch.Tensor:
        x = self._resize(imgs_01).to(self.device)
        if self.spec.kind in {"transformers", "timm", "open_clip", "conch", "channelvit"}:
            x = self._norm(x)
        if self.spec.kind == "transformers":
            kwargs = {"pixel_values": x}
            model = self.model.vision_model if hasattr(self.model, "vision_model") else self.model
            try:
                out = model(**kwargs, interpolate_pos_encoding=True)
            except TypeError:
                out = model(**kwargs)
            feat = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            return self._tokens_to_map(feat).float()
        if self.spec.kind == "timm":
            out = self.model.forward_features(x)
            return self._tokens_to_map(out).float()
        if self.spec.kind == "pe_openphenom":
            x = self.model.input_norm(x)
            latent, _mask, _ind_restore = self.model.encoder.forward_masked(x, 0.0)
            if latent.ndim == 3 and latent.shape[1] > 1:
                latent = latent[:, 1:]
            return self._tokens_to_map(latent).float()
        if self.spec.kind == "open_clip":
            v = self.model.visual
            y = v.conv1(x)
            y = y.reshape(y.shape[0], y.shape[1], -1).permute(0, 2, 1)
            cls = v.class_embedding.to(y.dtype) + torch.zeros(y.shape[0], 1, y.shape[-1], dtype=y.dtype, device=y.device)
            y = torch.cat([cls, y], dim=1)
            y = y + v.positional_embedding.to(y.dtype)
            y = v.ln_pre(y)
            y = y.permute(1, 0, 2)
            y = v.transformer(y)
            y = y.permute(1, 0, 2)
            y = v.ln_post(y)
            return self._tokens_to_map(y[:, 1:]).float()
        if self.spec.kind == "conch":
            v = self.model.visual
            if hasattr(v, "trunk"):
                return self._tokens_to_map(v.trunk.forward_features(x)).float()
            # Fallback for OpenCLIP-like visual towers.
            y = v.conv1(x)
            y = y.reshape(y.shape[0], y.shape[1], -1).permute(0, 2, 1)
            cls = v.class_embedding.to(y.dtype) + torch.zeros(y.shape[0], 1, y.shape[-1], dtype=y.dtype, device=y.device)
            y = torch.cat([cls, y], dim=1)
            y = y + v.positional_embedding.to(y.dtype)
            y = v.ln_pre(y)
            y = y.permute(1, 0, 2)
            y = v.transformer(y)
            y = y.permute(1, 0, 2)
            y = v.ln_post(y)
            return self._tokens_to_map(y[:, 1:]).float()
        if self.spec.kind == "channelvit":
            rgb = x
            xx = torch.zeros((rgb.shape[0], self.in_chans, rgb.shape[2], rgb.shape[3]), dtype=rgb.dtype, device=rgb.device)
            xx[:, :3] = rgb
            channels = torch.arange(self.in_chans, device=rgb.device).repeat(rgb.shape[0], 1)
            outs = self.model.get_intermediate_layers(xx, extra_tokens={"channels": channels}, n=1)
            tokens = outs[0] if isinstance(outs, (tuple, list)) else outs
            spatial = (self.input_size // self.patch_size) ** 2
            if tokens.shape[1] % spatial == 0:
                c = tokens.shape[1] // spatial
                tokens = tokens.reshape(tokens.shape[0], c, spatial, tokens.shape[2]).mean(1)
            return self._tokens_to_map(tokens).float()
        if self.spec.kind == "cytoself":
            gray = x.mean(dim=1, keepdim=True)
            xx = gray.repeat(1, self.in_channels, 1, 1)
            return self.model(xx).float()
        if self.spec.kind == "keras_torch":
            # Keras EfficientNet includes its own 1/255 rescaling and expects
            # NHWC pixels in [0, 255]. Its final convolution is a 7x7 dense map.
            xx = (x * 255.0).permute(0, 2, 3, 1).contiguous()
            out = self.model(xx, training=False)
            return out.permute(0, 3, 1, 2).contiguous().float()
        raise AssertionError(self.spec.kind)


@torch.inference_mode()
def extract_cache(args, model_name: str, dataset_name: str, split: str) -> Path:
    img_size = args.img_size or DATASET_IMG_SIZE[dataset_name]
    cache_path = OUT_ROOT / "cache" / dataset_name / model_name / f"{split}.npz"
    if cache_path.exists() and not args.overwrite_cache:
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Initialize CUDA before importing the OpenCV-backed segmentation datasets.
    # With this host's OpenCV/PyTorch builds, importing cv2 first can segfault
    # when a Transformers/timm model is subsequently moved to CUDA.
    extractor = DenseFeatureExtractor(model_name, args.device, canonical=getattr(args, "feature_canonical", False))
    ds = build_bioseg_dataset(dataset_name, split, img_size)
    loader = DataLoader(ds, batch_size=args.extract_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_bioseg)
    tmp = str(cache_path) + f".tmp.{os.getpid()}"
    import zipfile
    from numpy.lib import format as npy_format

    def write_array(zf, key, arr):
        with zf.open(f"{key}.npy", "w", force_zip64=True) as h:
            npy_format.write_array(h, np.asarray(arr), allow_pickle=False)

    n_chunks = 0
    n_samples = 0
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for b, (imgs, sem, inst) in enumerate(loader, 1):
            feats = extractor(imgs)
            if args.max_feature_side and max(feats.shape[-2:]) > args.max_feature_side:
                feats = F.interpolate(
                    feats.float(),
                    size=(args.max_feature_side, args.max_feature_side),
                    mode="bilinear",
                    align_corners=False,
                )
            write_array(zf, f"features_{n_chunks:04d}", feats.half().cpu().numpy())
            write_array(zf, f"sem_masks_{n_chunks:04d}", sem.numpy().astype(np.int16))
            write_array(zf, f"inst_maps_{n_chunks:04d}", inst.numpy().astype(np.int32))
            n_samples += int(imgs.shape[0])
            n_chunks += 1
            if b == 1 or b % 20 == 0:
                print(f"[extract] {model_name} {dataset_name}/{split}: {n_samples}/{len(ds)} features={tuple(feats.shape)}", flush=True)
        meta = {
            "chunked": np.int8(1),
            "num_chunks": np.int32(n_chunks),
            "num_samples": np.int32(n_samples),
            "orig_H": np.int32(img_size),
            "orig_W": np.int32(img_size),
            "patch_size": np.int32(extractor.patch_size),
            "embed_dim": np.int32(feats.shape[1] if n_chunks else 0),
            "n_layers": np.int32(getattr(extractor, "n_layers", 1)),
        }
        for k, v in meta.items():
            write_array(zf, k, v)
    os.replace(tmp, cache_path)
    return cache_path


def run_linear_probe(args, model_name: str, dataset_name: str, caches: dict[str, Path]) -> Path:
    out_dir = OUT_ROOT / "linear_probe" / dataset_name / model_name
    result_path = out_dir / "results.json"
    if result_path.exists() and not args.overwrite_probe:
        return result_path
    cfg = DATASET_CONFIGS[dataset_name]
    # Use base python for skimage/scipy metric dependencies; cached features are model-agnostic.
    script = f"""
import sys
sys.path.insert(0, '{DINOV3_ROOT}')
from dinov3.eval.bio_segmentation.linear_probe import run_cached_linear_probe
run_cached_linear_probe(
    train_cache='{caches['train']}',
    val_cache='{caches['val']}',
    test_cache='{caches['test']}',
    output_dir='{out_dir}',
    num_classes={cfg['num_classes']},
    class_names={cfg['class_names']!r},
    epochs={args.epochs},
    lr={args.lr},
    batch_size={args.probe_batch_size},
    weight_decay={args.weight_decay},
    dropout={args.dropout},
    num_workers={args.probe_num_workers},
    eval_every={args.eval_every},
    train_samples={args.train_samples if args.train_samples is not None else 'None'},
    train_fraction={args.train_fraction if args.train_fraction is not None else 'None'},
    seed={args.seed},
)
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(DINOV3_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "")
    log_path = out_dir / "linear_probe.log"
    metric_python = os.environ.get("DENSE_PROBE_METRIC_PYTHON", "/home/deepcad/anaconda3/bin/python")
    with log_path.open("w") as log:
        proc = subprocess.run([metric_python, "-c", script], cwd=Path.cwd(), env=env, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"linear probe failed for {model_name}/{dataset_name}; see {log_path}")
    return result_path


def append_summary(model_name: str, dataset_name: str, result_path: Path) -> None:
    summary = OUT_ROOT / "summary.csv"
    fields = [
        "model",
        "dataset",
        "split",
        "mIoU",
        "mDice",
        "mPrecision",
        "mRecall",
        "AJI",
        "AP",
        "AP50",
        "AP75",
        "bPQ",
        "results_json",
    ]
    exists = summary.exists()
    data = json.loads(result_path.read_text())
    with summary.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        for split in ("val", "test"):
            if split not in data:
                continue
            row = {"model": model_name, "dataset": dataset_name, "split": split, "results_json": str(result_path)}
            row.update({k: data[split].get(k, "") for k in fields})
            w.writerow({k: row.get(k, "") for k in fields})


def main() -> int:
    add_dinov3_path()
    from run_fm_dense_rules import main as rules_main
    return rules_main()


def legacy_main() -> int:
    raise RuntimeError('Legacy dense protocol is disabled: use the unified FM14 Rules entrypoint')
    parser = argparse.ArgumentParser(description="Frozen FM dense linear-probe benchmark compatible with DINOv3 bio_segmentation outputs.")
    parser.add_argument("--models", nargs="+", default=["dinov2", "mae", "siglip2", "bioclip"])
    parser.add_argument("--datasets", nargs="+", default=["bbbc038", "conic", "monuseg", "pannuke", "tissuenet"])
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--img-size", type=int, default=0, help="0 uses DINOv3 canonical per-dataset size.")
    parser.add_argument("--feature-canonical", action="store_true", help="Variant B: extract at dataset-canonical res (no downsize to model-native); transformers/timm only.")
    parser.add_argument("--extract-batch-size", type=int, default=8)
    parser.add_argument("--max-feature-side", type=int, default=32, help="Downsample very dense feature maps before caching/probing; 0 disables.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--probe-batch-size", type=int, default=64)
    parser.add_argument("--probe-num-workers", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--overwrite-probe", action="store_true")
    parser.add_argument("--out-root", default=None, help="Override output/cache root (default benchmark_runs/dense_probe).")
    args = parser.parse_args()

    global OUT_ROOT
    if args.out_root:
        OUT_ROOT = Path(args.out_root)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for dataset_name in args.datasets:
        if dataset_name not in DATASET_CONFIGS:
            raise KeyError(f"Unknown dataset {dataset_name}")
        for model_name in args.models:
            if model_name not in MODEL_REGISTRY:
                raise KeyError(f"Unknown model {model_name}")
            print(f"[run] dense probe model={model_name} dataset={dataset_name}", flush=True)
            caches = {split: extract_cache(args, model_name, dataset_name, split) for split in args.splits}
            result_path = run_linear_probe(args, model_name, dataset_name, caches)
            append_summary(model_name, dataset_name, result_path)
            print(f"[done] {model_name} {dataset_name}: {result_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
