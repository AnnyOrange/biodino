from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms


MODEL_ROOT = Path("/mnt/huawei_deepcad/benchmark_model")


@dataclass
class ModelSpec:
    name: str
    kind: str
    path: Path
    note: str = ""


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "dinov2": ModelSpec("dinov2", "transformers", MODEL_ROOT / "General_baseline/DINOv2"),
    "mae": ModelSpec("mae", "transformers", MODEL_ROOT / "General_baseline/MAE"),
    "siglip2": ModelSpec("siglip2", "transformers", MODEL_ROOT / "General_baseline/SigLIP2"),
    "uni": ModelSpec("uni", "timm", MODEL_ROOT / "Pathology_FM/UNI"),
    "conch": ModelSpec("conch", "conch", MODEL_ROOT / "Pathology_FM/CONCH"),
    "virchow2": ModelSpec("virchow2", "timm", MODEL_ROOT / "Pathology_FM/Virchow2"),
    "gigapath": ModelSpec("gigapath", "timm", MODEL_ROOT / "Pathology_FM/GigaPath"),
    "hoptimus0": ModelSpec("hoptimus0", "timm", MODEL_ROOT / "Pathology_FM/H-optimus-0"),
    "phikon2": ModelSpec("phikon2", "transformers", MODEL_ROOT / "Pathology_FM/Phikon-v2"),
    "pe": ModelSpec("pe", "pe_openphenom", MODEL_ROOT / "General_baseline/PE_OpenPhenom"),
    "bioclip": ModelSpec("bioclip", "open_clip", MODEL_ROOT / "Cell_Bioimage_FM/BioCLIP", "requires open_clip_torch"),
    "cytoself": ModelSpec("cytoself", "cytoself", MODEL_ROOT / "Cell_Bioimage_FM/Cytoself", "uses the downloaded Cytoself encoder-1 weights"),
    "jump_cp": ModelSpec("jump_cp", "channelvit", MODEL_ROOT / "Cell_Bioimage_FM/JUMP_CP_encoder", "ChannelViT JUMP-CP cell-painting checkpoint"),
    "cytoimagenet": ModelSpec("cytoimagenet", "keras_torch", MODEL_ROOT / "Cell_Bioimage_FM/CytoImageNet"),
}


def _pil_collate(batch):
    imgs, labels, paths = zip(*batch)
    return list(imgs), np.asarray(labels), list(paths)


class BaseEncoder:
    feature_dim: int | None = None

    def encode_pil(self, images: list[Image.Image]) -> np.ndarray:
        raise NotImplementedError


class TransformersEncoder(BaseEncoder):
    def __init__(self, path: Path, device: str, batch_size: int):
        from transformers import AutoImageProcessor, AutoModel

        self.path = path
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.processor = AutoImageProcessor.from_pretrained(
            path, local_files_only=True, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            path, local_files_only=True, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode_pil(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if hasattr(self.model, "get_image_features"):
            feat = self.model.get_image_features(**inputs)
        else:
            out = self.model(**inputs)
            feat = None
        if feat is not None:
            pass
        elif hasattr(out, "image_embeds") and out.image_embeds is not None:
            feat = out.image_embeds
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            feat = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            feat = out.last_hidden_state[:, 0]
        else:
            raise RuntimeError(f"Cannot infer feature tensor from output type {type(out)}")
        feat = torch.nn.functional.normalize(feat.float(), dim=1)
        return feat.cpu().numpy()


class TimmEncoder(BaseEncoder):
    def __init__(self, path: Path, device: str, batch_size: int):
        import timm
        from safetensors.torch import load_file

        self.path = path
        self.device = torch.device(device)
        self.batch_size = batch_size
        cfg = json.loads((path / "config.json").read_text())
        arch = cfg["architecture"]
        model_args = dict(cfg.get("model_args", {}))
        model_args.setdefault("num_classes", cfg.get("num_classes", 0))
        model_args.setdefault("global_pool", cfg.get("global_pool", "token"))
        if path.name.lower() == "virchow2":
            # Virchow2 uses the SwiGLU MLP variant; without this timm builds a
            # same-name ViT-H whose MLP tensor shapes do not match the checkpoint.
            from timm.layers import SwiGLUPacked

            model_args.setdefault("mlp_layer", SwiGLUPacked)
            model_args.setdefault("act_layer", torch.nn.SiLU)
        if path.name.lower() == "h-optimus-0":
            # The timm architecture defaults to 518px, while the released
            # checkpoint/model card use a fixed 224px (16x16 patch) grid.
            model_args.setdefault("img_size", 224)
            model_args.setdefault("init_values", 1e-5)
            model_args.setdefault("dynamic_img_size", False)
        self.model = timm.create_model(arch, pretrained=False, **model_args)
        weight_path = path / "model.safetensors"
        if weight_path.exists():
            state = load_file(str(weight_path))
        else:
            state = torch.load(path / "pytorch_model.bin", map_location="cpu")
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if len(unexpected) > 20:
            unexpected = unexpected[:20]
        print(f"[timm] loaded {path.name}: missing={len(missing)} unexpected={len(unexpected)}")
        self.model.to(self.device).eval()
        pcfg = cfg.get("pretrained_cfg", {})
        mean = pcfg.get("mean", [0.485, 0.456, 0.406])
        std = pcfg.get("std", [0.229, 0.224, 0.225])
        size = pcfg.get("input_size", [3, 224, 224])[-1]
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    @torch.inference_mode()
    def encode_pil(self, images: list[Image.Image]) -> np.ndarray:
        x = torch.stack([self.transform(img) for img in images]).to(self.device)
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim == 3:
            out = out[:, 0]
        feat = torch.nn.functional.normalize(out.float(), dim=1)
        return feat.cpu().numpy()


class PEOpenPhenomEncoder(BaseEncoder):
    def __init__(self, path: Path, device: str, batch_size: int):
        import sys

        sys.path.insert(0, str(path.parent))
        from PE_OpenPhenom.huggingface_mae import MAEConfig, MAEModel

        self.path = path
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model = MAEModel(MAEConfig(mask_ratio=0.0))
        ckpt = torch.load(path / "model.safetensors", map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        print(f"[pe] loaded: missing={len(missing)} unexpected={len(unexpected)}")
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])

    @torch.inference_mode()
    def encode_pil(self, images: list[Image.Image]) -> np.ndarray:
        x = torch.stack([self.transform(img) for img in images]).to(self.device)
        x = self.model.input_norm(x)
        latent, _mask, _ind_restore = self.model.encoder.forward_masked(x, 0.0)
        if latent.ndim == 3:
            if latent.shape[1] > 1:
                feat = latent[:, 1:].mean(dim=1)
            else:
                feat = latent[:, 0]
        else:
            feat = latent
        feat = torch.nn.functional.normalize(feat.float(), dim=1)
        return feat.cpu().numpy()


class BioCLIPEncoder(BaseEncoder):
    def __init__(self, path: Path, device: str, batch_size: int):
        import json
        import open_clip

        self.path = path
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained=None)
        state = torch.load(path / "open_clip_pytorch_model.bin", map_location="cpu")
        missing, unexpected = self.model.load_state_dict(state.get("state_dict", state), strict=False)
        print(f"[bioclip] loaded: missing={len(missing)} unexpected={len(unexpected)}")
        self.model.to(self.device).eval()
        cfg = json.loads((path / "open_clip_config.json").read_text())["preprocess_cfg"]
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg["mean"], std=cfg["std"]),
        ])

    @torch.inference_mode()
    def encode_pil(self, images: list[Image.Image]) -> np.ndarray:
        x = torch.stack([self.transform(img) for img in images]).to(self.device)
        feat = self.model.encode_image(x)
        feat = torch.nn.functional.normalize(feat.float(), dim=1)
        return feat.cpu().numpy()


class CONCHEncoder(BaseEncoder):
    def __init__(self, path: Path, device: str, batch_size: int):
        import sys

        sys.path.insert(0, str(MODEL_ROOT / "_vendor"))
        from conch.open_clip_custom import create_model_from_pretrained

        self.path = path
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model, self.transform = create_model_from_pretrained(
            "conch_ViT-B-16", str(path / "pytorch_model.bin")
        )
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def encode_pil(self, images: list[Image.Image]) -> np.ndarray:
        x = torch.stack([self.transform(img) for img in images]).to(self.device)
        feat = self.model.encode_image(x, proj_contrast=False, normalize=False)
        feat = torch.nn.functional.normalize(feat.float(), dim=1)
        return feat.cpu().numpy()


class ChannelViTJumpCPEncoder(BaseEncoder):
    def __init__(self, path: Path, device: str, batch_size: int):
        import sys

        sys.path.insert(0, str(MODEL_ROOT / "_vendor"))
        from channelvit.backbone.hcs_channel_vit import hcs_channelvit_small

        self.path = path
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.in_chans = 5
        self.model = hcs_channelvit_small(patch_size=8, in_chans=self.in_chans, enable_sample=False)
        state = torch.load(
            path / "cpjump_cellpaint_channelvit_small_p8_with_hcs_supervised.pth",
            map_location="cpu",
        )
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        print(f"[jump_cp] loaded: missing={len(missing)} unexpected={len(unexpected)}")
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ])

    @torch.inference_mode()
    def encode_pil(self, images: list[Image.Image]) -> np.ndarray:
        rgb = torch.stack([self.transform(img) for img in images])
        x = torch.zeros((rgb.shape[0], self.in_chans, rgb.shape[2], rgb.shape[3]), dtype=rgb.dtype)
        x[:, :3] = rgb
        x = x.to(self.device)
        channels = torch.arange(self.in_chans, device=self.device).repeat(x.shape[0], 1)
        feat = self.model(x, extra_tokens={"channels": channels})
        feat = torch.nn.functional.normalize(feat.float(), dim=1)
        return feat.cpu().numpy()


class CytoselfEncoder1(torch.nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        import sys

        sys.path.insert(0, str(MODEL_ROOT / "_vendor"))
        from cytoself.trainer.autoencoder.encoders.efficientenc2d import efficientenc_b0

        block_args = [
            {"expand_ratio": 1, "kernel": 3, "stride": 1, "input_channels": 32, "out_channels": 16, "num_layers": 1},
            {"expand_ratio": 6, "kernel": 3, "stride": 2, "input_channels": 16, "out_channels": 24, "num_layers": 2},
            {"expand_ratio": 6, "kernel": 5, "stride": 1, "input_channels": 24, "out_channels": 40, "num_layers": 2},
        ]
        self.encoder1 = efficientenc_b0(
            blocks_args=block_args,
            in_channels=in_channels,
            out_channels=64,
            first_layer_stride=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder1(x)


class CytoselfEncoder(BaseEncoder):
    def __init__(self, path: Path, device: str, batch_size: int):
        import h5py

        self.path = path
        self.device = torch.device(device)
        self.batch_size = batch_size
        with h5py.File(path / "model_protein.h5", "r") as f:
            self.in_channels = int(f["encoder1"]["stem_conv"]["kernel:0"].shape[2])
            self.model = CytoselfEncoder1(self.in_channels)
            self._load_encoder1(f["encoder1"])
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((100, 100), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])

    @staticmethod
    def _tf_conv_to_torch(arr: np.ndarray) -> torch.Tensor:
        if arr.ndim == 4:
            return torch.from_numpy(np.asarray(arr).transpose(3, 2, 0, 1))
        raise ValueError(f"Unexpected convolution shape: {arr.shape}")

    @staticmethod
    def _tf_depthwise_to_torch(arr: np.ndarray) -> torch.Tensor:
        if arr.ndim == 4:
            arr = np.asarray(arr)
            return torch.from_numpy(arr.transpose(2, 3, 0, 1).reshape(arr.shape[2] * arr.shape[3], 1, arr.shape[0], arr.shape[1]))
        raise ValueError(f"Unexpected depthwise shape: {arr.shape}")

    @staticmethod
    def _copy_bn(state: dict[str, torch.Tensor], prefix: str, group, layer: str):
        src = group[layer]
        state[f"{prefix}.weight"] = torch.from_numpy(np.asarray(src["gamma:0"]))
        state[f"{prefix}.bias"] = torch.from_numpy(np.asarray(src["beta:0"]))
        state[f"{prefix}.running_mean"] = torch.from_numpy(np.asarray(src["moving_mean:0"]))
        state[f"{prefix}.running_var"] = torch.from_numpy(np.asarray(src["moving_variance:0"]))

    def _load_conv(self, state: dict[str, torch.Tensor], key: str, group, layer: str):
        state[key] = self._tf_conv_to_torch(np.asarray(group[layer]["kernel:0"]))

    def _load_dw(self, state: dict[str, torch.Tensor], key: str, group, layer: str):
        state[key] = self._tf_depthwise_to_torch(np.asarray(group[layer]["depthwise_kernel:0"]))

    def _load_encoder1(self, group):
        state = self.model.encoder1.state_dict()
        self._load_conv(state, "features.0.0.weight", group, "stem_conv")
        self._copy_bn(state, "features.0.1", group, "stem_bn")

        mapping = {
            "block1a": "features.1.0",
            "block2a": "features.2.0",
            "block2b": "features.2.1",
            "block3a": "features.3.0",
            "block3b": "features.3.1",
        }
        for block, prefix in mapping.items():
            has_expand = f"{block}_expand_conv" in group
            if has_expand:
                self._load_conv(state, f"{prefix}.block.0.0.weight", group, f"{block}_expand_conv")
                self._copy_bn(state, f"{prefix}.block.0.1", group, f"{block}_expand_bn")
                offset = 1
            else:
                offset = 0
            self._load_dw(state, f"{prefix}.block.{offset}.0.weight", group, f"{block}_dwconv")
            self._copy_bn(state, f"{prefix}.block.{offset}.1", group, f"{block}_bn")
            self._load_conv(state, f"{prefix}.block.{offset + 1}.fc1.weight", group, f"{block}_se_reduce")
            state[f"{prefix}.block.{offset + 1}.fc1.bias"] = torch.from_numpy(np.asarray(group[f"{block}_se_reduce"]["bias:0"]))
            self._load_conv(state, f"{prefix}.block.{offset + 1}.fc2.weight", group, f"{block}_se_expand")
            state[f"{prefix}.block.{offset + 1}.fc2.bias"] = torch.from_numpy(np.asarray(group[f"{block}_se_expand"]["bias:0"]))
            self._load_conv(state, f"{prefix}.block.{offset + 2}.0.weight", group, f"{block}_project_conv")
            self._copy_bn(state, f"{prefix}.block.{offset + 2}.1", group, f"{block}_project_bn")

        self._load_conv(state, "features.4.0.weight", group, "top_conv")
        self._copy_bn(state, "features.4.1", group, "top_bn")
        self.model.encoder1.load_state_dict(state, strict=True)
        print("[cytoself] loaded model_protein encoder1 weights")

    @torch.inference_mode()
    def encode_pil(self, images: list[Image.Image]) -> np.ndarray:
        rgb = torch.stack([self.transform(img) for img in images])
        gray = rgb.mean(dim=1, keepdim=True)
        x = gray.repeat(1, self.in_channels, 1, 1).to(self.device)
        out = self.model(x)
        feat = torch.nn.functional.adaptive_avg_pool2d(out.float(), 1).flatten(1)
        feat = torch.nn.functional.normalize(feat, dim=1)
        return feat.cpu().numpy()


class CytoImageNetEncoder(BaseEncoder):
    def __init__(self, path: Path, device: str, batch_size: int):
        import os

        # Keras 3 can execute the original Keras EfficientNet checkpoint on the
        # already-installed PyTorch backend, avoiding a second CUDA runtime.
        os.environ.setdefault("KERAS_BACKEND", "torch")
        from keras.applications.efficientnet import EfficientNetB0, preprocess_input

        self.path = path
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.preprocess_input = preprocess_input
        self.model = EfficientNetB0(
            include_top=False,
            weights=None,
            pooling="avg",
            input_shape=(224, 224, 3),
        )
        self.model.load_weights(path / "efficientnetb0_weights-notop.h5")
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def encode_pil(self, images: list[Image.Image]) -> np.ndarray:
        arrays = [
            np.asarray(image.resize((224, 224), Image.Resampling.BICUBIC).convert("RGB"))
            for image in images
        ]
        x = self.preprocess_input(np.asarray(arrays, dtype=np.float32))
        x = torch.from_numpy(x).to(self.device)
        output = self.model(x, training=False)
        if torch.is_tensor(output):
            output = output.detach().cpu().numpy()
        feat = np.asarray(output, dtype=np.float32)
        feat /= np.linalg.norm(feat, axis=1, keepdims=True) + 1e-12
        return feat


def build_encoder(model_name: str, device: str, batch_size: int) -> BaseEncoder:
    raise RuntimeError('Legacy native-only FM evaluation is retired; use run_fm_rules.py or run_fm_dense_rules.py with audited protocol manifests')
    hold = Path('/mnt/huawei_deepcad/benchmark_model/fair_plot_20260915/FM_ALIGNMENT_HOLD_20260917.json')
    if hold.exists():
        raise RuntimeError('FM alignment hold: legacy FM evaluation is paused; use the reviewed unified Rules entrypoint')
    key = model_name.lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {model_name}. Available: {sorted(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[key]
    if not spec.path.exists():
        raise FileNotFoundError(f"Model path does not exist: {spec.path}")
    if spec.kind == "transformers":
        return TransformersEncoder(spec.path, device, batch_size)
    if spec.kind == "timm":
        return TimmEncoder(spec.path, device, batch_size)
    if spec.kind == "pe_openphenom":
        return PEOpenPhenomEncoder(spec.path, device, batch_size)
    if spec.kind == "open_clip":
        return BioCLIPEncoder(spec.path, device, batch_size)
    if spec.kind == "conch":
        return CONCHEncoder(spec.path, device, batch_size)
    if spec.kind == "channelvit":
        return ChannelViTJumpCPEncoder(spec.path, device, batch_size)
    if spec.kind == "cytoself":
        return CytoselfEncoder(spec.path, device, batch_size)
    if spec.kind == "keras_torch":
        return CytoImageNetEncoder(spec.path, device, batch_size)
    raise NotImplementedError(f"{model_name} ({spec.kind}) is not implemented yet. {spec.note}")


def extract_features(
    dataset,
    model_name: str,
    output_path: str | Path,
    device: str = "cuda",
    batch_size: int = 32,
    num_workers: int = 4,
    overwrite: bool = False,
) -> Path:
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoder = build_encoder(model_name, device=device, batch_size=batch_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_pil_collate,
    )
    feats: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    paths: list[str] = []
    for i, (imgs, y, p) in enumerate(loader, 1):
        feats.append(encoder.encode_pil(imgs))
        labels.append(np.asarray(y))
        paths.extend(p)
        if i % 20 == 0:
            print(f"[features] {model_name}: {len(paths)} samples")
    np.savez(
        output_path,
        features=np.concatenate(feats, axis=0),
        labels=np.concatenate(labels, axis=0),
        paths=np.asarray(paths),
        model=model_name,
    )
    return output_path
