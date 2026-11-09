"""Frozen, identity-checked image-only teachers for joint feature distillation.

Inputs are student-normalized RGB crops.  A teacher returns its native final
CLS concatenated with its final spatial-token mean, without a learned decoder
or per-image normalization by default. The explicit ``readout='cls'`` path
returns only CLS and skips patch pooling. Statistics and compression belong to the caller.
No weights or Python model code are fetched from the network.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger("dinov3")

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_file(path: str | Path, expected: str, role: str) -> Path:
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{role}: not a regular file: {path}")
    if len(str(expected)) != 64 or any(c not in "0123456789abcdef" for c in str(expected)):
        raise ValueError(f"{role}: a lowercase SHA256 from the run manifest is required")
    before = path.stat()
    actual = sha256_file(path)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise RuntimeError(f"{role}: file changed during identity verification: {path}")
    if actual != expected:
        raise ValueError(f"{role}: SHA256 mismatch for {path}: {actual} != {expected}")
    return path


def _triple(value: Sequence[float], name: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{name}: expected three RGB values")
    result = tuple(float(v) for v in value)
    if not all(torch.isfinite(torch.tensor(result))):
        raise ValueError(f"{name}: non-finite value")
    if "std" in name and min(result) <= 0:
        raise ValueError(f"{name}: standard deviations must be positive")
    return result


def _torch_state(path: Path, checkpoint_branch: str = "teacher") -> Mapping[str, Tensor]:
    # mmap avoids loading a second full checkpoint into anonymous host RAM.
    raw = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(raw, Mapping):
        raise TypeError(f"Unexpected checkpoint type at {path}: {type(raw)}")
    if checkpoint_branch not in ("student", "teacher"):
        raise ValueError(f"Unknown checkpoint branch: {checkpoint_branch}")
    if checkpoint_branch == "student":
        if "model" not in raw or not isinstance(raw["model"], Mapping):
            raise ValueError(f"{path}: explicit student branch requires a full model checkpoint")
        prefix = "student.backbone."
        student = {k.removeprefix(prefix): v for k, v in raw["model"].items() if k.startswith(prefix)}
        if not student:
            raise ValueError(f"{path}: model checkpoint has no student.backbone weights")
        return student
    if "teacher" in raw:
        return {k.removeprefix("backbone."): v for k, v in raw["teacher"].items()
                if k.startswith("backbone.")}
    if "model" in raw:
        teacher = {k.removeprefix("teacher.backbone."): v for k, v in raw["model"].items()
                   if k.startswith("teacher.backbone.")}
        if teacher:
            return teacher
        raise ValueError(f"{path}: model checkpoint has no teacher.backbone weights")
    return raw


def _load_backbone_strict(model: nn.Module, state: Mapping[str, Tensor], name: str) -> None:
    if not state:
        raise ValueError(f"{name}: empty backbone state")
    missing, unexpected = model.load_state_dict(state, strict=False)
    parameters = dict(model.named_parameters())
    # DINOv3 has deterministic RoPE and attention bias-mask buffers; only these
    # may be reconstructed.  Every trainable backbone parameter must be present.
    missing_parameters = [key for key in missing if key in parameters]
    missing_other = [key for key in missing if key not in parameters
                     and "rope" not in key and not key.endswith("bias_mask")]
    if missing_parameters or missing_other or unexpected:
        raise RuntimeError(
            f"{name}: incomplete or incompatible backbone: missing parameters="
            f"{missing_parameters[:8]}, other missing={missing_other[:8]}, "
            f"unexpected={list(unexpected)[:8]}"
        )
    logger.info("[JOINT] %s strict backbone load; reconstructed buffers=%s", name, missing)


class FrozenJointTeacher(nn.Module):
    """One manifest-pinned teacher, permanently in evaluation mode.

    Required spec fields: name, kind, weights, weights_sha256, config,
    config_sha256, output_dim, and provenance.  Provenance must explicitly state
    training_supervision='image_only_self_supervised' and give source/method.
    Python files for custom models must also appear in code_sha256, keyed by
    their absolute file paths.  OpenPath additionally needs source_dir.
    """

    def __init__(
        self,
        spec: Mapping[str, Any],
        *,
        student_mean: Sequence[float] = _IMAGENET_MEAN,
        student_std: Sequence[float] = _IMAGENET_STD,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.spec = dict(spec)
        self.name = str(spec["name"])
        self.kind = str(spec["kind"])
        provenance = dict(spec.get("provenance", {}))
        if provenance.get("training_supervision") != "image_only_self_supervised":
            raise ValueError(f"{self.name}: manifest has not certified image-only self-supervision")
        if not provenance.get("source") or not provenance.get("method"):
            raise ValueError(f"{self.name}: missing source/method provenance")
        self.weights_path = _checked_file(spec["weights"], spec["weights_sha256"], self.name)
        self.config_path = _checked_file(spec["config"], spec["config_sha256"], self.name + " config")
        self.verified_code = {
            str(_checked_file(path, digest, self.name + " code")): digest
            for path, digest in dict(spec.get("code_sha256", {})).items()
        }
        self.img_size = int(spec.get("img_size", 224))
        self.chunk_size = int(spec.get("chunk_size", 2))
        if self.chunk_size < 1 or self.img_size < 1:
            raise ValueError(f"{self.name}: image and chunk sizes must be positive")
        self.model, native_mean, native_std, native_dim = self._build()
        self.output_dim = 2 * native_dim
        self.embed_dim = native_dim
        if int(spec["output_dim"]) != self.output_dim:
            raise ValueError(f"{self.name}: manifest output_dim={spec['output_dim']}, native={self.output_dim}")
        # Normalization comes from the pinned native model config.  Explicit
        # overrides can only confirm it, not silently change it.
        for key, native in (("mean", native_mean), ("std", native_std)):
            if key in spec and _triple(spec[key], key) != _triple(native, key):
                raise ValueError(f"{self.name}: {key} differs from native preprocessing")
        for key, value in (("student_mean", student_mean), ("student_std", student_std),
                           ("teacher_mean", native_mean), ("teacher_std", native_std)):
            self.register_buffer(key, torch.tensor(_triple(value, key)).view(1, 3, 1, 1), persistent=False)
        self._same_normalization = (
            _triple(student_mean, "student_mean") == _triple(native_mean, "teacher_mean")
            and _triple(student_std, "student_std") == _triple(native_std, "teacher_std")
        )
        self.model.requires_grad_(False).eval()
        # Cast before CUDA transfer to avoid an fp32 GPU allocation spike.
        self.model.to(dtype=dtype).to(device=device)
        for key in ("student_mean", "student_std", "teacher_mean", "teacher_std"):
            setattr(self, key, getattr(self, key).to(device=device))
        self.storage_dtype = dtype
        self.signature = {
            "name": self.name, "kind": self.kind,
            "weights_sha256": spec["weights_sha256"], "config_sha256": spec["config_sha256"],
            "code_sha256": dict(self.verified_code), "output_dim": self.output_dim,
            "img_size": self.img_size, "readout": "final_cls_concat_patch_mean",
            "mean": list(native_mean), "std": list(native_std),
            "native_preprocess": spec.get("native_preprocess", "published_model"),
            "checkpoint_branch": spec.get("checkpoint_branch", "teacher"),
            "provenance": provenance,
        }
        self.train(False)
        logger.info("[JOINT] teacher %s: kind=%s, output=%d, params=%d, dtype=%s",
                    self.name, self.kind, self.output_dim,
                    sum(p.numel() for p in self.model.parameters()), dtype)

    def _require_code(self, paths: Sequence[Path]) -> None:
        absent = [str(path.resolve()) for path in paths if str(path.resolve()) not in self.verified_code]
        if absent:
            raise ValueError(f"{self.name}: custom model code not pinned in manifest: {absent}")

    def _build(self) -> tuple[nn.Module, Sequence[float], Sequence[float], int]:
        if self.kind == "dinov3":
            from omegaconf import OmegaConf
            from dinov3.models import build_model

            cfg = OmegaConf.load(self.config_path)
            backbone, dim = build_model(cfg.student, only_teacher=True, img_size=self.img_size, device="cpu")
            backbone.init_weights()
            _load_backbone_strict(backbone, _torch_state(
                self.weights_path, checkpoint_branch=str(self.spec.get("checkpoint_branch", "teacher"))
            ), self.name)
            if self.spec.get("native_preprocess") == "checkpoint_config":
                if self.img_size != int(cfg.crops.global_crops_size):
                    raise ValueError(f"{self.name}: history image size must match its pinned training config")
                return backbone, list(cfg.crops.rgb_mean), list(cfg.crops.rgb_std), int(dim)
            return backbone, _IMAGENET_MEAN, _IMAGENET_STD, int(dim)

        config = json.loads(self.config_path.read_text())
        if self.kind == "timm_dir":
            import timm

            args = dict(config.get("model_args", {}))
            args.update(num_classes=0, img_size=self.img_size, dynamic_img_size=False)
            model = timm.create_model(config["architecture"], pretrained=False, **args)
            if self.weights_path.suffix == ".safetensors":
                from safetensors.torch import load_file
                state = load_file(str(self.weights_path))
            else:
                state = _torch_state(self.weights_path)
            _load_backbone_strict(model, state, self.name)
            self.num_prefix_tokens = int(getattr(model, "num_prefix_tokens", 1))
            pcfg = config["pretrained_cfg"]
            return model, pcfg["mean"], pcfg["std"], int(model.num_features)

        if self.kind == "genbio_pathfm":
            from transformers import AutoModel

            folder = self.config_path.parent
            if self.weights_path != (folder / "model.safetensors").resolve():
                raise ValueError("GenBio weights must be the pinned directory's model.safetensors")
            self._require_code([folder / "modeling_genbio_pathfm.py", folder / "configuration_genbio_pathfm.py"])
            model, loading = AutoModel.from_pretrained(
                str(folder), trust_remote_code=True, local_files_only=True,
                torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, output_loading_info=True,
            )
            problems = {key: loading.get(key) for key in
                        ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if loading.get(key)}
            if problems:
                raise RuntimeError(f"{self.name}: incomplete HF load: {problems}")
            if int(config["in_chans"]) != 1:
                raise ValueError("Unrecognized GenBio native channel encoder")
            # Its published RGB readout encodes R/G/B separately then concatenates.
            return model, config["image_mean"], config["image_std"], 3 * int(config["embed_dim"])

        if self.kind == "openpath":
            source = Path(self.spec["source_dir"]).resolve(strict=True)
            self._require_code([source / "dinov2/models/vision_transformer.py"])
            previous = sys.modules.get("dinov2")
            if previous is not None:
                actual = Path(previous.__file__).resolve()
                if source not in actual.parents:
                    raise RuntimeError(f"OpenPath needs its pinned dinov2 implementation, already imported {actual}")
            sys.path.insert(0, str(source))
            vits = importlib.import_module("dinov2.models.vision_transformer")
            if source not in Path(vits.__file__).resolve().parents:
                raise RuntimeError("OpenPath imported an unpinned vision_transformer implementation")
            model = vits.vit_giant2(
                patch_size=int(config["patch_size"]), img_size=self.img_size,
                block_chunks=int(config["block_chunks"]),
                num_register_tokens=int(config["num_register_tokens"]),
                ffn_layer=str(config["ffn_layer"]), init_values=float(config["init_values"]),
                interpolate_antialias=bool(config["interpolate_antialias"]),
                interpolate_offset=float(config["interpolate_offset"]),
            )
            _load_backbone_strict(model, _torch_state(self.weights_path), self.name)
            return model, _IMAGENET_MEAN, _IMAGENET_STD, int(config["embed_dim"])
        raise ValueError(f"Unsupported joint teacher kind: {self.kind}")

    def train(self, mode: bool = True) -> FrozenJointTeacher:
        # A parent's train() call must never enable teacher drop path or jitter.
        super().train(False)
        return self

    def preprocess(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"{self.name}: expected [B,3,H,W] normalized RGB, got {tuple(images.shape)}")
        if self._same_normalization and tuple(images.shape[-2:]) == (self.img_size, self.img_size):
            # In particular, keep the historical checkpoint's native input
            # bit-identical for calibration against the initial student.
            return images.float().contiguous()
        images = (images.float() * self.student_std.float() + self.student_mean.float()).clamp(0, 1)
        if tuple(images.shape[-2:]) != (self.img_size, self.img_size):
            images = F.interpolate(images, size=(self.img_size, self.img_size), mode="bicubic",
                                   align_corners=False, antialias=True).clamp_(0, 1)
        return ((images - self.teacher_mean.float()) / self.teacher_std.float()).contiguous()

    @torch.no_grad()
    def forward(self, images: Tensor, *, readout: str = "cls_patchmean") -> Tensor:
        if readout not in ("cls", "cls_patchmean"):
            raise ValueError(f"{self.name}: unsupported readout {readout}")
        output_dim = self.embed_dim if readout == "cls" else self.output_dim
        if len(images) == 0:
            return images.new_empty((0, output_dim), dtype=torch.float32)
        out = []
        for chunk in images.split(self.chunk_size):
            x = self.preprocess(chunk)
            with torch.autocast(device_type=x.device.type, dtype=self.storage_dtype,
                                enabled=self.storage_dtype in (torch.bfloat16, torch.float16)):
                if self.kind == "genbio_pathfm":
                    if readout == "cls":
                        cls = self.model(x)
                    else:
                        cls, patches = self.model.forward_with_patches(x)
                elif self.kind == "timm_dir":
                    tokens = self.model.forward_features(x)
                    cls = tokens[:, 0]
                    if readout != "cls":
                        patches = tokens[:, self.num_prefix_tokens:]
                else:
                    features = self.model.forward_features(x)
                    cls = features["x_norm_clstoken"]
                    if readout != "cls":
                        patches = features["x_norm_patchtokens"]
                features = cls.float() if readout == "cls" else torch.cat(
                    (cls.float(), patches.float().mean(dim=1)), dim=1
                )
            if features.shape != (len(chunk), output_dim):
                raise RuntimeError(f"{self.name}: unexpected readout shape {tuple(features.shape)}")
            if not torch.isfinite(features).all():
                raise FloatingPointError(f"{self.name}: nonfinite teacher features")
            out.append(features)
        return torch.cat(out, dim=0)


class JointTeacherPool(nn.Module):
    """Sequential teacher inference; concatenation order is fixed by the manifest."""

    def __init__(
        self,
        specs: Sequence[Mapping[str, Any]],
        *,
        student_mean: Sequence[float] = _IMAGENET_MEAN,
        student_std: Sequence[float] = _IMAGENET_STD,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        names = [str(spec["name"]) for spec in specs]
        if not names or len(names) != len(set(names)):
            raise ValueError("Teacher manifest must contain at least one unique teacher")
        self.teachers = nn.ModuleList([
            FrozenJointTeacher(spec, student_mean=student_mean, student_std=student_std,
                               device=device, dtype=dtype) for spec in specs
        ])
        self.names = names
        self.output_dims = [teacher.output_dim for teacher in self.teachers]
        self.output_dim = sum(self.output_dims)
        self.signatures = [teacher.signature for teacher in self.teachers]
        self.train(False)

    @classmethod
    def from_manifest(
        cls,
        path: str | Path,
        *,
        cfg: Any = None,
        student_mean: Sequence[float] = _IMAGENET_MEAN,
        student_std: Sequence[float] = _IMAGENET_STD,
        device: str | torch.device = "cpu",
        chunk_size: int | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> JointTeacherPool:
        manifest = json.loads(Path(path).read_text())
        specs = manifest["teachers"] if isinstance(manifest, Mapping) else manifest
        if chunk_size is not None:
            specs = [dict(spec, chunk_size=int(chunk_size)) for spec in specs]
        # cfg is accepted for training integration, but each teacher uses its
        # own manifest-pinned config rather than the mutable student's config.
        return cls(specs, student_mean=student_mean, student_std=student_std, device=device, dtype=dtype)

    def train(self, mode: bool = True) -> JointTeacherPool:
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        return torch.cat([teacher(images) for teacher in self.teachers], dim=1)

    @torch.no_grad()
    def forward_pair(self, first: Tensor, second: Tensor) -> tuple[Tensor, Tensor]:
        if first.shape != second.shape:
            raise ValueError("Paired teacher views must have the same shape and image ordering")
        # Pair within each teacher: no doubling of intermediate activation batches.
        first_features, second_features = [], []
        for teacher in self.teachers:
            first_features.append(teacher(first))
            second_features.append(teacher(second))
        return torch.cat(first_features, dim=1), torch.cat(second_features, dim=1)


def default_manifest(
    startpoint: str | Path, config: str | Path, *, alignment_branch: str = "teacher"
) -> dict[str, Any]:
    """Build and hash the four external teachers using main-building files.

    The caller persists this result once before calibration/training.  This
    function does not download/copy weights, load models, or use any test data.
    ``startpoint`` pins a separate, one-time initial-student coordinate
    alignment reference.  It is never included in teachers, their joint
    features/covariances, or the stable directions defining the target.
    """
    if alignment_branch not in ("student", "teacher"):
        raise ValueError(f"Unknown alignment branch: {alignment_branch}")
    benchmark = Path("/mnt/huawei_deepcad/benchmark_model/Pathology_FM")
    openpath_source = benchmark / "OpenPath-source/OpenPath"
    records = [
        ("dinov3l", "dinov3", Path("/mnt/huawei_deepcad/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
         Path(config), 2048, "DINO+iBOT", "https://github.com/facebookresearch/dinov3"),
        ("hoptimus0", "timm_dir", benchmark / "H-optimus-0/pytorch_model.bin",
         benchmark / "H-optimus-0/config.json", 3072, "DINOv2 image-only pretraining",
         "https://huggingface.co/bioptimus/H-optimus-0"),
        ("genbio_pathfm", "genbio_pathfm", benchmark / "GenBio-PathFM/model.safetensors",
         benchmark / "GenBio-PathFM/config.json", 9216, "JEDI: DINO then JEPA",
         "https://huggingface.co/genbio-ai/genbio-pathfm"),
        ("openpath", "openpath", benchmark / "OpenPath/teacher_checkpoint.pth",
         benchmark / "OpenPath/config.json", 3072, "DINO+iBOT+Gram anchoring",
         "https://huggingface.co/taejoon89/openpath"),
    ]
    teachers = []
    for name, kind, weights, cfg, dim, method, source in records:
        teacher = {
            "name": name, "kind": kind, "weights": str(weights.resolve(strict=True)),
            "weights_sha256": sha256_file(weights), "config": str(cfg.resolve(strict=True)),
            "config_sha256": sha256_file(cfg), "img_size": 224, "chunk_size": 2,
            "output_dim": dim,
            "provenance": {"training_supervision": "image_only_self_supervised",
                           "method": method, "source": source},
        }
        if kind == "genbio_pathfm":
            teacher["code_sha256"] = {str(p.resolve()): sha256_file(p) for p in (
                cfg.parent / "modeling_genbio_pathfm.py", cfg.parent / "configuration_genbio_pathfm.py")}
        elif kind == "openpath":
            teacher["source_dir"] = str(openpath_source.resolve(strict=True))
            teacher["code_sha256"] = {
                str(p.resolve()): sha256_file(p)
                for subtree in (openpath_source / "dinov2/layers", openpath_source / "dinov2/models")
                for p in sorted(subtree.rglob("*.py"))
            }
            teacher["provenance"]["known_evaluation_overlap"] = "PCam/CAMELYON pretraining overlap; exclude independent-generalization claim"
        teachers.append(teacher)
    from omegaconf import OmegaConf
    initial_config = OmegaConf.load(config)
    alignment_reference = {
        "name": "initial_student_coordinate_alignment_only", "kind": "dinov3",
        "weights": str(Path(startpoint).resolve(strict=True)), "weights_sha256": sha256_file(startpoint),
        "config": str(Path(config).resolve(strict=True)), "config_sha256": sha256_file(config),
        "img_size": int(initial_config.crops.global_crops_size), "chunk_size": 2, "output_dim": 2048,
        "native_preprocess": "checkpoint_config", "mean": list(initial_config.crops.rgb_mean),
        "std": list(initial_config.crops.rgb_std),
        "checkpoint_branch": alignment_branch,
        "usage": "orthogonal_coordinate_alignment_only_not_joint_teacher",
        "provenance": {"training_supervision": "image_only_self_supervised",
                       "method": "Initial student coordinate frame; not a joint target knowledge source",
                       "source": str(Path(config).resolve())},
    }
    return {"schema": "joint_image_ssl_teachers_v1", "readout": "final_cls_concat_patch_mean",
            "teachers": teachers, "output_dim": sum(t["output_dim"] for t in teachers),
            "alignment_reference": alignment_reference,
            "alignment_reference_note": "Used once for orthogonal alignment to initial student coordinates; "
                                        "excluded from teacher signatures, concatenation, V/C, and stable-direction selection."}
