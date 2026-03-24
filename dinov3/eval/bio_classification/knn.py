"""
生物医学图像 k-NN 分类评估脚本。

设计原则（遵循防腐层 + 代码复用架构）：
  1. 数据加载：直接使用 CamelyonPatchDataset，
     后者通过 bio_io.read_bio_image_as_numpy 屏蔽格式/位深/通道差异。
  2. 核心算法：严禁重写 k-NN 逻辑，全部复用：
       - dinov3.eval.utils.extract_features   （特征提取）
       - dinov3.eval.knn.eval_knn              （k-NN 分类）
       - dinov3.eval.data.get_num_classes      （自动检测类别数）
  3. 模型加载：通过 torch.hub.load（本地模式）复用官方 hub 入口。

典型用法：
  python -m dinov3.eval.bio_classification.knn \\
      --repo-dir /data1/xuzijing/biodino \\
      --weights   /data1/xuzijing/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \\
      --data-root /data1/zhusimiao/classifiction/camelyonpatch \\
      --output-dir ./outputs/camelyonpatch_knn \\
      --arch dinov3_vitl16 \\
      --batch-size 256 \\
      --num-workers 8 \\
      --ks 10 20 100 200
"""

import argparse
import json
import logging
import os
import sys
from functools import partial
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torchvision.transforms import v2

# ── 复用自然图像流水线的核心算法（绝对禁止重写）────────────────────────────
from dinov3.eval.data import create_train_dataset_dict, get_num_classes
from dinov3.eval.knn import TrainConfig as KnnTrainConfig
from dinov3.eval.knn import eval_knn
from dinov3.eval.utils import ModelWithNormalize, extract_features
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAveragePrecision,
    MulticlassF1Score,
)

# ── 新增生物数据集（防腐层出口）──────────────────────────────────────────────
from dinov3.eval.bio_classification.datasets.camelyonpatch import CamelyonPatchDataset
from dinov3.data import SamplerType, make_data_loader
from dinov3.data.adapters import DatasetWithEnumeratedTargets
from dinov3.data.transforms import (
    CROP_DEFAULT_SIZE,
    RESIZE_DEFAULT_SIZE,
    make_classification_eval_transform,
)
from dinov3.eval.data import pad_multilabel_and_collate
import dinov3.distributed as distributed

logger = logging.getLogger("dinov3.bio_knn")


# ─────────────────────────────────────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────────────────────────────────────

def load_model(repo_dir: str, arch: str, weights: str) -> torch.nn.Module:
    """
    通过 torch.hub.load（本地 source）加载 DINOv3 backbone。

    Args:
        repo_dir:  包含 hubconf.py 的仓库根目录（即 biodino/）。
        arch:      hub 入口名称，例如 "dinov3_vitl16"。
        weights:   本地 checkpoint 路径或远程 URL。

    Returns:
        已移到 CUDA 并切换为 eval 模式的 backbone。
    """
    logger.info(f"从 {repo_dir} 加载模型 {arch}，权重：{weights}")
    model = torch.hub.load(
        repo_dir,
        arch,
        source="local",
        weights=weights,
    )
    model.cuda()
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 数据集 & DataLoader 构建
# ─────────────────────────────────────────────────────────────────────────────

def build_transform(resize_size: int = RESIZE_DEFAULT_SIZE, crop_size: int = CROP_DEFAULT_SIZE):
    """
    构建与自然图像流水线完全相同的评估变换。

    bio_io 已将图像归一化到 [0, 1] float32，变换管道只需做：
      ToImage → Resize → CenterCrop → ToDtype(float32) → ImageNet Normalize
    """
    return make_classification_eval_transform(
        resize_size=resize_size,
        crop_size=crop_size,
    )


def build_datasets(data_root: str, transform, train_split: str = "train"):
    """
    构建训练集（用于建立 k-NN 特征库）和测试集（用于评估）。

    Args:
        data_root:   CamelyonPatch 数据集根目录。
        transform:   torchvision 变换（由 build_transform 构建）。
        train_split: 用于特征提取的划分，默认 "train"。

    Returns:
        train_dataset, test_dataset
    """
    train_dataset = CamelyonPatchDataset(
        root=data_root,
        split=train_split,
        transform=transform,
    )
    test_dataset = CamelyonPatchDataset(
        root=data_root,
        split="test",
        transform=transform,
    )
    logger.info(
        f"数据集构建完成：train={len(train_dataset)} 张，test={len(test_dataset)} 张"
    )
    return train_dataset, test_dataset


def build_test_data_loader(test_dataset, batch_size: int, num_workers: int):
    """
    构建测试集 DataLoader，使用分布式采样器（单卡时等价于顺序采样）。
    使用 DatasetWithEnumeratedTargets 包装以适配 eval_knn 的 evaluate 函数。
    """
    dataset_enum = DatasetWithEnumeratedTargets(
        test_dataset,
        pad_dataset=True,
        num_replicas=distributed.get_world_size(),
    )
    return make_data_loader(
        dataset=dataset_enum,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
        persistent_workers=(num_workers > 0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 主评估流程
# ─────────────────────────────────────────────────────────────────────────────

def run_bio_knn_eval(
    *,
    repo_dir: str,
    arch: str,
    weights: str,
    data_root: str,
    output_dir: str,
    ks: tuple = (10, 20, 100, 200),
    temperature: float = 0.07,
    batch_size: int = 256,
    num_workers: int = 8,
    resize_size: int = RESIZE_DEFAULT_SIZE,
    crop_size: int = CROP_DEFAULT_SIZE,
    train_split: str = "train",
    autocast_dtype: torch.dtype = torch.float,
) -> dict:
    """
    完整的生物医学图像 k-NN 评估流水线。

    流程：
      1. 加载模型（torch.hub + 本地 weights）
      2. 构建 CamelyonPatch train/test 数据集
      3. 提取训练集特征（复用 dinov3.eval.utils.extract_features）
      4. 构建测试集 DataLoader
      5. 运行 k-NN 评估（复用 dinov3.eval.knn.eval_knn）
      6. 打印并保存结果

    Returns:
        results_dict: {k: {"top-1": float, "top-5": float}, ...}
    """
    cudnn.benchmark = True
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. 模型 ──────────────────────────────────────────────────────────────
    model = load_model(repo_dir, arch, weights)
    model = ModelWithNormalize(model)  # L2 归一化 CLS token

    # ── 2. 变换 & 数据集 ─────────────────────────────────────────────────────
    transform = build_transform(resize_size=resize_size, crop_size=crop_size)
    train_dataset, test_dataset = build_datasets(data_root, transform, train_split)

    # ── 3. 提取训练集特征 ─────────────────────────────────────────────────────
    logger.info("开始提取训练集特征...")
    train_dataset_dict = create_train_dataset_dict(train_dataset)  # {0: dataset}

    with torch.autocast("cuda", dtype=autocast_dtype):
        few_shot_features_dict = {}
        for try_n, dataset in train_dataset_dict.items():
            features, labels = extract_features(
                model, dataset, batch_size, num_workers, gather_on_cpu=True
            )
            few_shot_features_dict[try_n] = {
                "train_features": features,
                "train_labels": labels,
            }

    # ── 4. 测试集 DataLoader ──────────────────────────────────────────────────
    test_data_loader = build_test_data_loader(test_dataset, batch_size, num_workers)

    # ── 5. 构建评估指标 & 运行 k-NN ──────────────────────────────────────────
    num_classes = get_num_classes(train_dataset)
    # 同时计算 Accuracy、F1（macro）、AUPRC（macro）
    metric_collection = MetricCollection({
        "top-1":  MulticlassAccuracy(top_k=1, num_classes=num_classes, average="macro"),
        "f1":     MulticlassF1Score(num_classes=num_classes, average="macro"),
        "auprc":  MulticlassAveragePrecision(num_classes=num_classes, average="macro"),
    })

    knn_train_config = KnnTrainConfig(
        dataset="",        # 本流程不使用 make_dataset，传空字符串占位
        batch_size=batch_size,
        num_workers=num_workers,
        ks=tuple(ks),
        temperature=temperature,
        skip_first_nn=False,
    )

    with torch.autocast("cuda", dtype=autocast_dtype):
        results_dict_knn = eval_knn(
            model=model,
            train_data_dict=few_shot_features_dict,
            test_data_loader=test_data_loader,
            metric_collection=metric_collection,
            knn_config=knn_train_config,
            num_classes=num_classes,
        )

    # ── 6. 格式化结果并保存 ───────────────────────────────────────────────────
    results_flat = {}
    for k, metrics in results_dict_knn.items():
        top1  = metrics.get("top-1",  float("nan"))
        f1    = metrics.get("f1",     float("nan"))
        auprc = metrics.get("auprc",  float("nan"))
        results_flat[f"{k}-NN Accuracy (%)"] = top1
        results_flat[f"{k}-NN F1 (%)"]       = f1
        results_flat[f"{k}-NN AUPRC (%)"]    = auprc
        logger.info(
            f"k={k:>4d}  Accuracy: {top1:.2f}%  F1: {f1:.2f}%  AUPRC: {auprc:.2f}%"
        )

    out_path = os.path.join(output_dir, "results_bio_knn.json")
    with open(out_path, "w") as f:
        json.dump(results_flat, f, indent=2)
    logger.info(f"结果已保存至 {out_path}")

    return results_flat


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="CamelyonPatch k-NN 分类评估（DINOv3 防腐层架构）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 模型 ──
    parser.add_argument(
        "--repo-dir",
        type=str,
        default="/data1/xuzijing/biodino",
        help="包含 hubconf.py 的仓库根目录（即 biodino/）",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="dinov3_vitl16",
        help="torch.hub 入口名称，例如 dinov3_vitl16、dinov3_vitb16",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="/data1/xuzijing/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        help="本地 checkpoint 路径或远程 URL",
    )

    # ── 数据 ──
    parser.add_argument(
        "--data-root",
        type=str,
        default="/data1/zhusimiao/classifiction/camelyonpatch",
        help="CamelyonPatch 数据集根目录",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        choices=["train", "valid"],
        help="用于构建 k-NN 特征库的划分（通常为 train）",
    )

    # ── 输出 ──
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/camelyonpatch_knn",
        help="结果保存目录",
    )

    # ── k-NN 超参 ──
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[10, 20, 100, 200],
        help="k-NN 的 k 值列表",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="k-NN softmax 温度",
    )

    # ── 数据加载 ──
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resize-size", type=int, default=RESIZE_DEFAULT_SIZE)
    parser.add_argument("--crop-size", type=int, default=CROP_DEFAULT_SIZE)

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # 初始化分布式进程组。
    # TorchDistributedEnvironment 会自动识别单节点手动启动场景（world_size=1, rank=0），
    # 也兼容 torchrun 多卡启动。DistributedSampler 依赖 torch.distributed 已初始化。
    distributed.enable(set_cuda_current_device=True, overwrite=True)

    run_bio_knn_eval(
        repo_dir=args.repo_dir,
        arch=args.arch,
        weights=args.weights,
        data_root=args.data_root,
        output_dir=args.output_dir,
        ks=tuple(args.ks),
        temperature=args.temperature,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resize_size=args.resize_size,
        crop_size=args.crop_size,
        train_split=args.train_split,
    )


if __name__ == "__main__":
    main()
