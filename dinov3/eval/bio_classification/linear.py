"""
生物医学图像 Linear Probe 评估脚本（CamelyonPatch）。

核心优化：特征预提取（Pre-extraction）
  Backbone 只做一次前向传播（每个 split），结果缓存到 CPU RAM，
  线性头仅在缓存特征上训练，彻底消除重复的大模型推理开销。

  时间复杂度对比（以 ViT-7B，262K train 为例）：
    在线推理版：epochs × steps × backbone_time  ≈ 3 天
    预提取版：  1 × extraction_steps × backbone_time + 瞬间训练  ≈ 数小时

设计原则（同 knn.py，防腐层 + 代码复用架构）：
  1. 数据加载：直接使用 CamelyonPatchDataset（bio_io 防腐层）。
  2. 核心算法：复用 dinov3.eval.linear 的：
       - create_linear_input    （特征拼接）
       - LinearClassifier       （线性分类头）
       - setup_linear_classifiers（LR 网格搜索初始化）
  3. 评估器：自定义 BioLinearEvaluator，操作预提取特征，无需再跑 backbone。

典型用法：
  python -m dinov3.eval.bio_classification.linear \\
      --repo-dir  /data1/xuzijing/biodino \\
      --arch      dinov3_vit7b16 \\
      --weights   /data1/xuzijing/checkpoints/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth \\
      --data-root /data1/zhusimiao/classifiction/camelyonpatch \\
      --output-dir ./outputs/camelyonpatch_linear \\
      --epochs 10 \\
      --batch-size 256
"""

import argparse
import json
import logging
import os
import sys
from functools import partial
from typing import Optional, Tuple

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAveragePrecision,
    MulticlassF1Score,
)

# ── 复用自然图像流水线的核心算法（绝对禁止重写）────────────────────────────
from dinov3.eval.linear import (
    LinearClassifier,
    create_linear_input,
    scale_lr,
)
from dinov3.eval.utils import ModelWithIntermediateLayers
from dinov3.data.transforms import (
    CROP_DEFAULT_SIZE,
    RESIZE_DEFAULT_SIZE,
    make_classification_eval_transform,
    make_classification_train_transform,
)
import dinov3.distributed as distributed
from dinov3.logging import MetricLogger

# ── 新增生物数据集（防腐层出口）──────────────────────────────────────────────
from dinov3.eval.bio_classification.datasets.camelyonpatch import CamelyonPatchDataset

logger = logging.getLogger("dinov3.bio_linear")


# ─────────────────────────────────────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────────────────────────────────────

def load_model(repo_dir: str, arch: str, weights: str) -> torch.nn.Module:
    logger.info(f"从 {repo_dir} 加载模型 {arch}，权重：{weights}")
    model = torch.hub.load(repo_dir, arch, source="local", weights=weights)
    model.cuda()
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 数据集构建
# ─────────────────────────────────────────────────────────────────────────────

def build_train_dataset(data_root: str, train_split: str = "train") -> CamelyonPatchDataset:
    """训练集：随机裁剪 + 水平翻转（增强特征多样性，抑制过拟合）。"""
    transform = make_classification_train_transform(crop_size=CROP_DEFAULT_SIZE)
    return CamelyonPatchDataset(root=data_root, split=train_split, transform=transform)


def build_eval_dataset(data_root: str, split: str) -> CamelyonPatchDataset:
    """验证 / 测试集：中心裁剪，不做随机增强。"""
    transform = make_classification_eval_transform(
        resize_size=RESIZE_DEFAULT_SIZE, crop_size=CROP_DEFAULT_SIZE
    )
    return CamelyonPatchDataset(root=data_root, split=split, transform=transform)


# ─────────────────────────────────────────────────────────────────────────────
# 核心优化：特征预提取
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def extract_features_to_cpu(
    feature_model: ModelWithIntermediateLayers,
    dataset,
    n_last_blocks: int,
    use_avgpool: bool,
    batch_size: int,
    num_workers: int,
    split_name: str = "",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    一次性提取全部图像的线性头输入特征，结果存入 CPU RAM。

    backbone 只在这里跑一次；后续所有 LR 候选的线性头训练都在缓存特征上进行，
    完全不再调用 backbone。

    Returns:
        features: [N, D] float32, CPU
        labels:   [N]   int64,   CPU
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    all_features, all_labels = [], []
    metric_logger = MetricLogger(delimiter="  ")
    logger.info(f"[{split_name}] 开始预提取特征，共 {len(dataset)} 张图像...")

    for images, labels in metric_logger.log_every(loader, 20, f"提取特征 [{split_name}]"):
        inter = feature_model(images.cuda())           # list of (patch_tokens, cls_token)
        feat = create_linear_input(inter, use_n_blocks=n_last_blocks, use_avgpool=use_avgpool)
        all_features.append(feat.float().cpu())
        all_labels.append(
            labels if isinstance(labels, torch.Tensor) else torch.tensor(labels)
        )

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0).long()
    logger.info(
        f"[{split_name}] 特征提取完成：features={features.shape}，labels={labels.shape}"
    )
    return features, labels


# ─────────────────────────────────────────────────────────────────────────────
# 在缓存特征上训练线性头（极快，通常 < 1 分钟）
# ─────────────────────────────────────────────────────────────────────────────

def train_linear_on_features(
    classifier: nn.Linear,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    lr: float,
    epochs: int,
    batch_size: int,
) -> nn.Linear:
    """
    在预提取的特征上训练单个线性分类头，使用 SGD + Cosine LR。

    Args:
        classifier:     未训练的 nn.Linear（已在 GPU 上）。
        train_features: [N, D] float32，CPU。
        train_labels:   [N]   int64，CPU。
        lr:             初始学习率（已被 scale_lr 缩放）。
        epochs:         训练轮数。
        batch_size:     mini-batch 大小。

    Returns:
        训练完毕的 classifier。
    """
    dataset = TensorDataset(train_features, train_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    max_iter = epochs * len(loader)

    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=0)
    criterion = nn.CrossEntropyLoss()

    classifier.train()
    for _ in range(epochs):
        for feats, lbls in loader:
            feats = feats.cuda(non_blocking=True)
            lbls = lbls.cuda(non_blocking=True)

            logits = classifier(feats)
            loss = criterion(logits, lbls)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    classifier.eval()
    return classifier


# ─────────────────────────────────────────────────────────────────────────────
# 在缓存特征上评估
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def evaluate_on_features(
    classifier: nn.Linear,
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 2,
    batch_size: int = 512,
) -> dict:
    """
    返回包含 Accuracy、F1（macro）、AUPRC（macro）三个指标的字典（均为百分比）。

    - Accuracy：Top-1 准确率（macro-averaged，即各类取平均）
    - F1：macro-averaged F1-score（对二分类等价于各类 F1 的算术平均）
    - AUPRC：precision-recall 曲线下面积（macro-averaged），
             衡量模型在不同阈值下的综合 precision/recall 性能，
             对类别不平衡数据比 AUROC 更敏感。
    """
    metrics = MetricCollection({
        "accuracy": MulticlassAccuracy(top_k=1, num_classes=num_classes, average="macro"),
        "f1":       MulticlassF1Score(num_classes=num_classes, average="macro"),
        "auprc":    MulticlassAveragePrecision(num_classes=num_classes, average="macro"),
    }).cuda()

    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for feats, lbls in loader:
        feats = feats.cuda(non_blocking=True)
        lbls  = lbls.cuda(non_blocking=True)
        # softmax 概率用于 F1 和 AUPRC；MulticlassAccuracy 内部做 argmax
        probs = torch.softmax(classifier(feats), dim=1)
        metrics.update(probs, lbls)

    return {k: v.item() * 100.0 for k, v in metrics.compute().items()}


# ─────────────────────────────────────────────────────────────────────────────
# LR 网格搜索
# ─────────────────────────────────────────────────────────────────────────────

def grid_search_lr(
    *,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    feature_dim: int,
    num_classes: int,
    learning_rates: Tuple[float, ...],
    epochs: int,
    batch_size: int,
) -> Tuple[float, nn.Linear, dict]:
    """
    在所有 LR 候选上依次训练线性头，在 valid 集选最优（以 Accuracy 为准则）。

    Returns:
        best_lr:           最优学习率（原始值，未缩放）。
        best_classifier:   已训练的最优线性头。
        best_val_metrics:  最优 LR 对应的 valid 三项指标字典（Accuracy/F1/AUPRC）。
    """
    best_lr = None
    best_val_acc = -1.0
    best_classifier = None
    best_val_metrics: dict = {}

    for raw_lr in learning_rates:
        lr = scale_lr(raw_lr, batch_size)  # 复用原始 scale_lr，与 linear.py 保持一致
        logger.info(f"尝试 lr={raw_lr:.2e}（缩放后 {lr:.5f}）...")

        clf = nn.Linear(feature_dim, num_classes).cuda()
        nn.init.normal_(clf.weight, mean=0.0, std=0.01)
        nn.init.zeros_(clf.bias)

        clf = train_linear_on_features(
            clf, train_features, train_labels,
            lr=lr, epochs=epochs, batch_size=batch_size,
        )

        val_metrics = evaluate_on_features(clf, val_features, val_labels, num_classes=num_classes)
        logger.info(
            f"  lr={raw_lr:.2e} → "
            f"Accuracy: {val_metrics['accuracy']:.2f}%  "
            f"F1: {val_metrics['f1']:.2f}%  "
            f"AUPRC: {val_metrics['auprc']:.2f}%"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_lr = raw_lr
            best_classifier = clf
            best_val_metrics = val_metrics

    logger.info(
        f"最优 lr={best_lr:.2e}  "
        f"valid Accuracy={best_val_metrics['accuracy']:.2f}%  "
        f"F1={best_val_metrics['f1']:.2f}%  "
        f"AUPRC={best_val_metrics['auprc']:.2f}%"
    )
    return best_lr, best_classifier, best_val_metrics


# ─────────────────────────────────────────────────────────────────────────────
# 主评估流程
# ─────────────────────────────────────────────────────────────────────────────

def run_bio_linear_eval(
    *,
    repo_dir: str,
    arch: str,
    weights: str,
    data_root: str,
    output_dir: str,
    epochs: int = 10,
    batch_size: int = 256,
    num_workers: int = 8,
    n_last_blocks: int = 1,
    use_avgpool: bool = True,
    learning_rates: Tuple[float, ...] = (
        1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1
    ),
    train_split: str = "train",
    val_split: str = "valid",
    autocast_dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """
    完整的生物医学图像 Linear Probe 评估流水线（预提取特征版）。

    流程：
      ① 加载模型
      ② 一次性提取 train / valid / test 特征到 CPU RAM（backbone 只跑这一次）
      ③ LR 网格搜索：在 valid 集上比较各候选 LR 的线性头
      ④ 用最优 LR 对应的分类头在 test 集报最终结果
      ⑤ 保存结果
    """
    cudnn.benchmark = True
    os.makedirs(output_dir, exist_ok=True)

    # ── ① 模型 ──────────────────────────────────────────────────────────────
    model = load_model(repo_dir, arch, weights)
    autocast_ctx = partial(torch.autocast, device_type="cuda", enabled=True, dtype=autocast_dtype)
    feature_model = ModelWithIntermediateLayers(
        model, n=n_last_blocks, autocast_ctx=autocast_ctx
    )

    # ── ② 数据集 & 特征预提取 ────────────────────────────────────────────────
    train_dataset = build_train_dataset(data_root, train_split)
    val_dataset   = build_eval_dataset(data_root, val_split)
    test_dataset  = build_eval_dataset(data_root, "test")
    num_classes   = CamelyonPatchDataset.NUM_CLASSES  # 2

    logger.info("=" * 60)
    logger.info("阶段 1/3：特征预提取（backbone 仅此一次）")
    logger.info("=" * 60)

    train_features, train_labels = extract_features_to_cpu(
        feature_model, train_dataset, n_last_blocks, use_avgpool,
        batch_size, num_workers, split_name="train",
    )
    val_features, val_labels = extract_features_to_cpu(
        feature_model, val_dataset, n_last_blocks, use_avgpool,
        batch_size, num_workers, split_name="valid",
    )
    test_features, test_labels = extract_features_to_cpu(
        feature_model, test_dataset, n_last_blocks, use_avgpool,
        batch_size, num_workers, split_name="test",
    )

    feature_dim = train_features.shape[1]
    logger.info(f"特征维度：{feature_dim}，类别数：{num_classes}")

    # ── ③ LR 网格搜索 ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("阶段 2/3：LR 网格搜索（linear head 训练，极快）")
    logger.info("=" * 60)

    best_lr, best_clf, best_val_metrics = grid_search_lr(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        feature_dim=feature_dim,
        num_classes=num_classes,
        learning_rates=tuple(learning_rates),
        epochs=epochs,
        batch_size=batch_size,
    )

    # ── ④ Test 集最终评估 ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("阶段 3/3：Test 集最终评估")
    logger.info("=" * 60)

    test_metrics = evaluate_on_features(
        best_clf, test_features, test_labels, num_classes=num_classes
    )
    logger.info(
        f"Test  Accuracy: {test_metrics['accuracy']:.2f}%  "
        f"F1: {test_metrics['f1']:.2f}%  "
        f"AUPRC: {test_metrics['auprc']:.2f}%  "
        f"(best_lr={best_lr:.2e})"
    )

    # ── ⑤ 保存结果 ────────────────────────────────────────────────────────────
    results = {
        "valid_accuracy":  best_val_metrics["accuracy"],
        "valid_f1":        best_val_metrics["f1"],
        "valid_auprc":     best_val_metrics["auprc"],
        "test_accuracy":   test_metrics["accuracy"],
        "test_f1":         test_metrics["f1"],
        "test_auprc":      test_metrics["auprc"],
        "best_lr":         best_lr,
        "feature_dim":     feature_dim,
        "n_last_blocks":   n_last_blocks,
        "use_avgpool":     use_avgpool,
        "epochs":          epochs,
    }

    out_path = os.path.join(output_dir, "results_bio_linear.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"结果已保存至 {out_path}")

    # 保存最优分类头权重（便于后续直接加载）
    clf_path = os.path.join(output_dir, "best_linear_classifier.pth")
    torch.save(best_clf.state_dict(), clf_path)
    logger.info(f"最优分类头权重已保存至 {clf_path}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="CamelyonPatch Linear Probe 评估（预提取特征版，DINOv3 防腐层架构）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 模型 ──
    parser.add_argument("--repo-dir", type=str, default="/data1/xuzijing/biodino")
    parser.add_argument("--arch", type=str, default="dinov3_vitl16")
    parser.add_argument(
        "--weights", type=str,
        default="/data1/xuzijing/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    )

    # ── 数据 ──
    parser.add_argument(
        "--data-root", type=str,
        default="/data1/zhusimiao/classifiction/camelyonpatch",
    )
    parser.add_argument("--train-split", type=str, default="train", choices=["train", "valid"])
    parser.add_argument("--val-split",   type=str, default="valid", choices=["train", "valid"])

    # ── 输出 ──
    parser.add_argument("--output-dir", type=str, default="./outputs/camelyonpatch_linear")

    # ── 超参 ──
    parser.add_argument("--epochs",      type=int, default=10,  help="线性头训练轮数")
    parser.add_argument("--batch-size",  type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--n-last-blocks", type=int, default=1,
        help="使用 backbone 最后 N 个 block 的输出拼接",
    )
    parser.add_argument(
        "--no-avgpool", action="store_true",
        help="不使用 patch token 的均值池化（默认开启 avgpool）",
    )
    parser.add_argument(
        "--learning-rates", type=float, nargs="+",
        default=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1],
        help="LR 网格搜索候选列表",
    )

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # 初始化分布式进程组（单卡 / 多卡 torchrun 均兼容）
    distributed.enable(set_cuda_current_device=True, overwrite=True)

    run_bio_linear_eval(
        repo_dir=args.repo_dir,
        arch=args.arch,
        weights=args.weights,
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_last_blocks=args.n_last_blocks,
        use_avgpool=not args.no_avgpool,
        learning_rates=tuple(args.learning_rates),
        train_split=args.train_split,
        val_split=args.val_split,
    )


if __name__ == "__main__":
    main()
