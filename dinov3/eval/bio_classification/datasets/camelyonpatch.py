"""
CamelyonPatch（PCam）数据集加载类。

数据目录约定（遵循官方 PCam 格式）：
    <root>/
      train/
        train_000000.png
        train_000001.png
        ...
        camelyonpatch_level_2_split_train_y.h5
      valid/
        valid_000000.png
        ...
        camelyonpatch_level_2_split_valid_y.h5
      test/
        test_000000.png
        ...
        camelyonpatch_level_2_split_test_y.h5

标签文件格式：HDF5，key='y'，shape=(N, 1, 1, 1)，dtype=uint8，值为 0 或 1。

图像读取：通过 dinov3.utils.bio_io.read_bio_image_as_numpy（防腐层）完成，
保证返回 (H, W, 3) float32 [0, 1] 的 numpy 数组，
再交给 torchvision v2 变换管道（ToImage → resize/crop → normalize）处理。
"""

import logging
import os
from typing import Callable, List, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from dinov3.utils.bio_io import read_bio_image_as_numpy

logger = logging.getLogger(__name__)

_SPLIT_META = {
    "train": ("train", "camelyonpatch_level_2_split_train_y.h5"),
    "valid": ("valid", "camelyonpatch_level_2_split_valid_y.h5"),
    "test":  ("test",  "camelyonpatch_level_2_split_test_y.h5"),
}


def _load_labels_from_h5(h5_path: str) -> np.ndarray:
    """
    从官方 PCam HDF5 标签文件读取 1-D 标签数组。

    HDF5 内 key='y'，shape=(N, 1, 1, 1)，压缩为 (N,) int64 返回。
    """
    with h5py.File(h5_path, "r") as f:
        labels = f["y"][:].reshape(-1).astype(np.int64)
    return labels


def _collect_image_paths(split_dir: str, prefix: str, n: int) -> List[str]:
    """
    按照官方命名规则 {prefix}_{idx:06d}.png 构建路径列表，并做存在性校验。
    """
    paths = []
    for i in range(n):
        fname = f"{prefix}_{i:06d}.png"
        fpath = os.path.join(split_dir, fname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(
                f"找不到图像文件 {fpath}。"
                f"请检查数据根目录或命名规则是否正确。"
            )
        paths.append(fpath)
    return paths


class CamelyonPatchDataset(Dataset):
    """
    PCam 二分类数据集（肿瘤组织 vs 正常组织）。

    Args:
        root:        数据集根目录，包含 train/valid/test 三个子目录。
        split:       数据集划分，可选 "train" | "valid" | "test"。
        transform:   图像变换（接受 numpy HWC float32 或 PIL Image，
                     建议直接使用 dinov3 的 make_classification_eval_transform）。
        target_channels: 读取图像时的目标通道数，默认 3（RGB）。
    """

    NUM_CLASSES: int = 2
    CLASS_NAMES: List[str] = ["normal", "tumor"]

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_channels: int = 3,
    ) -> None:
        super().__init__()

        if split not in _SPLIT_META:
            raise ValueError(
                f"split 必须是 {list(_SPLIT_META.keys())} 之一，当前为 '{split}'"
            )

        self.root = root
        self.split = split
        self.transform = transform
        self.target_channels = target_channels

        subdir_name, label_filename = _SPLIT_META[split]
        split_dir = os.path.join(root, subdir_name)
        h5_path = os.path.join(split_dir, label_filename)

        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"数据目录不存在：{split_dir}")
        if not os.path.isfile(h5_path):
            raise FileNotFoundError(f"标签文件不存在：{h5_path}")

        self._labels: np.ndarray = _load_labels_from_h5(h5_path)
        n = len(self._labels)
        self._img_paths: List[str] = _collect_image_paths(split_dir, subdir_name, n)

        logger.info(
            f"[CamelyonPatch] split={split}，共 {n} 张图像，"
            f"正样本（tumor）{int(self._labels.sum())} 张，"
            f"负样本（normal）{n - int(self._labels.sum())} 张。"
        )

    # ------------------------------------------------------------------
    # 与 dinov3.eval.data.get_labels / get_num_classes 兼容的接口
    # ------------------------------------------------------------------

    def get_targets(self) -> np.ndarray:
        """返回全量标签数组（np.ndarray，shape=(N,)，dtype=int64）。"""
        return self._labels

    def get_target(self, index: int) -> int:
        return int(self._labels[index])

    # ------------------------------------------------------------------
    # torch.utils.data.Dataset 标准接口
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int):
        """
        返回 (image_tensor, label)。

        image_tensor 经过 transform 后为 [3, H, W] float32（已被 ImageNet 均值方差归一化）。
        label 为 int（0 = normal，1 = tumor）。
        """
        # 防腐层：通过 bio_io 读取，屏蔽格式/位深/通道差异
        img_np = read_bio_image_as_numpy(
            self._img_paths[index],
            target_channels=self.target_channels,
            normalize=True,
        )  # (H, W, C) float32 [0, 1]

        if self.transform is not None:
            # torchvision v2 变换接受 numpy (H, W, C) float32
            img = self.transform(img_np)
        else:
            # 未指定变换时直接转为 [C, H, W] Tensor
            img = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()

        label = int(self._labels[index])
        return img, label
