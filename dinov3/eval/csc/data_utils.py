"""
CSC 数据集数据加载工具

支持多种图像格式（tif, tiff, png, bmp）和位深（8bit, 16bit）
"""

import os
from glob import glob
from typing import List, Tuple
import cv2
import numpy as np
import tifffile


def load_image_multi_format(img_path: str) -> np.ndarray:
    """
    加载多种格式的图像文件（支持 tif, tiff, png, bmp, jpg）
    
    Args:
        img_path: 图像文件路径
    
    Returns:
        图像数组，保持原始 dtype 和格式
    """
    ext = os.path.splitext(img_path)[1].lower()
    
    if ext in ['.tif', '.tiff']:
        # 使用 tifffile 读取 TIFF 文件（可以处理 16-bit）
        img = tifffile.imread(img_path)
    else:
        # 使用 cv2 读取其他格式
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")
    
    # cv2 读取的 RGB 图像是 BGR 格式，需要转换
    if len(img.shape) == 3 and img.shape[2] == 3 and ext not in ['.tif', '.tiff']:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img


def load_mask_tiff(mask_path: str) -> np.ndarray:
    """
    加载 TIFF 格式的 mask 文件（instance segmentation mask）
    
    Args:
        mask_path: mask 文件路径
    
    Returns:
        instance mask（uint16），每个实例有唯一的 ID
    """
    mask = tifffile.imread(mask_path)
    if mask is None:
        raise ValueError(f"无法读取 mask: {mask_path}")
    return mask


def instance_mask_to_binary(mask: np.ndarray) -> np.ndarray:
    """
    将 instance segmentation mask 转换为二值 mask
    
    Args:
        mask: instance mask，背景为 0，每个实例有唯一 ID
    
    Returns:
        二值 mask（0: 背景, 1: 前景）
    """
    return (mask > 0).astype(np.int64)


def get_base_name(fname: str) -> str:
    """
    从文件名提取基础名称（用于匹配图像和标签）
    
    Examples:
        cell_00001.bmp -> cell_00001
        cell_00001_label.tiff -> cell_00001
    """
    base = fname.rsplit('_label', 1)[0]
    for ext in ['.bmp', '.png', '.tif', '.tiff', '.jpg', '.jpeg']:
        base = base.replace(ext, '')
    return base


def get_csc_dataset_paths(data_root: str, split: str = 'train') -> Tuple[List[str], List[str]]:
    """
    获取 CSC 数据集的图像和掩码路径
    
    CSC 数据集结构：
    - Training-labeled/Training-labeled/images/ 和 labels/
    - Tuning/Tuning/images/ 和 labels/
    - Testing/Testing/Public/images/ 和 labels/
    - Testing/Testing/Hidden/images/ 和 osilab_seg/osilab_seg/（标签）
    
    Args:
        data_root: 数据集根目录（例如 /mnt/deepcad_nfs/0-large-model-dataset/56-CSC）
        split: 'train', 'tune', 'test_public', 或 'test_hidden'
    
    Returns:
        (img_paths, mask_paths) - 匹配的图像和标签路径列表
    """
    if split == 'train':
        img_dir = os.path.join(data_root, "Training-labeled", "Training-labeled", "images")
        label_dir = os.path.join(data_root, "Training-labeled", "Training-labeled", "labels")
    elif split == 'tune':
        img_dir = os.path.join(data_root, "Tuning", "Tuning", "images")
        label_dir = os.path.join(data_root, "Tuning", "Tuning", "labels")
    elif split == 'test' or split == 'test_public':
        # Public 测试集：images 和 labels 都在 Public 目录下
        img_dir = os.path.join(data_root, "Testing", "Testing", "Public", "images")
        label_dir = os.path.join(data_root, "Testing", "Testing", "Public", "labels")
    elif split == 'test_hidden':
        # Hidden 测试集：images 在 Hidden/images，labels 在 Hidden/osilab_seg/osilab_seg
        img_dir = os.path.join(data_root, "Testing", "Testing", "Hidden", "images")
        label_dir = os.path.join(data_root, "Testing", "Testing", "Hidden", "osilab_seg", "osilab_seg")
    else:
        raise ValueError(f"不支持的 split: {split}，应该是 'train', 'tune', 'test'/'test_public', 或 'test_hidden'")
    
    if not os.path.exists(img_dir):
        raise ValueError(f"图像目录不存在: {img_dir}")
    
    # 获取所有图像文件（支持多种格式）
    img_files = []
    for ext in ['*.tif', '*.tiff', '*.png', '*.bmp', '*.jpg', '*.jpeg']:
        img_files.extend(glob(os.path.join(img_dir, ext)))
        img_files.extend(glob(os.path.join(img_dir, ext.upper())))
    
    img_files = [f for f in img_files if not os.path.basename(f).startswith('.')]
    img_files = sorted(img_files)
    
    if not os.path.exists(label_dir):
        raise ValueError(f"标签目录不存在: {label_dir}")
    
    # 获取所有标签文件（通常是 tiff）
    label_files = glob(os.path.join(label_dir, '*_label.tiff'))
    label_files.extend(glob(os.path.join(label_dir, '*_label.tif')))
    label_files = sorted(label_files)
    
    # 匹配图像和标签
    img_base_to_path = {get_base_name(os.path.basename(f)): f for f in img_files}
    label_base_to_path = {get_base_name(os.path.basename(f)): f for f in label_files}
    
    matched_bases = set(img_base_to_path.keys()) & set(label_base_to_path.keys())
    
    img_paths = [img_base_to_path[base] for base in sorted(matched_bases)]
    mask_paths = [label_base_to_path[base] for base in sorted(matched_bases)]
    
    print(f"[CSC {split}] 找到 {len(img_files)} 张图像，{len(label_files)} 个标签，匹配 {len(img_paths)} 对")
    
    return img_paths, mask_paths

