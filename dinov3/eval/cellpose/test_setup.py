#!/usr/bin/env python3
"""
测试 Cellpose Linear Probing 环境设置

运行此脚本检查所有依赖和数据是否就绪：
    cd /mnt/huawei_deepcad/dinov3
    python -m dinov3.eval.cellpose.test_setup
"""

import os
import sys
from pathlib import Path

# 获取项目根目录
DINOV3_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(DINOV3_ROOT))


def check_dependencies():
    """检查 Python 依赖"""
    print("=" * 60)
    print("1. 检查 Python 依赖")
    print("=" * 60)
    
    dependencies = {
        'torch': None,
        'numpy': None,
        'cv2': 'opencv-python',
        'tqdm': None,
        'matplotlib': None,
    }
    
    missing = []
    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            pkg = package or module
            missing.append(pkg)
            print(f"  ✗ {module} (pip install {pkg})")
    
    if missing:
        print(f"\n请安装缺失的包: pip install {' '.join(missing)}")
        return False
    return True


def check_cuda():
    """检查 CUDA 可用性"""
    print("\n" + "=" * 60)
    print("2. 检查 CUDA")
    print("=" * 60)
    
    import torch
    
    if torch.cuda.is_available():
        print(f"  ✓ CUDA 可用")
        print(f"    - 设备: {torch.cuda.get_device_name(0)}")
        print(f"    - 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("  ✗ CUDA 不可用")
        print("    警告: 此实验需要 GPU 才能有效运行")
        return False


def check_checkpoints():
    """检查模型 checkpoint"""
    print("\n" + "=" * 60)
    print("3. 检查 DINOv3 Checkpoints")
    print("=" * 60)
    
    checkpoint_dir = "/mnt/deepcad_nfs/xuzijing/checkpoints"
    checkpoints = {
        'ViT-L': 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
        'ViT-7B': 'dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth',
    }
    
    available = []
    for name, filename in checkpoints.items():
        path = os.path.join(checkpoint_dir, filename)
        if os.path.exists(path):
            size_gb = os.path.getsize(path) / 1e9
            print(f"  ✓ {name}: {filename} ({size_gb:.1f} GB)")
            available.append(name)
        else:
            print(f"  ✗ {name}: {filename} (不存在)")
    
    return len(available) > 0


def check_data():
    """检查 Cellpose 数据集"""
    print("\n" + "=" * 60)
    print("4. 检查 Cellpose 数据集")
    print("=" * 60)
    
    data_path = "/mnt/deepcad_nfs/0-large-model-dataset/11-Cellpose"
    
    if not os.path.exists(data_path):
        print(f"  ✗ 数据目录不存在: {data_path}")
        return False
    
    # 检查训练集
    train_dir = os.path.join(data_path, "train", "train")
    if not os.path.exists(train_dir):
        train_dir = os.path.join(data_path, "train")
    
    if os.path.exists(train_dir):
        train_imgs = len([f for f in os.listdir(train_dir) if f.endswith('_img.png')])
        train_masks = len([f for f in os.listdir(train_dir) if f.endswith('_masks.png')])
        print(f"  ✓ 训练集: {train_imgs} 图像, {train_masks} 掩码")
    else:
        print(f"  ✗ 训练集目录不存在")
        return False
    
    # 检查测试集
    test_dir = os.path.join(data_path, "test", "test")
    if not os.path.exists(test_dir):
        test_dir = os.path.join(data_path, "test")
    
    if os.path.exists(test_dir):
        test_imgs = len([f for f in os.listdir(test_dir) if f.endswith('_img.png')])
        test_masks = len([f for f in os.listdir(test_dir) if f.endswith('_masks.png')])
        print(f"  ✓ 测试集: {test_imgs} 图像, {test_masks} 掩码")
    else:
        print(f"  ✗ 测试集目录不存在")
        return False
    
    # 检查一张图片的格式
    import cv2
    sample_img_path = os.path.join(train_dir, "000_img.png")
    if os.path.exists(sample_img_path):
        img = cv2.imread(sample_img_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            print(f"  ✓ 图像格式: {img.dtype}, 形状: {img.shape}")
            print(f"    - 值范围: [{img.min()}, {img.max()}]")
            if img.dtype == 'uint16':
                print(f"    - 确认为 16-bit 图像 ✓")
        else:
            print(f"  ✗ 无法读取样本图像")
    
    return True


def check_dinov3_import():
    """检查 DINOv3 模块导入"""
    print("\n" + "=" * 60)
    print("5. 检查 DINOv3 模块")
    print("=" * 60)
    
    try:
        from dinov3.hub.backbones import dinov3_vitl16, dinov3_vit7b16
        print("  ✓ dinov3.hub.backbones")
    except ImportError as e:
        print(f"  ✗ dinov3.hub.backbones: {e}")
        return False
    
    try:
        from dinov3.eval.cellpose.linear_probe import (
            apply_preprocessing,
            CellposeLinearProbeDataset,
            DINOv3LinearSegmenter,
            calculate_miou,
        )
        print("  ✓ dinov3.eval.cellpose.linear_probe")
    except ImportError as e:
        print(f"  ✗ dinov3.eval.cellpose.linear_probe: {e}")
        return False
    
    return True


def test_preprocessing():
    """测试预处理函数"""
    print("\n" + "=" * 60)
    print("6. 测试预处理函数")
    print("=" * 60)
    
    import numpy as np
    from dinov3.eval.cellpose.linear_probe import apply_preprocessing
    
    # 创建模拟 16-bit 图像
    np.random.seed(42)
    fake_img = np.random.randint(100, 60000, size=(256, 256), dtype=np.uint16)
    # 添加一些极端噪点
    fake_img[0, 0] = 65535
    fake_img[255, 255] = 0
    
    print(f"  输入图像: dtype={fake_img.dtype}, range=[{fake_img.min()}, {fake_img.max()}]")
    
    for mode in ['minmax', 'percentile', 'hybrid']:
        result = apply_preprocessing(fake_img, mode=mode)
        print(f"  {mode:12s}: range=[{result.min():.4f}, {result.max():.4f}]")
    
    print("  ✓ 预处理函数工作正常")
    return True


def main():
    print("\n" + "=" * 60)
    print("Cellpose Linear Probing 环境检查")
    print("=" * 60 + "\n")
    
    results = {}
    
    results['dependencies'] = check_dependencies()
    results['cuda'] = check_cuda()
    results['checkpoints'] = check_checkpoints()
    results['data'] = check_data()
    
    if results['dependencies']:
        results['dinov3'] = check_dinov3_import()
        if results['dinov3']:
            results['preprocessing'] = test_preprocessing()
    
    # 汇总
    print("\n" + "=" * 60)
    print("检查结果汇总")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("所有检查通过! 可以运行实验:")
        print("  python -m dinov3.eval.cellpose.run_experiment")
    else:
        print("部分检查失败，请根据上述提示修复问题。")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())

