#!/usr/bin/env python3
"""
Cellpose 16-bit Linear Probing 实验 - 直接运行脚本

这个脚本可以直接运行，无需命令行参数。
所有配置都在脚本内部设置。

使用方法:
    cd /mnt/huawei_deepcad/dinov3
    python -m dinov3.eval.cellpose.run_experiment
"""

import os
import sys
from pathlib import Path

# 获取项目根目录
DINOV3_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(DINOV3_ROOT))

# ============================================================================
# 配置参数 - 根据需要修改
# ============================================================================

CONFIG = {
    # 数据路径
    'data_path': '/mnt/deepcad_nfs/0-large-model-dataset/11-Cellpose',
    
    # 输出目录
    'output_dir': str(DINOV3_ROOT / 'dinov3' / 'outputs' / 'cellpose_linear_probe'),
    
    # Checkpoint 配置
    'checkpoint_dir': '/mnt/deepcad_nfs/xuzijing/checkpoints',
    
    # 模型选择: 'l' (ViT-L, 推荐) 或 '7b' (ViT-7B, 需要更多显存)
    'model_size': 'l',
    
    # 训练参数
    'epochs': 10,
    'batch_size': 8,  # ViT-L 建议 8, ViT-7B 建议 2
    'learning_rate': 1e-3,
    'num_workers': 4,
    
    # 是否使用多尺度特征 (使用多层特征，效果可能更好但更慢)
    'use_multi_scale': False,
    
    # 要测试的预处理模式
    'modes': ['minmax', 'percentile', 'hybrid'],
}

# 根据模型大小自动选择 checkpoint
CHECKPOINT_MAP = {
    'l': 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
    '7b': 'dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth',
}


def main():
    import torch
    from datetime import datetime
    import json
    import logging
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('cellpose_experiment')
    
    # 检查 CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA 不可用! 此实验需要 GPU。")
        sys.exit(1)
    
    device = torch.device('cuda')
    logger.info(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"可用显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 构建 checkpoint 路径
    checkpoint_name = CHECKPOINT_MAP.get(CONFIG['model_size'])
    if not checkpoint_name:
        logger.error(f"不支持的模型大小: {CONFIG['model_size']}")
        sys.exit(1)
    
    checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], checkpoint_name)
    
    # 检查 checkpoint 是否存在
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint 不存在: {checkpoint_path}")
        sys.exit(1)
    
    # 检查数据目录
    if not os.path.exists(CONFIG['data_path']):
        logger.error(f"数据目录不存在: {CONFIG['data_path']}")
        sys.exit(1)
    
    # 调整 batch size (7B 模型需要更小的 batch size)
    batch_size = CONFIG['batch_size']
    if CONFIG['model_size'] == '7b' and batch_size > 2:
        batch_size = 2
        logger.warning(f"ViT-7B 模型需要较小的 batch size, 已调整为 {batch_size}")
    
    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(CONFIG['output_dir'], f"{CONFIG['model_size']}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    config_to_save = {**CONFIG, 'checkpoint_path': checkpoint_path, 'batch_size': batch_size}
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config_to_save, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("Cellpose 16-bit Linear Probing 实验")
    logger.info("=" * 60)
    logger.info(f"模型: DINOv3 ViT-{CONFIG['model_size'].upper()}")
    logger.info(f"数据: {CONFIG['data_path']}")
    logger.info(f"输出: {output_dir}")
    logger.info(f"Epochs: {CONFIG['epochs']}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"预处理模式: {CONFIG['modes']}")
    logger.info("=" * 60)
    
    # 导入实验模块
    from dinov3.eval.cellpose.linear_probe import (
        load_dinov3_backbone,
        get_dataset_paths,
        run_experiment,
    )
    
    # 加载数据路径
    train_img_paths, train_mask_paths = get_dataset_paths(CONFIG['data_path'], 'train')
    test_img_paths, test_mask_paths = get_dataset_paths(CONFIG['data_path'], 'test')
    
    # 加载 backbone
    backbone = load_dinov3_backbone(
        checkpoint_path,
        model_size=CONFIG['model_size'],
        device=device
    )
    
    # 运行所有实验
    all_results = {}
    
    for mode in CONFIG['modes']:
        results = run_experiment(
            mode=mode,
            backbone=backbone,
            train_img_paths=train_img_paths,
            train_mask_paths=train_mask_paths,
            test_img_paths=test_img_paths,
            test_mask_paths=test_mask_paths,
            output_dir=output_dir,
            epochs=CONFIG['epochs'],
            batch_size=batch_size,
            learning_rate=CONFIG['learning_rate'],
            device=device,
            num_workers=CONFIG['num_workers'],
            use_multi_scale=CONFIG['use_multi_scale'],
        )
        all_results[mode] = results
    
    # 汇总结果
    logger.info("\n" + "=" * 60)
    logger.info("实验结果汇总")
    logger.info("=" * 60)
    
    summary = []
    for mode, results in all_results.items():
        summary.append({
            'mode': mode,
            'mIoU': results['best_mIoU'],
            'Dice': results['Dice'],
            'IoU_cell': results['IoU_cell'],
            'IoU_background': results['IoU_background'],
        })
        logger.info(
            f"[{mode.upper():12s}] mIoU: {results['best_mIoU']:.4f}, "
            f"Dice: {results['Dice']:.4f}, "
            f"Cell IoU: {results['IoU_cell']:.4f}"
        )
    
    # 保存汇总结果
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 找出最佳模式
    best_mode = max(summary, key=lambda x: x['mIoU'])
    logger.info(f"\n最佳预处理模式: {best_mode['mode'].upper()}")
    logger.info(f"最佳 mIoU: {best_mode['mIoU']:.4f}")
    
    # 打印预期结论
    logger.info("\n" + "=" * 60)
    logger.info("预处理方法分析")
    logger.info("=" * 60)
    logger.info("""
预期结果分析:
- MinMax (A组): 预期效果最差
  原因: 16-bit 图像中的极亮噪点会将所有其他像素压缩到接近 0 的范围
  
- Percentile (B组): 预期效果中等
  原因: 截断两端可以排除噪点，但 99.7% 截断可能丢失细胞核的高亮纹理
  
- Hybrid (C组): 预期效果最好
  原因: 只剔除最极端的 0.1% 离群值，最大程度保留 16-bit 的动态范围
""")
    
    logger.info(f"\n实验完成! 所有结果保存在: {output_dir}")
    
    return all_results


if __name__ == '__main__':
    main()

