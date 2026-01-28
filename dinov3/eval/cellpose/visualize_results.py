#!/usr/bin/env python3
"""
可视化 Cellpose Linear Probing 实验结果

使用方法:
    cd /mnt/huawei_deepcad/dinov3
    python -m dinov3.eval.cellpose.visualize_results <results_dir>

示例:
    python -m dinov3.eval.cellpose.visualize_results dinov3/outputs/cellpose_linear_probe/l_20251225_120000
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir: str):
    """加载实验结果"""
    summary_path = os.path.join(results_dir, 'summary.json')
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"找不到结果文件: {summary_path}")
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # 加载每个模式的历史
    histories = {}
    for mode_result in summary:
        mode = mode_result['mode']
        history_path = os.path.join(results_dir, mode, 'history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                histories[mode] = json.load(f)
    
    return summary, histories


def plot_training_curves(histories: dict, save_path: str):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {
        'minmax': '#e74c3c',      # 红色
        'percentile': '#f39c12',   # 橙色
        'hybrid': '#27ae60',       # 绿色
    }
    
    mode_names = {
        'minmax': 'A组: Min-Max',
        'percentile': 'B组: Percentile',
        'hybrid': 'C组: Hybrid',
    }
    
    for mode, history in histories.items():
        color = colors.get(mode, '#3498db')
        label = mode_names.get(mode, mode)
        epochs = range(1, len(history['train']) + 1)
        
        # Train Loss
        train_loss = [h['loss'] for h in history['train']]
        axes[0, 0].plot(epochs, train_loss, color=color, label=label, linewidth=2)
        
        # Test mIoU
        test_miou = [h['mIoU'] for h in history['test']]
        axes[0, 1].plot(epochs, test_miou, color=color, label=label, linewidth=2)
        
        # Test Dice
        test_dice = [h['Dice'] for h in history['test']]
        axes[1, 0].plot(epochs, test_dice, color=color, label=label, linewidth=2)
        
        # Cell IoU
        cell_iou = [h['IoU_cell'] for h in history['test']]
        axes[1, 1].plot(epochs, cell_iou, color=color, label=label, linewidth=2)
    
    axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Test mIoU', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('mIoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Test Dice', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Cell IoU', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('IoU')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存: {save_path}")


def plot_comparison_bar(summary: list, save_path: str):
    """绘制结果对比柱状图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    modes = [s['mode'] for s in summary]
    miou = [s['mIoU'] for s in summary]
    dice = [s['Dice'] for s in summary]
    cell_iou = [s['IoU_cell'] for s in summary]
    
    x = np.arange(len(modes))
    width = 0.25
    
    colors = {
        'minmax': '#e74c3c',
        'percentile': '#f39c12',
        'hybrid': '#27ae60',
    }
    bar_colors = [colors.get(m, '#3498db') for m in modes]
    
    bars1 = ax.bar(x - width, miou, width, label='mIoU', color=bar_colors, alpha=1.0, edgecolor='black')
    bars2 = ax.bar(x, dice, width, label='Dice', color=bar_colors, alpha=0.7, edgecolor='black')
    bars3 = ax.bar(x + width, cell_iou, width, label='Cell IoU', color=bar_colors, alpha=0.4, edgecolor='black')
    
    ax.set_xlabel('Preprocessing Method', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Cellpose 16-bit Segmentation - Preprocessing Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    
    mode_labels = {
        'minmax': 'Group A\n(Min-Max)',
        'percentile': 'Group B\n(Percentile)',
        'hybrid': 'Group C\n(Hybrid)',
    }
    ax.set_xticklabels([mode_labels.get(m, m) for m in modes])
    
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis='y', alpha=0.3)
    
    # 在柱子上添加数值
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"对比图已保存: {save_path}")


def generate_report(results_dir: str, summary: list, histories: dict):
    """生成文本报告"""
    report_path = os.path.join(results_dir, 'report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Cellpose 16-bit Linear Probing Experiment Report\n")
        f.write("=" * 70 + "\n\n")
        
        # 结果表格
        f.write("Results Summary:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Method':<15} {'mIoU':<10} {'Dice':<10} {'Cell IoU':<10} {'BG IoU':<10}\n")
        f.write("-" * 60 + "\n")
        
        for s in summary:
            f.write(f"{s['mode']:<15} {s['mIoU']:.4f}     {s['Dice']:.4f}     "
                   f"{s['IoU_cell']:.4f}     {s['IoU_background']:.4f}\n")
        f.write("-" * 60 + "\n\n")
        
        # 最佳结果
        best = max(summary, key=lambda x: x['mIoU'])
        f.write(f"Best Preprocessing Method: {best['mode'].upper()}\n")
        f.write(f"Best mIoU: {best['mIoU']:.4f}\n\n")
        
        # 分析
        f.write("Preprocessing Method Analysis:\n")
        f.write("-" * 60 + "\n")
        f.write("""
Group A (Min-Max): Global normalization
  - Maps entire dynamic range (0-65535) linearly to 0-1
  - Problem: Extreme bright pixels compress most cell signals to near 0
  - Expected: Worst performance

Group B (Percentile): Percentile clipping
  - Clips 0.3% and 99.7% percentiles before normalization
  - Advantage: Effectively excludes extreme noise
  - Problem: May clip too much high-intensity regions
  - Expected: Medium performance

Group C (Hybrid): Hybrid method
  - Only clips 99.9% high end before normalization
  - Advantage: Maximally preserves 16-bit dynamic range
  - Only removes extreme outlier noise
  - Expected: Best performance
""")
        f.write("-" * 60 + "\n\n")
        
        # 结论
        if best['mode'] == 'hybrid':
            conclusion = "Results match expectations! Hybrid method performs best for 16-bit cell image segmentation."
        elif best['mode'] == 'percentile':
            conclusion = "Percentile method performs best. Dataset may have more noise requiring aggressive clipping."
        else:
            conclusion = "Min-Max method unexpectedly performs well. Check dataset characteristics."
        
        f.write(f"Conclusion: {conclusion}\n")
    
    print(f"报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Cellpose experiment results')
    parser.add_argument('results_dir', type=str, help='Experiment results directory')
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"错误: 目录不存在: {args.results_dir}")
        sys.exit(1)
    
    print(f"加载结果: {args.results_dir}")
    summary, histories = load_results(args.results_dir)
    
    # 生成可视化
    if histories:
        plot_training_curves(
            histories,
            os.path.join(args.results_dir, 'training_curves.png')
        )
    
    plot_comparison_bar(
        summary,
        os.path.join(args.results_dir, 'comparison.png')
    )
    
    # 生成报告
    generate_report(args.results_dir, summary, histories)
    
    print("\n可视化完成!")


if __name__ == '__main__':
    main()

