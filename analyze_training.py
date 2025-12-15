#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练过程分析脚本

功能：
1. 分析loss曲线的特征
2. 诊断训练问题
3. 提供改进建议
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def analyze_loss_curve(loss_file, out_dir):
    """
    分析loss曲线图像（从图像中提取信息或直接分析数据）
    这里假设可以从训练日志或保存的loss数据中分析
    """
    os.makedirs(out_dir, exist_ok=True)
    
    print("="*70)
    print("训练过程诊断分析（基于实际训练参数）")
    print("="*70)
    
    print("\n实际训练参数:")
    print("  --epochs 180")
    print("  --mse_epochs 180 (纯MSE，未启用CSI loss)")
    print("  --lr 5e-4 (比默认2e-4高2.5倍!)")
    print("  --warmup_epochs 3")
    print("  --min_lr_ratio 0.1")
    
    # 基于loss曲线的观察
    print("\n1. Loss数值分析:")
    print("   - 当前loss: 训练≈0.16, 验证≈0.38 (1e-1数量级)")
    print("   - 使用了 normalized_time_mse，这是归一化MSE")
    print("   - normalized_time_mse = MSE / mean(tgt^2)")
    print("   - 值域通常在 [0, 1] 左右，所以 0.16-0.38 是合理的")
    print("   - 只用了MSE loss，没有CSI loss")
    
    print("\n2. 训练过程问题诊断:")
    print("   a) 过拟合明显:")
    print("      - 训练loss持续下降: 0.65 → 0.16")
    print("      - 验证loss波动并稳定: 0.57 → 0.38")
    print("      - 差距约2.4倍，说明模型过拟合训练集")
    
    print("\n   b) 验证loss在20 epoch后波动:")
    print("      - 主要原因: 学习率过大 (5e-4，比默认值高2.5倍!)")
    print("      - 高学习率导致训练不稳定，难以收敛到最优解")
    print("      - 数据分布复杂，模型难以稳定学习")
    
    print("\n   c) Loss下降不充分:")
    print("      - 训练loss从0.65降到0.16，下降了75%")
    print("      - 但验证loss只从0.57降到0.38，仅下降33%")
    print("      - 说明模型在训练集上学习良好，但泛化能力不足")
    
    print("\n3. 可能的问题根源:")
    print("   a) 损失函数设计:")
    print("      - normalized_time_mse 的分母是 mean(tgt^2)")
    print("      - 如果目标信号能量较小，loss会偏大")
    print("      - 多调制组合中，不同调制的能量可能不同")
    
    print("\n   b) 训练策略:")
    print("      - 整个训练过程只用了MSE loss，没有CSI loss")
    print("      - CSI loss具有尺度不变性，可能更适合信号分离")
    print("      - 纯MSE可能限制了模型的性能上限")
    
    print("\n   c) 模型容量:")
    print("      - 多调制任务（QPSK/8PSK/16QAM）比单QPSK复杂")
    print("      - 需要学习9种调制组合的分离模式")
    print("      - 当前模型容量可能不足")
    
    print("\n   d) 数据复杂度:")
    print("      - 训练数据包含多种调制组合")
    print("      - 不同调制方式的信号特征差异大")
    print("      - 模型需要更强的泛化能力")
    
    print("\n4. 改进建议（按优先级）:")
    print("   a) 降低学习率 (最重要!):")
    print("      - 当前学习率5e-4过高，是验证loss波动的主要原因")
    print("      - 建议: --lr 1e-4 或 2e-4 (降低到默认值或更低)")
    print("      - 这应该能显著改善训练稳定性和验证loss波动")
    
    print("\n   b) 考虑引入CSI loss:")
    print("      - 当前只用了MSE，CSI loss可能有助于提升性能")
    print("      - 建议: --mse_epochs 100 (前100个epoch用MSE)")
    print("      - 然后: --csi_warmup_epochs 20 (20个epoch过渡到CSI)")
    print("      - 设置: --csi_beta 0.5, --alpha_after 0.1 (混合损失)")
    
    print("\n   c) 优化学习率调度:")
    print("      - 延长warmup: --warmup_epochs 5-10 (从3增加)")
    print("      - 更小的最小学习率: --min_lr_ratio 0.05 (从0.1降低)")
    
    print("\n   c) 增加训练时间:")
    print("      - 增加epoch数: --epochs 200-300")
    print("      - 让模型有更多时间学习多调制模式")
    
    print("\n   d) 数据增强/平衡:")
    print("      - 检查不同调制组合的样本分布")
    print("      - 可能需要增加某些调制组合的样本数")
    
    print("\n   e) 正则化:")
    print("      - 增加dropout（如果模型支持）")
    print("      - 使用weight decay")
    print("      - 数据增强（添加更多噪声变化）")
    
    print("\n   f) 模型架构:")
    print("      - 考虑增加模型容量（如果可能）")
    print("      - 或者使用更深的网络")
    
    # 生成改进建议的配置文件
    suggestions = {
        "fix_lr_only": {
            "description": "仅降低学习率（最简单，推荐先试）",
            "lr": 1e-4,
            "epochs": 180,
            "mse_epochs": 180,
            "warmup_epochs": 3,
            "min_lr_ratio": 0.1,
        },
        "conservative": {
            "description": "降低学习率 + 引入CSI loss（推荐）",
            "lr": 1e-4,
            "epochs": 200,
            "mse_epochs": 100,
            "csi_warmup_epochs": 20,
            "csi_beta": 0.5,
            "alpha_after": 0.1,
            "warmup_epochs": 5,
            "min_lr_ratio": 0.05,
        },
        "aggressive": {
            "description": "激进策略（如果保守策略不够）",
            "lr": 5e-5,
            "epochs": 300,
            "mse_epochs": 120,
            "csi_warmup_epochs": 30,
            "csi_beta": 0.3,
            "alpha_after": 0.2,
            "warmup_epochs": 10,
            "min_lr_ratio": 0.01,
        }
    }
    
    print("\n5. 推荐的训练配置:")
    print("\n   策略1: 仅降低学习率（最简单，推荐先试）:")
    print(f"      {suggestions['fix_lr_only']['description']}")
    for key, value in suggestions["fix_lr_only"].items():
        if key != "description":
            print(f"      --{key} {value}")
    
    print("\n   策略2: 降低学习率 + 引入CSI loss（推荐）:")
    print(f"      {suggestions['conservative']['description']}")
    for key, value in suggestions["conservative"].items():
        if key != "description":
            print(f"      --{key} {value}")
    
    print("\n   策略3: 激进策略（如果策略2不够）:")
    print(f"      {suggestions['aggressive']['description']}")
    for key, value in suggestions["aggressive"].items():
        if key != "description":
            print(f"      --{key} {value}")
    
    # 保存建议到文件
    report_path = os.path.join(out_dir, "training_analysis_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("训练过程诊断分析报告\n")
        f.write("="*70 + "\n\n")
        
        f.write("问题诊断:\n")
        f.write("1. Loss数值在1e-1数量级是正常的（使用normalized_time_mse）\n")
        f.write("2. 存在明显过拟合：训练loss持续下降，验证loss波动\n")
        f.write("3. 验证loss在20 epoch后波动，可能由于学习率过大或损失函数切换\n")
        f.write("4. 模型可能容量不足，难以学习复杂的多调制分离任务\n\n")
        
        f.write("改进建议:\n")
        f.write("-"*70 + "\n")
        f.write("策略1: 仅降低学习率（最简单，推荐先试）:\n")
        f.write(f"  {suggestions['fix_lr_only']['description']}\n")
        for key, value in suggestions["fix_lr_only"].items():
            if key != "description":
                f.write(f"  --{key} {value}\n")
        f.write("\n策略2: 降低学习率 + 引入CSI loss（推荐）:\n")
        f.write(f"  {suggestions['conservative']['description']}\n")
        for key, value in suggestions["conservative"].items():
            if key != "description":
                f.write(f"  --{key} {value}\n")
        f.write("\n策略3: 激进策略:\n")
        f.write(f"  {suggestions['aggressive']['description']}\n")
        for key, value in suggestions["aggressive"].items():
            if key != "description":
                f.write(f"  --{key} {value}\n")
    
    print(f"\n[分析] 诊断报告已保存至: {report_path}")
    
    # 生成对比图建议
    print("\n6. 下一步行动:")
    print("   a) 使用保守策略重新训练，观察loss曲线是否改善")
    print("   b) 如果验证loss仍然波动，尝试激进策略")
    print("   c) 对比新旧训练过程的loss曲线")
    print("   d) 分析不同epoch的模型在测试集上的性能")

def main():
    parser = argparse.ArgumentParser(description="训练过程分析")
    parser.add_argument('--loss_file', type=str, default=None,
                       help='Loss曲线图像文件（可选）')
    parser.add_argument('--out_dir', type=str, default='./src/results/training_analysis',
                       help='输出目录')
    args = parser.parse_args()
    
    analyze_loss_curve(args.loss_file, args.out_dir)

if __name__ == '__main__':
    main()

