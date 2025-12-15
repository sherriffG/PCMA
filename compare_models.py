#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新旧模型对比分析脚本

功能：
1. 在同一个QPSK测试集上测试新旧模型
2. 对比性能指标（BER1, BER2, EVM等）
3. 生成对比图表和报告
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def save_fig(path, dpi=150):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()

def load_metrics(csv_path):
    """加载metrics CSV文件"""
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    return df

def compare_models(old_csv, new_csv, out_dir):
    """
    对比新旧模型的性能
    
    Args:
        old_csv: 旧模型测试结果的CSV路径
        new_csv: 新模型测试结果的CSV路径
        out_dir: 输出目录
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # 加载数据
    print(f"[对比] 加载旧模型结果: {old_csv}")
    df_old = load_metrics(old_csv)
    if df_old is None:
        print(f"[错误] 无法加载旧模型结果: {old_csv}")
        return
    
    print(f"[对比] 加载新模型结果: {new_csv}")
    df_new = load_metrics(new_csv)
    if df_new is None:
        print(f"[错误] 无法加载新模型结果: {new_csv}")
        return
    
    # 确保两路都是QPSK（过滤数据）
    if 'mod1' in df_old.columns and 'mod2' in df_old.columns:
        df_old = df_old[(df_old['mod1'] == 'QPSK') & (df_old['mod2'] == 'QPSK')].copy()
    if 'mod1' in df_new.columns and 'mod2' in df_new.columns:
        df_new = df_new[(df_new['mod1'] == 'QPSK') & (df_new['mod2'] == 'QPSK')].copy()
    
    print(f"[对比] 旧模型QPSK样本数: {len(df_old)}")
    print(f"[对比] 新模型QPSK样本数: {len(df_new)}")
    
    # 1. 整体统计对比
    print("\n" + "="*70)
    print("整体性能对比")
    print("="*70)
    
    metrics_to_compare = ['BER1', 'BER2', 'BER', 'evm1', 'evm2', 'loss1', 'loss2']
    comparison_stats = []
    
    for metric in metrics_to_compare:
        if metric not in df_old.columns or metric not in df_new.columns:
            continue
        
        old_mean = df_old[metric].mean()
        old_std = df_old[metric].std()
        new_mean = df_new[metric].mean()
        new_std = df_new[metric].std()
        
        if old_mean > 0:
            change_pct = ((new_mean - old_mean) / old_mean) * 100
        else:
            change_pct = float('inf') if new_mean > 0 else 0
        
        comparison_stats.append({
            'Metric': metric,
            'Old_Mean': old_mean,
            'Old_Std': old_std,
            'New_Mean': new_mean,
            'New_Std': new_std,
            'Change_%': change_pct,
            'Change_Abs': new_mean - old_mean
        })
        
        print(f"\n{metric}:")
        print(f"  旧模型: {old_mean:.6f} ± {old_std:.6f}")
        print(f"  新模型: {new_mean:.6f} ± {new_std:.6f}")
        print(f"  变化: {change_pct:+.2f}% ({new_mean - old_mean:+.6f})")
    
    # 保存统计对比表
    stats_df = pd.DataFrame(comparison_stats)
    stats_csv = os.path.join(out_dir, "comparison_stats.csv")
    stats_df.to_csv(stats_csv, index=False)
    print(f"\n[对比] 统计对比表已保存至: {stats_csv}")
    
    # 2. BER vs SNR 对比（如果有SNR列）
    if 'snr' in df_old.columns and 'snr' in df_new.columns:
        print("\n" + "="*70)
        print("按SNR分组对比")
        print("="*70)
        
        # 聚合按SNR
        old_snr = (df_old.groupby('snr', as_index=False)
                   .agg(mean_BER1=('BER1', 'mean'),
                        mean_BER2=('BER2', 'mean'),
                        mean_BER=('BER', 'mean'),
                        sem_BER1=('BER1', lambda x: np.std(x, ddof=1) / np.sqrt(max(1, len(x)))),
                        sem_BER2=('BER2', lambda x: np.std(x, ddof=1) / np.sqrt(max(1, len(x)))),
                        sem_BER=('BER', lambda x: np.std(x, ddof=1) / np.sqrt(max(1, len(x)))))
                   .sort_values('snr'))
        
        new_snr = (df_new.groupby('snr', as_index=False)
                   .agg(mean_BER1=('BER1', 'mean'),
                        mean_BER2=('BER2', 'mean'),
                        mean_BER=('BER', 'mean'),
                        sem_BER1=('BER1', lambda x: np.std(x, ddof=1) / np.sqrt(max(1, len(x)))),
                        sem_BER2=('BER2', lambda x: np.std(x, ddof=1) / np.sqrt(max(1, len(x)))),
                        sem_BER=('BER', lambda x: np.std(x, ddof=1) / np.sqrt(max(1, len(x)))))
                   .sort_values('snr'))
        
        # 绘制对比图
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # BER1 vs SNR
        ax = axes[0]
        snr_vals = sorted(set(old_snr['snr'].tolist() + new_snr['snr'].tolist()))
        for snr in snr_vals:
            old_row = old_snr[old_snr['snr'] == snr]
            new_row = new_snr[new_snr['snr'] == snr]
            if len(old_row) > 0 and len(new_row) > 0:
                old_val = old_row['mean_BER1'].values[0]
                new_val = new_row['mean_BER1'].values[0]
                old_sem = old_row['sem_BER1'].values[0]
                new_sem = new_row['sem_BER1'].values[0]
                
                ax.errorbar(snr, old_val, yerr=old_sem, marker='o', label='Old Model' if snr == snr_vals[0] else '', 
                           color='blue', capsize=3)
                ax.errorbar(snr, new_val, yerr=new_sem, marker='s', label='New Model' if snr == snr_vals[0] else '', 
                           color='red', capsize=3)
        
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Mean BER1 ± SEM')
        ax.set_yscale('log')
        ax.set_title('BER1 vs SNR Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # BER2 vs SNR
        ax = axes[1]
        for snr in snr_vals:
            old_row = old_snr[old_snr['snr'] == snr]
            new_row = new_snr[new_snr['snr'] == snr]
            if len(old_row) > 0 and len(new_row) > 0:
                old_val = old_row['mean_BER2'].values[0]
                new_val = new_row['mean_BER2'].values[0]
                old_sem = old_row['sem_BER2'].values[0]
                new_sem = new_row['sem_BER2'].values[0]
                
                ax.errorbar(snr, old_val, yerr=old_sem, marker='o', label='Old Model' if snr == snr_vals[0] else '', 
                           color='blue', capsize=3)
                ax.errorbar(snr, new_val, yerr=new_sem, marker='s', label='New Model' if snr == snr_vals[0] else '', 
                           color='red', capsize=3)
        
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Mean BER2 ± SEM')
        ax.set_yscale('log')
        ax.set_title('BER2 vs SNR Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Average BER vs SNR
        ax = axes[2]
        for snr in snr_vals:
            old_row = old_snr[old_snr['snr'] == snr]
            new_row = new_snr[new_snr['snr'] == snr]
            if len(old_row) > 0 and len(new_row) > 0:
                old_val = old_row['mean_BER'].values[0]
                new_val = new_row['mean_BER'].values[0]
                old_sem = old_row['sem_BER'].values[0]
                new_sem = new_row['sem_BER'].values[0]
                
                ax.errorbar(snr, old_val, yerr=old_sem, marker='o', label='Old Model' if snr == snr_vals[0] else '', 
                           color='blue', capsize=3)
                ax.errorbar(snr, new_val, yerr=new_sem, marker='s', label='New Model' if snr == snr_vals[0] else '', 
                           color='red', capsize=3)
        
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Mean BER ± SEM')
        ax.set_yscale('log')
        ax.set_title('Average BER vs SNR Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        save_fig(os.path.join(out_dir, "ber_vs_snr_comparison.png"))
        print(f"[对比] BER vs SNR 对比图已保存")
        
        # 保存按SNR的对比数据
        snr_comparison = pd.merge(old_snr, new_snr, on='snr', suffixes=('_old', '_new'))
        snr_comparison_csv = os.path.join(out_dir, "snr_comparison.csv")
        snr_comparison.to_csv(snr_comparison_csv, index=False)
        print(f"[对比] SNR对比数据已保存至: {snr_comparison_csv}")
    
    # 3. BER vs AMP 对比（如果有AMP列）
    if 'amp' in df_old.columns and 'amp' in df_new.columns:
        print("\n" + "="*70)
        print("按AMP分组对比")
        print("="*70)
        
        old_amp = (df_old.groupby('amp', as_index=False)
                   .agg(mean_BER1=('BER1', 'mean'),
                        mean_BER2=('BER2', 'mean'),
                        mean_BER=('BER', 'mean'))
                   .sort_values('amp'))
        
        new_amp = (df_new.groupby('amp', as_index=False)
                   .agg(mean_BER1=('BER1', 'mean'),
                        mean_BER2=('BER2', 'mean'),
                        mean_BER=('BER', 'mean'))
                   .sort_values('amp'))
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, metric in enumerate(['BER1', 'BER2', 'BER']):
            ax = axes[idx]
            old_col = f'mean_{metric}'
            new_col = f'mean_{metric}'
            
            ax.semilogy(old_amp['amp'], old_amp[old_col], marker='o', label='Old Model', linewidth=2)
            ax.semilogy(new_amp['amp'], new_amp[new_col], marker='s', label='New Model', linewidth=2)
            
            ax.set_xlabel('Amplitude Ratio')
            ax.set_ylabel(f'Mean {metric}')
            ax.set_yscale('log')
            ax.set_title(f'{metric} vs AMP Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        save_fig(os.path.join(out_dir, "ber_vs_amp_comparison.png"))
        print(f"[对比] BER vs AMP 对比图已保存")
    
    # 4. 性能分布对比（直方图）
    print("\n" + "="*70)
    print("性能分布对比")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # BER1 分布
    ax = axes[0, 0]
    ax.hist(df_old['BER1'].clip(1e-6, 1.0), bins=50, alpha=0.6, label='Old Model', density=True)
    ax.hist(df_new['BER1'].clip(1e-6, 1.0), bins=50, alpha=0.6, label='New Model', density=True)
    ax.set_xlabel('BER1')
    ax.set_ylabel('Density')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title('BER1 Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # BER2 分布
    ax = axes[0, 1]
    ax.hist(df_old['BER2'].clip(1e-6, 1.0), bins=50, alpha=0.6, label='Old Model', density=True)
    ax.hist(df_new['BER2'].clip(1e-6, 1.0), bins=50, alpha=0.6, label='New Model', density=True)
    ax.set_xlabel('BER2')
    ax.set_ylabel('Density')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title('BER2 Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # EVM1 分布
    if 'evm1' in df_old.columns and 'evm1' in df_new.columns:
        ax = axes[1, 0]
        ax.hist(df_old['evm1'].dropna(), bins=50, alpha=0.6, label='Old Model', density=True)
        ax.hist(df_new['evm1'].dropna(), bins=50, alpha=0.6, label='New Model', density=True)
        ax.set_xlabel('EVM1')
        ax.set_ylabel('Density')
        ax.set_title('EVM1 Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # EVM2 分布
    if 'evm2' in df_old.columns and 'evm2' in df_new.columns:
        ax = axes[1, 1]
        ax.hist(df_old['evm2'].dropna(), bins=50, alpha=0.6, label='Old Model', density=True)
        ax.hist(df_new['evm2'].dropna(), bins=50, alpha=0.6, label='New Model', density=True)
        ax.set_xlabel('EVM2')
        ax.set_ylabel('Density')
        ax.set_title('EVM2 Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    save_fig(os.path.join(out_dir, "distribution_comparison.png"))
    print(f"[对比] 性能分布对比图已保存")
    
    # 5. 生成对比报告
    report_path = os.path.join(out_dir, "comparison_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("新旧模型性能对比报告\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"旧模型测试结果: {old_csv}\n")
        f.write(f"新模型测试结果: {new_csv}\n")
        f.write(f"测试样本数 (旧): {len(df_old)}\n")
        f.write(f"测试样本数 (新): {len(df_new)}\n\n")
        
        f.write("整体性能对比:\n")
        f.write("-"*70 + "\n")
        for stat in comparison_stats:
            f.write(f"{stat['Metric']}:\n")
            f.write(f"  旧模型: {stat['Old_Mean']:.6f} ± {stat['Old_Std']:.6f}\n")
            f.write(f"  新模型: {stat['New_Mean']:.6f} ± {stat['New_Std']:.6f}\n")
            f.write(f"  变化: {stat['Change_%']:+.2f}% ({stat['Change_Abs']:+.6f})\n\n")
        
        # 计算性能下降的SNR等效值（如果可能）
        if 'snr' in df_old.columns and 'snr' in df_new.columns:
            old_mean_ber = df_old['BER'].mean()
            new_mean_ber = df_new['BER'].mean()
            
            # 简单估算：如果BER从1e-3变到1e-2，相当于SNR下降约3-4dB
            if old_mean_ber > 0 and new_mean_ber > 0:
                ber_ratio = new_mean_ber / old_mean_ber
                estimated_snr_degradation = 10 * np.log10(ber_ratio)  # 粗略估算
                f.write(f"\n估算SNR等效下降: {estimated_snr_degradation:.2f} dB\n")
                f.write(f"(基于平均BER变化: {old_mean_ber:.6f} -> {new_mean_ber:.6f})\n")
    
    print(f"\n[对比] 对比报告已保存至: {report_path}")
    print("\n" + "="*70)
    print("对比分析完成！")
    print("="*70)

def main():
    parser = argparse.ArgumentParser(description="新旧模型性能对比分析")
    parser.add_argument('--old_csv', type=str, required=True,
                       help='旧模型测试结果的CSV文件路径')
    parser.add_argument('--new_csv', type=str, required=True,
                       help='新模型测试结果的CSV文件路径')
    parser.add_argument('--out_dir', type=str, default='./src/results/model_comparison',
                       help='输出目录（默认: ./src/results/model_comparison）')
    args = parser.parse_args()
    
    compare_models(args.old_csv, args.new_csv, args.out_dir)

if __name__ == '__main__':
    main()

