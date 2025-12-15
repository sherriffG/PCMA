# viz_metrics.py
# -*- coding: utf-8 -*-
import os, argparse, numpy as np, pandas as pd
import matplotlib.pyplot as plt

def save_fig(path, dpi=150):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()

def wrap_2pi(x):
    return np.mod(x, 2*np.pi)

# ========== mode=orig：保留原来的 test_all 可视化逻辑 ==========
def viz_orig(df: pd.DataFrame, out_dir: str):
    def save_fig2(path, dpi=150):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.tight_layout(); plt.savefig(path, dpi=dpi); plt.close()

    # 1) BER 直方 + CDF（survival）
    if "BER" in df.columns and len(df):
        small = 1e-12
        ber_vals = df["BER"].clip(lower=small, upper=1.0)

        plt.figure(figsize=(6,4))
        plt.hist(ber_vals, bins=40, density=True, alpha=0.8)
        plt.yscale('log')
        plt.xlabel("BER"); plt.ylabel("Density (log)"); plt.title("BER Histogram (log y)")
        save_fig2(os.path.join(out_dir, "ber_hist_semi.png"))

        plt.figure(figsize=(6,4))
        ber_sorted = np.sort(ber_vals.values)
        p = np.linspace(0,1,len(ber_sorted))
        survival = 1.0 - p
        plt.semilogy(ber_sorted, survival)
        plt.xlabel("BER"); plt.ylabel("Survival = 1-CDF (log)"); plt.title("BER Survival (semilogy)")
        save_fig2(os.path.join(out_dir, "ber_survival_semi.png"))

    # 2) EVM CDF（分 SNR）
    def plot_evm_cdf_by_snr(col, name):
        if col not in df.columns: return
        if 'snr' not in df.columns: return
        plt.figure(figsize=(7,5))
        snr_vals = sorted([v for v in df['snr'].dropna().unique()])
        for snr in snr_vals:
            sub = df[df['snr']==snr][col].dropna().values
            if len(sub) == 0: continue
            xs = np.sort(sub); pp = np.linspace(0,1,len(xs))
            plt.plot(xs, pp, label=f"SNR={snr:g}dB")
        plt.grid(True); plt.legend()
        plt.xlabel(name); plt.ylabel("CDF"); plt.title(f"{name} CDF by SNR")
        save_fig2(os.path.join(out_dir, f"{name.lower()}_cdf_by_snr.png"))

    plot_evm_cdf_by_snr('evm1', 'EVM1')
    plot_evm_cdf_by_snr('evm2', 'EVM2')

    # 3) BER vs SNR（per amp）
    if all(c in df.columns for c in ["amp", "snr", "BER"]):
        plt.figure(figsize=(8,5))
        for a in sorted([v for v in df['amp'].dropna().unique()]):
            sub_raw = df[df['amp']==a].dropna(subset=['BER','snr'])
            if len(sub_raw)==0: continue
            sub = (sub_raw.groupby('snr', as_index=False)
                   .agg(mean_BER=('BER', 'mean'))
                   .sort_values('snr'))
            plt.semilogy(sub['snr'], sub['mean_BER'], marker='o', label=f"amp={a:.2f}")
        plt.grid(True); plt.legend(ncol=2)
        plt.xlabel("SNR (dB)"); plt.ylabel("Mean BER (log y)")
        plt.title("BER vs SNR (per amplitude, semilogy)")
        save_fig2(os.path.join(out_dir, "ber_vs_snr_per_amp_semi.png"))

    # 4) BER vs amp（per SNR）
    if all(c in df.columns for c in ["amp", "snr", "BER"]):
        plt.figure(figsize=(8,5))
        for s in sorted([v for v in df['snr'].dropna().unique()]):
            sub_raw = df[df['snr']==s].dropna(subset=['BER','amp'])
            if len(sub_raw)==0: continue
            sub = (sub_raw.groupby('amp', as_index=False)
                   .agg(mean_BER=('BER', 'mean'))
                   .sort_values('amp'))
            plt.semilogy(sub['amp'], sub['mean_BER'], marker='o', label=f"SNR={s:g}dB")
        plt.grid(True); plt.legend(ncol=2)
        plt.xlabel("Amplitude ratio (a=|s2|/|s1|)")
        plt.ylabel("Mean BER (log y)")
        plt.title("BER vs amplitude (per SNR, semilogy)")
        save_fig2(os.path.join(out_dir, "ber_vs_amp_per_snr_semi.png"))

    # 5) (f1,f2) 热力图按相位差
    if all(c in df.columns for c in ["f1", "f2", "BER", "phi1", "phi2"]):
        def bin_phi_diff(x):
            if x is None or (isinstance(x, float) and np.isnan(x)): return None
            step = 2*np.pi/8
            return float(int(np.floor(wrap_2pi(x)/step + 0.5)) % 8) * step
        df2 = df.copy()
        df2['phi_diff'] = wrap_2pi(df2['phi2'] - df2['phi1'])
        df2['phi_diff_bin'] = df2['phi_diff'].apply(bin_phi_diff)
        mean_fmap = (df2.dropna(subset=['f1','f2','BER'])
                     .groupby(['phi_diff_bin','f1','f2'], as_index=False)['BER'].mean())
        phi_bins = sorted([v for v in mean_fmap['phi_diff_bin'].dropna().unique()])

        for pbin in phi_bins:
            sub = mean_fmap[mean_fmap['phi_diff_bin']==pbin]
            if len(sub)==0: continue
            piv = sub.pivot(index='f2', columns='f1', values='BER').sort_index().sort_index(axis=1)
            small = 1e-12
            piv_log = np.log10(np.clip(piv.values, small, 1.0))
            plt.figure(figsize=(7,6))
            im = plt.imshow(piv_log, aspect='auto', origin='lower',
                            extent=[piv.columns.min(), piv.columns.max(),
                                    piv.index.min(), piv.index.max()])
            plt.colorbar(im, label='log10(Mean BER)')
            plt.xlabel('f1 (Hz)'); plt.ylabel('f2 (Hz)')
            deg = int(np.degrees(pbin))
            plt.title(f'log10(BER) Heatmap | phi_diff={pbin:.2f} rad ({deg}°)')
            save_fig2(os.path.join(out_dir, f"heatmap_f1f2_phidiff_{pbin:.2f}rad_log10.png"))

    # 6) BER vs delta CFO（分相位差）
    if all(c in df.columns for c in ["f1", "f2", "phi1", "phi2", "BER"]):
        plot_df = df.dropna(subset=['f1','f2','phi1','phi2','BER']).copy()
        plot_df['delta'] = plot_df['f2'] - plot_df['f1']
        delta_step = 1.0
        plot_df['delta_q'] = np.round(plot_df['delta'] / delta_step) * delta_step
        plot_df['phi_diff'] = wrap_2pi(plot_df['phi2'] - plot_df['phi1'])
        def bin_phi_diff2(x):
            step = 2*np.pi/8
            return float(int(np.floor(wrap_2pi(x)/step + 0.5)) % 8) * step
        plot_df['phi_bin']  = plot_df['phi_diff'].apply(bin_phi_diff2)

        agg = (plot_df
               .groupby(['phi_bin','delta_q'], as_index=False)
               .agg(N=('BER','count'),
                    mean_BER=('BER', 'mean'),
                    sem_BER=('BER', lambda x: np.std(x, ddof=1) / np.sqrt(max(1, len(x))))) )

        plt.figure(figsize=(9,5))
        for pbin in sorted([v for v in agg['phi_bin'].dropna().unique()]):
            sub = agg[agg['phi_bin']==pbin].sort_values('delta_q')
            if len(sub)==0: continue
            xs = sub['delta_q'].values
            y = sub['mean_BER'].values
            e = sub['sem_BER'].fillna(0).values
            plt.semilogy(xs, y, marker='o',
                         label=f'phi_diff={pbin:.2f} rad ({int(np.degrees(pbin))}°)')
            y_low = np.clip(y - e, 1e-12, 1.0)
            y_high = np.clip(y + e, 1e-12, 1.0)
            plt.fill_between(xs, y_low, y_high, alpha=0.15)
        plt.axvline(0, color='k', lw=0.8)
        plt.grid(True); plt.legend(ncol=2)
        plt.xlabel('delta = f2 - f1 (Hz)')
        plt.ylabel('Mean BER ± SEM (semilogy)')
        plt.title('BER vs delta CFO (grouped by phase diff, mean ± SEM)')
        save_fig2(os.path.join(out_dir, "ber_vs_delta_grouped.png"))

    # 7) 平均 BER vs amp CSV
    if 'amp' in df.columns and 'BER' in df.columns and len(df):
        mean_by_amp = df.groupby('amp', as_index=True)['BER'].mean().sort_index()
        out_amp_csv = os.path.join(out_dir, 'mean_ber_per_amp.csv')
        os.makedirs(out_dir, exist_ok=True)
        mean_by_amp.to_csv(out_amp_csv, header=['mean_BER'])
        print(f"[orig] Saved mean BER per amp to: {out_amp_csv}")
        print(mean_by_amp)


# ========== mode=snr_amp：test_snr_amp ==========
def viz_snr_amp(df: pd.DataFrame, out_dir: str):
    """
    期望 CSV 至少有：
      - BER1, BER2, snr, amp, mod1, mod2
    我们按 mod_pair = (mod1,mod2) 分四组，每组各画：
      1) BER1 & BER2 vs SNR (per AMP) - 两个子图
      2) BER1 & BER2 vs AMP (per SNR) - 两个子图
    """
    required = {"BER1", "BER2", "snr", "amp", "mod1", "mod2"}
    if not required.issubset(df.columns):
        print("[snr_amp] CSV 缺少必要列，跳过；需要：", required)
        return

    mod_pairs = sorted(df[['mod1','mod2']].dropna().drop_duplicates().values.tolist())
    for mod1, mod2 in mod_pairs:
        sub_df = df[(df['mod1']==mod1) & (df['mod2']==mod2)].copy()
        if len(sub_df) == 0: continue
        tag = f"{mod1}_{mod2}"

        # 1) BER1 & BER2 vs SNR（per amp）- 两个子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        
        # 子图1: BER1
        for a in sorted([v for v in sub_df['amp'].dropna().unique()]):
            sub_raw = sub_df[sub_df['amp']==a].dropna(subset=['BER1','snr'])
            if len(sub_raw)==0: continue
            grp = (sub_raw.groupby('snr', as_index=False)
                   .agg(mean_BER1=('BER1','mean'))
                   .sort_values('snr'))
            ax1.semilogy(grp['snr'], grp['mean_BER1'], marker='o', label=f"amp={a:.2f}")
        ax1.grid(True); ax1.legend(ncol=2)
        ax1.set_xlabel("SNR (dB)"); ax1.set_ylabel("Mean BER1 (log y)")
        ax1.set_title(f"BER1 vs SNR per AMP | {tag} ({mod1})")
        
        # 子图2: BER2
        for a in sorted([v for v in sub_df['amp'].dropna().unique()]):
            sub_raw = sub_df[sub_df['amp']==a].dropna(subset=['BER2','snr'])
            if len(sub_raw)==0: continue
            grp = (sub_raw.groupby('snr', as_index=False)
                   .agg(mean_BER2=('BER2','mean'))
                   .sort_values('snr'))
            ax2.semilogy(grp['snr'], grp['mean_BER2'], marker='o', label=f"amp={a:.2f}")
        ax2.grid(True); ax2.legend(ncol=2)
        ax2.set_xlabel("SNR (dB)"); ax2.set_ylabel("Mean BER2 (log y)")
        ax2.set_title(f"BER2 vs SNR per AMP | {tag} ({mod2})")
        
        plt.tight_layout()
        save_fig(os.path.join(out_dir, f"snr_amp_ber_vs_snr_{tag}.png"))

        # 2) BER1 & BER2 vs AMP（per SNR）- 两个子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        
        # 子图1: BER1
        for s in sorted([v for v in sub_df['snr'].dropna().unique()]):
            sub_raw = sub_df[sub_df['snr']==s].dropna(subset=['BER1','amp'])
            if len(sub_raw)==0: continue
            grp = (sub_raw.groupby('amp', as_index=False)
                   .agg(mean_BER1=('BER1','mean'))
                   .sort_values('amp'))
            ax1.semilogy(grp['amp'], grp['mean_BER1'], marker='o', label=f"SNR={s:g}dB")
        ax1.grid(True); ax1.legend(ncol=2)
        ax1.set_xlabel("Amplitude ratio (a)")
        ax1.set_ylabel("Mean BER1 (log y)")
        ax1.set_title(f"BER1 vs AMP per SNR | {tag} ({mod1})")
        
        # 子图2: BER2
        for s in sorted([v for v in sub_df['snr'].dropna().unique()]):
            sub_raw = sub_df[sub_df['snr']==s].dropna(subset=['BER2','amp'])
            if len(sub_raw)==0: continue
            grp = (sub_raw.groupby('amp', as_index=False)
                   .agg(mean_BER2=('BER2','mean'))
                   .sort_values('amp'))
            ax2.semilogy(grp['amp'], grp['mean_BER2'], marker='o', label=f"SNR={s:g}dB")
        ax2.grid(True); ax2.legend(ncol=2)
        ax2.set_xlabel("Amplitude ratio (a)")
        ax2.set_ylabel("Mean BER2 (log y)")
        ax2.set_title(f"BER2 vs AMP per SNR | {tag} ({mod2})")
        
        plt.tight_layout()
        save_fig(os.path.join(out_dir, f"snr_amp_ber_vs_amp_{tag}.png"))


# ========== mode=cfo_phase：test_cfo_phase ==========
def viz_cfo_phase(df: pd.DataFrame, out_dir: str):
    """
    期望 CSV 至少有：
      - BER1, BER2, f1, f2, phi1, phi2, mod1, mod2
    对 test_cfo_phase 来说：
      - f1 ≈ 0, f2 ∈ CFO_GRID_CFO_PHASE
      - phi1 ≈ 0, phi2 ≈ Δphi
    图：对每个 mod_pair，画 BER1 & BER2 vs (f2-f1)（横轴），每条曲线一个 Δphi。
    分成两个子图：BER1 和 BER2。
    """
    required = {"BER1", "BER2", "f1", "f2", "phi1", "phi2", "mod1", "mod2"}
    if not required.issubset(df.columns):
        print("[cfo_phase] CSV 缺少必要列，跳过；需要：", required)
        return

    df2 = df.dropna(subset=['f1','f2','phi1','phi2','BER1','BER2','mod1','mod2']).copy()
    df2['delta_cfo'] = df2['f2'] - df2['f1']
    df2['phi_diff'] = wrap_2pi(df2['phi2'] - df2['phi1'])

    def bin_phi(x):
        step = 2*np.pi/8
        return float(int(np.floor(wrap_2pi(x)/step + 0.5)) % 8) * step
    df2['phi_bin'] = df2['phi_diff'].apply(bin_phi)

    mod_pairs = sorted(df2[['mod1','mod2']].drop_duplicates().values.tolist())
    for mod1, mod2 in mod_pairs:
        sub = df2[(df2['mod1']==mod1) & (df2['mod2']==mod2)].copy()
        if len(sub)==0: continue
        tag = f"{mod1}_{mod2}"

        # 分别聚合 BER1 和 BER2
        agg1 = (sub
                .groupby(['phi_bin','delta_cfo'], as_index=False)
                .agg(N=('BER1','count'),
                     mean_BER1=('BER1','mean'),
                     sem_BER1=('BER1', lambda x: np.std(x, ddof=1) / np.sqrt(max(1, len(x))))) )
        
        agg2 = (sub
               .groupby(['phi_bin','delta_cfo'], as_index=False)
                .agg(N=('BER2','count'),
                     mean_BER2=('BER2','mean'),
                     sem_BER2=('BER2', lambda x: np.std(x, ddof=1) / np.sqrt(max(1, len(x))))) )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
        
        # 子图1: BER1
        for pbin in sorted([v for v in agg1['phi_bin'].dropna().unique()]):
            g = agg1[agg1['phi_bin']==pbin].sort_values('delta_cfo')
            if len(g)==0: continue
            xs = g['delta_cfo'].values
            y = g['mean_BER1'].values
            e = g['sem_BER1'].fillna(0).values
            lbl = f"phi_diff={pbin:.2f} rad ({int(np.degrees(pbin))}°)"
            ax1.semilogy(xs, y, marker='o', label=lbl)
            y_low = np.clip(y - e, 1e-12, 1.0)
            y_high = np.clip(y + e, 1e-12, 1.0)
            ax1.fill_between(xs, y_low, y_high, alpha=0.15)
        ax1.axvline(0, color='k', lw=0.8)
        ax1.grid(True); ax1.legend(ncol=2)
        ax1.set_xlabel("delta CFO = f2 - f1 (Hz)")
        ax1.set_ylabel("Mean BER1 ± SEM (log y)")
        ax1.set_title(f"BER1 vs delta CFO per phase diff | {tag} ({mod1})")
        
        # 子图2: BER2
        for pbin in sorted([v for v in agg2['phi_bin'].dropna().unique()]):
            g = agg2[agg2['phi_bin']==pbin].sort_values('delta_cfo')
            if len(g)==0: continue
            xs = g['delta_cfo'].values
            y = g['mean_BER2'].values
            e = g['sem_BER2'].fillna(0).values
            lbl = f"phi_diff={pbin:.2f} rad ({int(np.degrees(pbin))}°)"
            ax2.semilogy(xs, y, marker='o', label=lbl)
            y_low = np.clip(y - e, 1e-12, 1.0)
            y_high = np.clip(y + e, 1e-12, 1.0)
            ax2.fill_between(xs, y_low, y_high, alpha=0.15)
        ax2.axvline(0, color='k', lw=0.8)
        ax2.grid(True); ax2.legend(ncol=2)
        ax2.set_xlabel("delta CFO = f2 - f1 (Hz)")
        ax2.set_ylabel("Mean BER2 ± SEM (log y)")
        ax2.set_title(f"BER2 vs delta CFO per phase diff | {tag} ({mod2})")
        
        plt.tight_layout()
        save_fig(os.path.join(out_dir, f"cfo_phase_ber_vs_delta_{tag}.png"))


# ========== mode=delay：test_delay ==========
def viz_delay(df: pd.DataFrame, out_dir: str):
    """
    期望 CSV 至少有：
      - BER1, BER2, delay_diff, mod1, mod2
    图：对每个 mod_pair，分别画 BER1 和 BER2 vs delay_diff。
    """
    required = {"BER1", "BER2", "delay_diff", "mod1", "mod2"}
    if not required.issubset(df.columns):
        print("[delay] CSV 缺少必要列，跳过；需要：", required)
        return

    df2 = df.dropna(subset=['delay_diff','BER1','BER2','mod1','mod2']).copy()
    mod_pairs = sorted(df2[['mod1','mod2']].drop_duplicates().values.tolist())
    for mod1, mod2 in mod_pairs:
        sub = df2[(df2['mod1']==mod1) & (df2['mod2']==mod2)]
        if len(sub)==0: continue
        tag = f"{mod1}_{mod2}"

        # 分别聚合 BER1 和 BER2
        agg1 = (sub
               .groupby('delay_diff', as_index=False)
                .agg(N=('BER1','count'),
                     mean_BER1=('BER1','mean'),
                     sem_BER1=('BER1', lambda x: np.std(x, ddof=1) / np.sqrt(max(1, len(x))))) )
        agg1 = agg1.sort_values('delay_diff')

        agg2 = (sub
                .groupby('delay_diff', as_index=False)
                .agg(N=('BER2','count'),
                     mean_BER2=('BER2','mean'),
                     sem_BER2=('BER2', lambda x: np.std(x, ddof=1) / np.sqrt(max(1, len(x))))) )
        agg2 = agg2.sort_values('delay_diff')

        xs = agg1['delay_diff'].values
        y1 = agg1['mean_BER1'].values
        e1 = agg1['sem_BER1'].fillna(0).values
        y2 = agg2['mean_BER2'].values
        e2 = agg2['sem_BER2'].fillna(0).values

        plt.figure(figsize=(7,5))
        # 绘制 BER1
        plt.semilogy(xs, y1, marker='o', label=f'BER1 ({mod1})', linewidth=1.5)
        y1_low = np.clip(y1 - e1, 1e-12, 1.0)
        y1_high = np.clip(y1 + e1, 1e-12, 1.0)
        plt.fill_between(xs, y1_low, y1_high, alpha=0.15)
        
        # 绘制 BER2
        plt.semilogy(xs, y2, marker='s', label=f'BER2 ({mod2})', linewidth=1.5)
        y2_low = np.clip(y2 - e2, 1e-12, 1.0)
        y2_high = np.clip(y2 + e2, 1e-12, 1.0)
        plt.fill_between(xs, y2_low, y2_high, alpha=0.15)
        
        plt.grid(True)
        plt.legend()
        plt.xlabel("delay_diff (samples)")
        plt.ylabel("Mean BER ± SEM (log y)")
        plt.title(f"BER1 & BER2 vs delay_diff | {tag}")
        save_fig(os.path.join(out_dir, f"delay_ber_vs_delaydiff_{tag}.png"))


# ==================== 主入口 ====================
def get_default_paths(mode: str):
    """
    根据 mode 自动推断默认的 metrics_csv 和 out_dir 路径。
    """
    base_dir = "./src/results"
    
    mode_to_dir = {
        "snr-amp": "snr-amp",
        "cfo-phase": "cfo-phase",
        "delay": "delay",
        "orig": "qpsk_all",  # orig 模式默认使用 qpsk_all 目录
    }
    
    result_dir = mode_to_dir.get(mode, mode)
    default_metrics_csv = os.path.join(base_dir, result_dir, "metrics_all.csv")
    default_out_dir = os.path.join(base_dir, result_dir, "figs")
    
    return default_metrics_csv, default_out_dir


def main():
    parser = argparse.ArgumentParser("Visualization for SignalSeparator metrics CSV")
    parser.add_argument("--mode", type=str, default="orig",
                        choices=["orig", "snr-amp", "cfo-phase", "delay"],
                        help="可视化模式，会根据模式自动推断 metrics_csv 和 out_dir")
    parser.add_argument("--metrics_csv", type=str, default=None,
                        help="test_sim_SignalSeparator.py 输出的 metrics_all.csv（可选，会根据 mode 自动推断）")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="输出目录（可选，会根据 mode 自动推断）")
    args = parser.parse_args()

    # 根据 mode 自动推断默认路径
    default_metrics_csv, default_out_dir = get_default_paths(args.mode)
    
    # 如果用户提供了参数，优先使用用户提供的；否则使用自动推断的默认值
    metrics_csv = args.metrics_csv if args.metrics_csv is not None else default_metrics_csv
    out_dir = args.out_dir if args.out_dir is not None else default_out_dir
    
    # 检查 metrics_csv 文件是否存在
    if not os.path.exists(metrics_csv):
        print(f"[ERROR] metrics_csv 文件不存在: {metrics_csv}")
        print(f"[INFO] 自动推断的路径: {default_metrics_csv}")
        print(f"[INFO] 请检查文件路径或使用 --metrics_csv 手动指定")
        return
    
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(metrics_csv)
    print(f"[viz] Loaded {len(df)} rows from {metrics_csv}")
    print(f"[viz] Output directory: {out_dir}")
    print(f"[viz] Mode: {args.mode}")

    if args.mode == "orig":
        viz_orig(df, out_dir)
    elif args.mode == "snr-amp":
        viz_snr_amp(df, out_dir)
    elif args.mode == "cfo-phase":
        viz_cfo_phase(df, out_dir)
    elif args.mode == "delay":
        viz_delay(df, out_dir)

    print(f"[viz] Done. Figures saved to {out_dir}")

if __name__ == "__main__":
    main()
