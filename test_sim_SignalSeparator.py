# -*- coding: utf-8 -*-
import os, re, math, argparse, numpy as np, pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
from scipy.signal import convolve

from compensation import costas_loop
from model_complex import SignalSeparator

plt.rcParams['font.size'] = 13
plt.rcParams['axes.unicode_minus'] = False
torch.backends.cudnn.benchmark = True

# ==================== 分布式工具 ====================
def dist_is_initialized():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if dist_is_initialized() else 0

def get_world_size():
    return dist.get_world_size() if dist_is_initialized() else 1

def setup_ddp(backend="nccl"):
    if dist_is_initialized():
        return
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend=backend, init_method="env://")

def cleanup_ddp():
    if dist_is_initialized():
        dist.barrier()
        dist.destroy_process_group()

# ==================== DSP 基本件 ====================
beta, sps, num_taps = 0.33, 8, 64

def rc_filter(beta, sps, num_taps):
    t = np.arange(-num_taps//2, num_taps//2) / sps
    with np.errstate(divide='ignore', invalid='ignore'):
        h = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t) ** 2)
        h[np.isnan(h)] = 1.0 - beta + (4 * beta / np.pi)
    # 能量归一化：sum |h|^2 = 1
    h = h / np.sqrt(np.sum(h**2))
    return h

rc = rc_filter(beta, sps, num_taps)

def qpsk_demod(symbols):
    """
    与你的 qpsk_mod 一致：
      00 -> (+,+)
      01 -> (-,+)
      10 -> (+,-)
      11 -> (-,-)
    你之前是 1±1j / sqrt(2)，这里乘回 sqrt(2) 再判决。
    """
    bits = []
    sym = symbols * np.sqrt(2)
    for s in sym:
        if s.real >= 0 and s.imag >= 0: b1, b2 = 0, 0
        elif s.real < 0 and s.imag >= 0: b1, b2 = 0, 1
        elif s.real >= 0 and s.imag < 0: b1, b2 = 1, 0
        else: b1, b2 = 1, 1
        bits.extend([b1, b2])
    return np.array(bits, dtype=np.int8)

def align_phase(ref, est):
    """
    对 est 乘以 e^{jθ}，使其在最小二乘意义上与 ref 相位对齐。
    """
    c = np.mean(ref * np.conj(est) + 1e-12)
    a = np.angle(c)
    return est * np.exp(1j * (-a))

def wrap_2pi(x): return np.mod(x, 2*np.pi)

def evm_rms(ref_syms, est_syms):
    num = np.mean(np.abs(est_syms - ref_syms)**2)
    den = np.mean(np.abs(ref_syms)**2) + 1e-12
    return float(np.sqrt(num / den))

def find_best_offset(y_mf, sps):
    """
    在 0..sps-1 中搜索能量最大的抽样相位。
    """
    best_off = 0
    best_eng = -1.0
    for off in range(sps):
        sym = y_mf[off::sps]
        eng = np.mean(np.abs(sym)**2)
        if eng > best_eng:
            best_eng = eng
            best_off = off
    return best_off

def mf_and_sample(wave, sps, rc, num_taps, guard_sym=None):
    """
    匹配滤波 + best_offset + 抽样 + 去掉头尾 guard_sym 个符号 + 幅度归一化
    返回符号序列（complex ndarray）。
    """
    if guard_sym is None:
        guard_sym = num_taps // sps  # 比如 64/8 = 8 个符号

    if wave is None or len(wave) == 0:
        return np.zeros(0, dtype=np.complex64)

    # 匹配滤波
    y_mf = convolve(wave, rc, mode='same')

    # 在一个符号周期内搜索最佳采样相位
    off = find_best_offset(y_mf, sps)

    # 抽样得到符号
    syms = y_mf[off::sps]

    # 去掉滤波边缘的若干个符号
    if len(syms) <= 2 * guard_sym:
        return np.zeros(0, dtype=np.complex64)
    syms = syms[guard_sym:-guard_sym]

    # 幅度归一化
    m = np.mean(np.abs(syms))
    if m > 0:
        syms = syms / m

    return syms.astype(np.complex64)

def slice_bits_to_match_syms(bits_full: np.ndarray, n_syms_used: int):
    """
    bits_full: 形如 [0,1,1,0,...]，长度约等于 2 * 总符号数
    n_syms_used: mf_and_sample 后保留的符号数

    这里假设：
      总符号数 n_sym_total = len(bits_full) / 2
      中间 n_syms_used 个符号对应 mf_and_sample 保留的那部分，
      两边各砍 (n_sym_total - n_syms_used)/2 个符号。
    """
    if len(bits_full) == 0 or n_syms_used <= 0:
        return np.zeros(0, dtype=np.int8)

    n_sym_total = len(bits_full) // 2
    n_syms_used = min(n_syms_used, n_sym_total)
    if n_sym_total <= n_syms_used:
        return bits_full[:2*n_syms_used]

    guard_sym = max((n_sym_total - n_syms_used) // 2, 0)
    start = guard_sym * 2
    end = start + 2 * n_syms_used
    end = min(end, len(bits_full))
    return bits_full[start:end]

# ==================== 解析 test_all 的 params ====================
def parse_params_all(pstr: str):
    if not isinstance(pstr, str) or not pstr:
        return {}
    parts = [x.strip() for x in pstr.split(',')]
    floats = []
    for x in parts:
        try: floats.append(float(x))
        except: pass
    snr = floats[0] if len(floats) >= 1 else None
    amp = floats[1] if len(floats) >= 2 else None

    m_f1  = re.search(r"f_off1=([\-0-9.]+)\s*Hz", pstr)
    m_f2  = re.search(r"f_off2=([\-0-9.]+)\s*Hz", pstr)
    m_p1  = re.search(r"phi1=([\-0-9.]+)\s*rad", pstr)
    m_p2  = re.search(r"phi2=([\-0-9.]+)\s*rad", pstr)
    m_rep = re.search(r"rep=([0-9]+)", pstr)

    return {
        'snr': snr,
        'amp': amp,
        'f_off1': float(m_f1.group(1)) if m_f1 else None,
        'f_off2': float(m_f2.group(1)) if m_f2 else None,
        'phi1': float(m_p1.group(1)) if m_p1 else None,
        'phi2': float(m_p2.group(1)) if m_p2 else None,
        'rep': int(m_rep.group(1)) if m_rep else None
    }

# ==================== 画图工具（仅 rank0 使用） ====================
def save_fig(path, dpi=140):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()

def plot_time_overlay(sig_pred, sig_true, title, path, n_show=1024):
    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.plot(np.real(sig_true[:n_show]), label='True (Real)', lw=1)
    plt.plot(np.real(sig_pred[:n_show]), label='Pred (Real)', lw=1, alpha=0.8)
    plt.grid(True); plt.legend(); plt.title(title + " — Real")
    plt.subplot(2,1,2)
    plt.plot(np.imag(sig_true[:n_show]), label='True (Imag)', lw=1)
    plt.plot(np.imag(sig_pred[:n_show]), label='Pred (Imag)', lw=1, alpha=0.8)
    plt.grid(True); plt.legend(); plt.title(title + " — Imag")
    save_fig(path)

def plot_constellation(ps, gs, title, path, s=8):
    plt.figure(figsize=(5.8,5.8))
    plt.scatter(np.real(gs), np.imag(gs), s=s, label='True', alpha=0.9)
    plt.scatter(np.real(ps), np.imag(ps), s=s, label='Pred', alpha=0.9)
    plt.axhline(0, lw=0.8, color='k'); plt.axvline(0, lw=0.8, color='k')
    plt.grid(True); plt.legend(); plt.title(title)
    plt.xlabel('I'); plt.ylabel('Q')
    save_fig(path)

# ==================== Dataset ====================
class TestAllDataset(Dataset):
    def __init__(self, entries): self.entries = entries
    def __len__(self): return len(self.entries)
    def __getitem__(self, idx):
        e = self.entries[idx]
        def c2ri(x):
            x = np.asarray(x)
            return np.stack([x.real.astype(np.float32), x.imag.astype(np.float32)], axis=0)
        raw = e.get('params', "")
        params_str = ", ".join(map(str, raw)) if isinstance(raw, (list, tuple)) else str(raw)
        b1 = np.asarray(e.get('bits1', np.array([-1], dtype=np.int8)), dtype=np.int8)
        b2 = np.asarray(e.get('bits2', np.array([-1], dtype=np.int8)), dtype=np.int8)
        return {
            'mixsignal_ri': c2ri(e['mixsignal']),
            'rfsignal1_ri': c2ri(e['rfsignal1']),
            'rfsignal2_ri': c2ri(e['rfsignal2']),
            'bits1': b1, 'bits2': b2, 'params': params_str
        }

# ==================== 主流程 ====================
def main():
    parser = argparse.ArgumentParser("DDP Inference for SignalSeparator (new demod, fixed BER)")
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--test_data_path', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='./src/pics/test_all_viz_ddp_newdemod')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--max_const_plots', type=int, default=12)
    parser.add_argument('--max_time_plots', type=int, default=12)
    parser.add_argument('--amp', action='store_true', default=True,
                        help='use torch.cuda.amp for inference')
    # 新增：坏样本可视化参数
    parser.add_argument('--bad_ber_thresh', type=float, default=0.3,
                        help='BER 大于该阈值的样本将被额外可视化')
    parser.add_argument('--max_bad_plots', type=int, default=10,
                        help='最多保存多少个高 BER 样本的可视化')
    args = parser.parse_args()

    # 分布式初始化 + 设备绑定
    setup_ddp(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    rank, world = get_rank(), get_world_size()
    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"[DDP] world_size={world}, device={device}")
        print(f"[IO] ckpt={args.ckpt_path}\n[IO] test={args.test_data_path}\n[IO] out={args.out_dir}")

    # 加载模型
    model = SignalSeparator().to(device)

    state = torch.load(args.ckpt_path, map_location='cpu')
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        model.load_state_dict(state, strict=False)
    model.eval()
    if rank == 0:
        print("Params:", sum(p.numel() for p in model.parameters()))

    # 读取数据 & 切分
    loaded_data = torch.load(args.test_data_path)  # list of dict
    dataset = TestAllDataset(loaded_data)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank,
                                 shuffle=False, drop_last=False) if dist_is_initialized() else None
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False if sampler else False,
                        num_workers=args.num_workers, pin_memory=True, sampler=sampler)

    OUT_DIR = args.out_dir
    TMP_DIR = os.path.join(OUT_DIR, "_tmp_csv")
    if rank == 0:
        os.makedirs(TMP_DIR, exist_ok=True)

    DO_PLOT = (rank == 0)
    max_const = args.max_const_plots if DO_PLOT else 0
    max_time  = args.max_time_plots if DO_PLOT else 0
    const_cnt = 0
    time_cnt  = 0
    bad_cnt   = 0  # 统计坏样本图数量

    results = []
    autocast_enabled = (device.type == 'cuda') and args.amp

    # 推理
    with torch.no_grad():
        for bi, batch in enumerate(loader, 1):
            x = batch['mixsignal_ri'].to(device)  # (B,2,T)

            if autocast_enabled:
                with torch.cuda.amp.autocast():
                    y = model(x)
            else:
                y = model(x)
            if isinstance(y, (tuple, list)):
                # 现在模型返回的是长度为 4 的 list，每个 [B,1,T]
                # 拼成 [B,4,T]
                y = torch.cat(y, dim=1)

            # 期望 y: (B,4,T)
            y_np = y.detach().cpu().numpy()
            p1 = y_np[:, 0:2, :]
            p2 = y_np[:, 2:4, :]


            g1 = batch['rfsignal1_ri'].cpu().numpy()  # (B,2,T)
            g2 = batch['rfsignal2_ri'].cpu().numpy()

            # 归一化 MSE（与你训练一致）
            loss1 = ((p1 - g1) ** 2).mean(axis=(1,2)) / (np.linalg.norm(g1, axis=(1,2)) + 1e-12)
            loss2 = ((p2 - g2) ** 2).mean(axis=(1,2)) / (np.linalg.norm(g2, axis=(1,2)) + 1e-12)

            B = p1.shape[0]
            for i in range(B):
                # 还原为复基带波形
                pr1 = (p1[i,0] + 1j*p1[i,1]).astype(np.complex64)
                pr2 = (p2[i,0] + 1j*p2[i,1]).astype(np.complex64)
                gt1 = (g1[i,0] + 1j*g1[i,1]).astype(np.complex64)
                gt2 = (g2[i,0] + 1j*g2[i,1]).astype(np.complex64)

                # params
                params_batch = batch['params']
                if isinstance(params_batch, list) or isinstance(params_batch, tuple):
                    params_i = params_batch[i]
                else:
                    params_i = str(params_batch)
                meta = parse_params_all(str(params_i))

                # ========= 解调链路：Costas + MF + best_offset + guard =========
                # Costas 环（各路独立）
                pr1_c, _ = costas_loop(pr1, loop_bandwidth=0.001, sps=sps)
                pr2_c, _ = costas_loop(pr2, loop_bandwidth=0.001, sps=sps)
                gt1_c, _ = costas_loop(gt1, loop_bandwidth=0.001, sps=sps)
                gt2_c, _ = costas_loop(gt2, loop_bandwidth=0.001, sps=sps)

                # RC 匹配滤波 + 抽样 + guard + 幅度归一化
                ps1 = mf_and_sample(pr1_c, sps, rc, num_taps)
                ps2 = mf_and_sample(pr2_c, sps, rc, num_taps)
                gs1 = mf_and_sample(gt1_c, sps, rc, num_taps)
                gs2 = mf_and_sample(gt2_c, sps, rc, num_taps)

                # 如果太短，跳过该样本
                L1 = min(len(ps1), len(gs1))
                L2 = min(len(ps2), len(gs2))
                if L1 <= 0 or L2 <= 0:
                    continue

                ps1 = ps1[:L1]; gs1 = gs1[:L1]
                ps2 = ps2[:L2]; gs2 = gs2[:L2]

                # 相位对齐（只对 Pred 调整，使其最接近 GT）
                ps1_aligned = align_phase(gs1, ps1)
                ps2_aligned = align_phase(gs2, ps2)

                # EVM
                evm1 = evm_rms(gs1, ps1_aligned)
                evm2 = evm_rms(gs2, ps2_aligned)

                # ============ BER：用数据集里的 bits1/bits2 作为参考 ============
                # DataLoader 默认把 numpy 转成 tensor，这里转回 numpy
                b1_full = batch['bits1'][i].cpu().numpy().astype(np.int8)
                b2_full = batch['bits2'][i].cpu().numpy().astype(np.int8)

                # 根据 mf_and_sample 后的符号数，把 bits 截成对应中间一段
                b1_ref = slice_bits_to_match_syms(b1_full, len(gs1))
                b2_ref = slice_bits_to_match_syms(b2_full, len(gs2))

                # Pred 对齐后符号解调为 bits
                b1_hat = qpsk_demod(ps1_aligned)
                b2_hat = qpsk_demod(ps2_aligned)

                Lb1 = min(len(b1_hat), len(b1_ref))
                Lb2 = min(len(b2_hat), len(b2_ref))

                ber1 = float(np.mean(b1_hat[:Lb1] != b1_ref[:Lb1])) if Lb1 > 0 else 1.0
                ber2 = float(np.mean(b2_hat[:Lb2] != b2_ref[:Lb2])) if Lb2 > 0 else 1.0
                ber  = 0.5 * (ber1 + ber2)

                phi1 = meta.get('phi1', None); phi2 = meta.get('phi2', None)
                phi_diff = None if (phi1 is None or phi2 is None) \
                           else float(wrap_2pi(phi2 - phi1))

                results.append({
                    'loss1': float(loss1[i]), 'loss2': float(loss2[i]),
                    'BER1': ber1, 'BER2': ber2, 'BER': ber,
                    'snr': meta.get('snr'), 'amp': meta.get('amp'),
                    'f1': meta.get('f_off1'), 'f2': meta.get('f_off2'),
                    'phi1': phi1, 'phi2': phi2, 'phi_diff': phi_diff,
                    'rep': meta.get('rep'),
                    'evm1': evm1, 'evm2': evm2
                })

                # ====== 仅 rank0 少量绘图 ======
                if DO_PLOT:
                    fostr = f"f1={meta.get('f_off1','NA')}_f2={meta.get('f_off2','NA')}"
                    if (phi1 is not None) and (phi2 is not None):
                        phstr = f"phi1={phi1:.2f}_phi2={phi2:.2f}"
                    else:
                        phstr = "phi=NA"

                    # 常规少量可视化
                    if const_cnt < max_const:
                        plot_constellation(ps1_aligned, gs1,
                                           f"Sig1 | {fostr} | {phstr}",
                                           os.path.join(OUT_DIR, f"const_sig1_{bi}_{i}.png"))
                        plot_constellation(ps2_aligned, gs2,
                                           f"Sig2 | {fostr} | {phstr}",
                                           os.path.join(OUT_DIR, f"const_sig2_{bi}_{i}.png"))
                        const_cnt += 2

                    if time_cnt < max_time:
                        plot_time_overlay(pr1, gt1,
                                          f"Sig1 | {fostr} | {phstr}",
                                          os.path.join(OUT_DIR, f"time_sig1_{bi}_{i}.png"))
                        plot_time_overlay(pr2, gt2,
                                          f"Sig2 | {fostr} | {phstr}",
                                          os.path.join(OUT_DIR, f"time_sig2_{bi}_{i}.png"))
                        time_cnt += 2

                        # ========== 高 BER 样本调试与可视化 ==========
                        if (ber >= args.bad_ber_thresh):
                            # ---------- 1) 打印 pred / GT 的能量 ----------
                            energy_pr1 = float(np.mean(np.abs(pr1)**2))
                            energy_pr2 = float(np.mean(np.abs(pr2)**2))
                            energy_gt1 = float(np.mean(np.abs(gt1)**2))
                            energy_gt2 = float(np.mean(np.abs(gt2)**2))

                            print(f"[BAD SAMPLE] bi={bi}, i={i}, BER={ber:.4f}")
                            print(f"    E_pred1={energy_pr1:.3e}, E_gt1={energy_gt1:.3e}")
                            print(f"    E_pred2={energy_pr2:.3e}, E_gt2={energy_gt2:.3e}")

                            # ---------- 2) 打印 Separator 的 mask 统计 ----------
                            if hasattr(model.separator, "last_mask_stats") and model.separator.last_mask_stats:
                                m = model.separator.last_mask_stats
                                print(f"    Mask stats: mean={m['mean']:.4f}, min={m['min']:.4f}, max={m['max']:.4f}")
                            else:
                                print("    Mask stats: (no data)")

                            # ---------- 3) 可视化 bad plots（原逻辑） ----------
                            if bad_cnt < args.max_bad_plots:
                                bad_dir = os.path.join(OUT_DIR, "bad_samples")
                                os.makedirs(bad_dir, exist_ok=True)
                                title1 = f"[BAD BER={ber:.3f}] Sig1 | {fostr}"
                                title2 = f"[BAD BER={ber:.3f}] Sig2 | {fostr}"

                                plot_constellation(ps1_aligned, gs1,
                                    title1,
                                    os.path.join(bad_dir, f"bad_const_sig1_b{bi}_i{i}_ber{ber:.3f}.png"))
                                plot_constellation(ps2_aligned, gs2,
                                    title2,
                                    os.path.join(bad_dir, f"bad_const_sig2_b{bi}_i{i}_ber{ber:.3f}.png"))

                                plot_time_overlay(pr1, gt1,
                                    title1,
                                    os.path.join(bad_dir, f"bad_time_sig1_b{bi}_i{i}_ber{ber:.3f}.png"))
                                plot_time_overlay(pr2, gt2,
                                    title2,
                                    os.path.join(bad_dir, f"bad_time_sig2_b{bi}_i{i}_ber{ber:.3f}.png"))

                                bad_cnt += 1

            if rank == 0:
                print(f"[Rank0][Batch {bi}] done.  (bad_plots={bad_cnt})")

    # 每个 rank 写各自的 csv
    tmp_csv = os.path.join(TMP_DIR, f"metrics_rank{rank}.csv")
    os.makedirs(TMP_DIR, exist_ok=True)
    pd.DataFrame(results).to_csv(tmp_csv, index=False)

    # 同步，随后 rank0 合并
    if dist_is_initialized():
        dist.barrier()

    if rank == 0:
        # 合并所有 rank 的 CSV
        dfs = []
        for r in range(world):
            path = os.path.join(TMP_DIR, f"metrics_rank{r}.csv")
            if os.path.exists(path):
                dfs.append(pd.read_csv(path))
        df = pd.concat(dfs, ignore_index=True) if len(dfs) else pd.DataFrame()
        final_csv = os.path.join(OUT_DIR, "metrics_test_all.csv")
        df.to_csv(final_csv, index=False)
        print("Overall mean BER:", df["BER"].mean() if "BER" in df.columns and len(df) else "N/A")

        # ===== 可视化（全部在 rank0 进行） =====
        def save_fig2(path, dpi=150):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.tight_layout(); plt.savefig(path, dpi=dpi); plt.close()

        # 1) BER 分布/CDF
        if "BER" in df.columns and len(df):
            plt.figure(figsize=(6,4))
            plt.hist(df["BER"].clip(0,1), bins=40, density=True, alpha=0.8)
            plt.xlabel("BER"); plt.ylabel("Density"); plt.title("BER Histogram")
            save_fig2(os.path.join(OUT_DIR, "ber_hist.png"))

            plt.figure(figsize=(6,4))
            ber_sorted = np.sort(df["BER"].values)
            p = np.linspace(0,1,len(ber_sorted))
            plt.plot(ber_sorted, p)
            plt.xlabel("BER"); plt.ylabel("CDF"); plt.title("BER CDF")
            save_fig2(os.path.join(OUT_DIR, "ber_cdf.png"))

        # 2) EVM CDF（分 SNR）
        def plot_evm_cdf_by_snr(col, name):
            if col not in df.columns: return
            plt.figure(figsize=(7,5))
            snr_vals = sorted([v for v in df['snr'].dropna().unique()]) if 'snr' in df.columns else []
            for snr in snr_vals:
                sub = df[df['snr']==snr][col].dropna().values
                if len(sub) == 0: continue
                xs = np.sort(sub); pp = np.linspace(0,1,len(xs))
                plt.plot(xs, pp, label=f"SNR={snr:g}dB")
            plt.grid(True); plt.legend()
            plt.xlabel(name); plt.ylabel("CDF"); plt.title(f"{name} CDF by SNR")
            save_fig2(os.path.join(OUT_DIR, f"{name.lower()}_cdf_by_snr.png"))

        plot_evm_cdf_by_snr('evm1', 'EVM1')
        plot_evm_cdf_by_snr('evm2', 'EVM2')

        # 3) BER vs SNR（分幅度比）
        if all(c in df.columns for c in ["amp", "snr", "BER"]):
            plt.figure(figsize=(8,5))
            for a in sorted([v for v in df['amp'].dropna().unique()]):
                sub = df[df['amp']==a].groupby('snr', as_index=False)['BER'].mean().sort_values('snr')
                if len(sub)==0: continue
                plt.plot(sub['snr'], sub['BER'], marker='o', label=f"amp={a:.2f}")
            plt.grid(True); plt.legend(ncol=2)
            plt.xlabel("SNR (dB)"); plt.ylabel("Mean BER"); plt.title("BER vs SNR (per amplitude)")
            save_fig2(os.path.join(OUT_DIR, "ber_vs_snr_per_amp.png"))

        # 4) BER vs amp（分 SNR）
        if all(c in df.columns for c in ["amp", "snr", "BER"]):
            plt.figure(figsize=(8,5))
            for s in sorted([v for v in df['snr'].dropna().unique()]):
                sub = df[df['snr']==s].groupby('amp', as_index=False)['BER'].mean().sort_values('amp')
                if len(sub)==0: continue
                plt.plot(sub['amp'], sub['BER'], marker='o', label=f"SNR={s:g}dB")
            plt.grid(True); plt.legend(ncol=2)
            plt.xlabel("Amplitude ratio (a=|s2|/|s1|)"); plt.ylabel("Mean BER"); plt.title("BER vs amplitude (per SNR)")
            save_fig2(os.path.join(OUT_DIR, "ber_vs_amp_per_snr.png"))

        # 5) Heatmap (f1,f2) 按相位差分层
        if all(c in df.columns for c in ["f1", "f2", "BER", "phi1", "phi2"]):
            def bin_phi_diff(x):
                if x is None or (isinstance(x, float) and np.isnan(x)): return None
                step = 2*np.pi/8
                return float(int(np.floor(wrap_2pi(x)/step + 0.5)) % 8) * step
            df['phi_diff'] = wrap_2pi(df['phi2'] - df['phi1'])
            df['phi_diff_bin'] = df['phi_diff'].apply(bin_phi_diff)
            mean_fmap = (df.dropna(subset=['f1','f2','BER'])
                         .groupby(['phi_diff_bin','f1','f2'], as_index=False)['BER'].mean())
            phi_bins = sorted([v for v in mean_fmap['phi_diff_bin'].dropna().unique()])

            for pbin in phi_bins:
                sub = mean_fmap[mean_fmap['phi_diff_bin']==pbin]
                if len(sub)==0: continue
                piv = sub.pivot(index='f2', columns='f1', values='BER').sort_index().sort_index(axis=1)
                plt.figure(figsize=(7,6))
                im = plt.imshow(piv.values, aspect='auto', origin='lower',
                                extent=[piv.columns.min(), piv.columns.max(),
                                        piv.index.min(), piv.index.max()])
                plt.colorbar(im, label='Mean BER'); plt.xlabel('f1 (Hz)'); plt.ylabel('f2 (Hz)')
                deg = int(np.degrees(pbin)); plt.title(f'BER Heatmap | phi_diff={pbin:.2f} rad ({deg}°)')
                save_fig2(os.path.join(OUT_DIR, f"heatmap_f1f2_phidiff_{pbin:.2f}rad.png"))

        # 6) BER vs delta = f2 - f1（分相位差）
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
                   .agg(BER_mean=('BER','mean'),
                        BER_std =('BER','std'),
                        N=('BER','count')))
            agg['BER_sem'] = agg['BER_std'] / np.sqrt(agg['N'].clip(lower=1))

            plt.figure(figsize=(9,5))
            for pbin in sorted([v for v in agg['phi_bin'].dropna().unique()]):
                sub = agg[agg['phi_bin']==pbin].sort_values('delta_q')
                if len(sub)==0: continue
                plt.plot(sub['delta_q'], sub['BER_mean'], marker='o',
                         label=f'phi_diff={pbin:.2f} rad ({int(np.degrees(pbin))}°)')
                y = sub['BER_mean'].values
                e = sub['BER_sem'].fillna(0).values
                plt.fill_between(sub['delta_q'].values, y-e, y+e, alpha=0.15)
            plt.axvline(0, color='k', lw=0.8)
            plt.grid(True); plt.legend(ncol=2)
            plt.xlabel('delta = f2 - f1 (Hz)')
            plt.ylabel('Mean BER ± SEM')
            plt.title('BER vs delta CFO (grouped by phase diff, mean ± SEM)')
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, "ber_vs_delta_grouped.png"), dpi=150)
            plt.close()

        print("All visualizations saved to:", OUT_DIR)

    cleanup_ddp()

if __name__ == '__main__':
    main()
