import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
import torch
import os
import argparse
import itertools
import hashlib

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

# ============= 参数解析 =============
parser = argparse.ArgumentParser(
    description="QPSK 数据集生成器（train: 多N规模随机采样 + 分片保存；"
                "test_eqcfo: 固定间隔栅格采样，相同cfo；"
                "test_diffcfo: 固定间隔栅格采样，不同cfo；"
                "test_all: 固定间隔栅格采样，全集（f1,f2独立；phi1,phi2独立；幅度比与SNR入网格, 含delay））"
)
parser.add_argument("--mode", type=str, default="train",
                    choices=["train", "test_eqcfo", "test_diffcfo", "test_all"],
                    help="选择生成训练集或测试集")
parser.add_argument("--test_repeats", type=int, default=5,
                    help="测试集中每组超参数要生成的块数量（至少1个）")
parser.add_argument("--shard_size", type=int, default=10000,
                    help="train 模式下每个分片包含的样本数（默认 10000）")
parser.add_argument("--train_sizes", type=str, default="auto",
                    help='仅用于 --mode=train。可选：'
                         '"auto"（默认，使用预设列表），'
                         '或逗号分隔的数字/带k：如 "5k,10k,50k" 或 "5000,10000,50000"')
# 可选：是否将保存的数据转为 complex64（更省空间）
parser.add_argument("--save_complex64", action="store_true",
                    help="保存前将 complex 数据转换为 complex64（节省约一半空间）")

args = parser.parse_args()

# ============= 通用参数 =============
beta = 0.33
sps = 8
fs = 12e6
num_taps = 64
input_len = 3072  # 每块样本点数
bit_len = int(2 * (input_len // sps))  # = 768 bits（QPSK每符号2bit）
assert (bit_len // 2) * sps == input_len

# ============= 训练用随机区间（更贴近真实，增强泛化） =============
# CFO 取值范围（正负对称使用随机符号）
FREQ_RANGE_TRAIN = (0, 200)            # Hz，两个信号各自独立取值并随机符号
PHASE1_RANGE = (0.0, 2*np.pi)          # rad，信号1初相位 φ1
PHASE2_RANGE = (0.0, 2*np.pi)          # rad，信号2初相位 φ2
AMP_RANGE_TRAIN = (0.30, 0.90)         # 幅度比 a = |s2|/|s1|
SNR_RANGE_TRAIN = (12.0, 30.0)         # dB，全局SNR（合路后统一加噪）

# 注意：train 模式不再使用 DELAY_RANGE；改为对两路分别做 0~(sps-1) 采样级时延
DELAY_RANGE = None

# ============= 测试集网格（eq/diff 维持原样） =============
FREQ_GRID = np.linspace(0, 200, 10)                 # Hz（eq/diff 使用）
PHASE_GRID = np.linspace(0.0, 2*np.pi, 8, endpoint=False)  # eq/diff：单一路相位（phi0）

AMP_GRID = None
DELAY_GRID = None
DEFAULT_AMP_FOR_TEST = 0.5
DEFAULT_DELAY_FOR_TEST = 0
AMP_GRID_EFF = [DEFAULT_AMP_FOR_TEST] if AMP_GRID is None else AMP_GRID
DELAY_GRID_EFF = [DEFAULT_DELAY_FOR_TEST] if DELAY_GRID is None else DELAY_GRID

# ============= test_all 的网格（两路独立 + 幅度比 + SNR；无 delay） =============
PHASE1_GRID = np.linspace(0.0, 2*np.pi, 8, endpoint=False)  # φ1
PHASE2_GRID = np.linspace(0.0, 2*np.pi, 8, endpoint=False)  # φ2
FREQ1_GRID  = FREQ_GRID                                     # f1
FREQ2_GRID  = FREQ_GRID                                     # f2
AMP_ALL_GRID = np.round(np.linspace(0.30, 0.90, 5), 2)      # a
SNR_GRID     = np.array([12.0, 18.0, 24.0, 30.0])           # SNR(dB)
# sps = 8 时，对应 0, T/4, T/2, 3T/4 -> 0, 2, 4, 6 samples
DELAY_SAMP_GRID = np.array([0, 2, 4, 6], dtype=int)



# 旧两种测试模式固定 SNR
TEST_SNR_DB = 23.8

# ============= 数据集保存目录 =============
save_dir = '/nas/datasets/yixin/PCMA/sim_data'
os.makedirs(save_dir, exist_ok=True)

# ============= 预设训练规模列表（多种 N） =============
N_LIST_DEFAULT = [10_000, 20_000, 50_000, 100_000, 200_000]

def parse_train_sizes(s: str):
    """解析 --train_sizes 参数。支持 'auto' 或 '5k,10k,50000' 等。"""
    if s.strip().lower() == "auto":
        return N_LIST_DEFAULT
    parts = [p.strip().lower() for p in s.split(",") if p.strip()]
    out = []
    for p in parts:
        if p.endswith("k"):
            val = float(p[:-1]) * 1000
        else:
            val = float(p)
        out.append(int(val))
    # 去重并排序
    out = sorted(set(out))
    return out

def qpsk_mod(bits):
    # Gray-like 映射
    symbols = []
    for i in range(0, len(bits), 2):
        b1, b2 = bits[i], bits[i+1]
        if b1 == 0 and b2 == 0:
            symbols.append(1 + 1j)
        elif b1 == 0 and b2 == 1:
            symbols.append(-1 + 1j)
        elif b1 == 1 and b2 == 0:
            symbols.append(1 - 1j)
        else:
            symbols.append(-1 - 1j)
    return np.array(symbols) / np.sqrt(2)

def rc_filter(beta, sps, num_taps):
    t = np.arange(-num_taps//2, num_taps//2) / sps
    with np.errstate(divide='ignore', invalid='ignore'):
        h = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t) ** 2)
        h[np.isnan(h)] = 1.0 - beta + (4 * beta / np.pi)
    h = h / np.sqrt(np.sum(h**2))
    return h

rc = rc_filter(beta, sps, num_taps)

def awgn_with_seed(signal, snr_db, seed=None):
    signal_power = np.mean(np.abs(signal)**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    noise = np.sqrt(noise_power / 2) * (rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal)))
    return signal + noise

def energy_normalize_dataset(dataset):
    energies = [np.mean(np.abs(e['mixsignal'])**2) for e in dataset]
    mean_e = np.mean(energies) if energies else 1.0
    scale = np.sqrt(mean_e)
    for e in dataset:
        e['mixsignal'] /= scale
        e['rfsignal1']  /= scale
        e['rfsignal2']  /= scale
    return dataset

def maybe_cast_complex64(entry):
    if args.save_complex64:
        entry['mixsignal'] = entry['mixsignal'].astype(np.complex64)
        entry['rfsignal1'] = entry['rfsignal1'].astype(np.complex64)
        entry['rfsignal2'] = entry['rfsignal2'].astype(np.complex64)
    return entry

def build_save_path(mode, extra_tag="", shard_idx=None, N_for_name=None):
    def fmt_range(tag, r, fmt='{:.2f}'):
        return f"{tag}U[{fmt.format(r[0])},{fmt.format(r[1])}]"

    if mode == "train":
        freq_tag = fmt_range('freq', FREQ_RANGE_TRAIN, fmt='{:.0f}')
        phi1_tag = fmt_range('phi1', PHASE1_RANGE, fmt='{:.4f}')
        phi2_tag = fmt_range('phi2', PHASE2_RANGE, fmt='{:.4f}')
        amp_tag  = fmt_range('amp',  AMP_RANGE_TRAIN, fmt='{:.2f}')
        snr_tag  = fmt_range('snr',  SNR_RANGE_TRAIN, fmt='{:.0f}')
        dtype_tag = "_c64" if args.save_complex64 else "_c128"
        base = f"qpsk_train_rand_{freq_tag}_{phi1_tag}_{phi2_tag}_{amp_tag}_{snr_tag}_N{N_for_name}{extra_tag}{dtype_tag}"
        if shard_idx is not None:
            return os.path.join(save_dir, f"{base}_shard{int(shard_idx)}.pth")
        else:
            return os.path.join(save_dir, f"{base}.pth")

    elif mode == "test_eqcfo":
        return os.path.join(
            save_dir,
            f'qpsk_test_eqcfo_grid_F{len(FREQ_GRID)}_P{len(PHASE_GRID)}_A{len(AMP_GRID_EFF)}_D{len(DELAY_GRID_EFF)}_R{args.test_repeats}{extra_tag}.pth'
        )
    elif mode == "test_diffcfo":
        return os.path.join(
            save_dir,
            f'qpsk_test_diffcfo_grid_F1{len(FREQ_GRID)}_F2{len(FREQ_GRID)}_P{len(PHASE_GRID)}_A{len(AMP_GRID_EFF)}_D{len(DELAY_GRID_EFF)}_R{args.test_repeats}{extra_tag}.pth'
        )
    elif mode == "test_all":
        # F1/F2/P1/P2/A/S/R（不含 D）
        return os.path.join(
            save_dir,
            f'qpsk_test_all_grid_F1{len(FREQ1_GRID)}_F2{len(FREQ2_GRID)}_P1{len(PHASE1_GRID)}_P2{len(PHASE2_GRID)}_A{len(AMP_ALL_GRID)}_S{len(SNR_GRID)}_R{args.test_repeats}{extra_tag}.pth'
        )

# ============= 训练生成（多N + 分片） =============
def gen_one_train_dataset(N_total, SHARD_SIZE):
    """生成一个给定规模 N_total 的训练集，并按 shard 增量保存。"""
    num_shards = (N_total + SHARD_SIZE - 1) // SHARD_SIZE
    shard_entries = []
    shard_idx = 1
    saved_paths = []

    def flush_current_shard(entries, shard_idx):
        if not entries:
            return None
        # 分片级别归一化 + 可选 complex64
        entries_norm = energy_normalize_dataset(entries)
        entries_norm = [maybe_cast_complex64(e) for e in entries_norm]
        path = build_save_path(
                "train",
                extra_tag="_varsnr_ampr_phi1phi2_delay0T",
                shard_idx=shard_idx,
                N_for_name=N_total
            )
        torch.save(entries_norm, path)
        print(f"📦 已保存分片 {shard_idx}/{num_shards}: {path} （样本数 {len(entries_norm)}）")
        return path

    for k in range(N_total):
        # -------- 1) 随机比特并 QPSK 映射 --------
        bits1 = np.random.randint(0, 2, bit_len)
        bits2 = np.random.randint(0, 2, bit_len)
        symbols1 = qpsk_mod(bits1)
        symbols2 = qpsk_mod(bits2)

        # -------- 2) 上采样 + 分数符号时延（0~T 区间，采样级） --------
        up_len = len(symbols1) * sps
        symbols_up1 = np.zeros(up_len, dtype=complex)
        symbols_up2 = np.zeros(up_len, dtype=complex)

        # 每路在 [0, sps-1] 之间随机一个采样级时延
        delay_samp1 = np.random.randint(0, sps)  # 对应 τ1 ∈ [0, T)
        delay_samp2 = np.random.randint(0, sps)  # 对应 τ2 ∈ [0, T)

        amplititude_ratio = np.random.uniform(*AMP_RANGE_TRAIN)

        # 注意：这里的步长依旧是 sps，只是起始位置不同
        symbols_up1[delay_samp1::sps] = symbols1
        symbols_up2[delay_samp2::sps] = symbols2 * amplititude_ratio

        # -------- 3) RC 成型滤波 --------
        tx1 = convolve(symbols_up1, rc, mode='same')
        tx2 = convolve(symbols_up2, rc, mode='same')

        # -------- 4) 随机 CFO + 初相位 --------
        freq_offset_k1 = np.random.uniform(*FREQ_RANGE_TRAIN) * np.random.choice([-1, 1])
        freq_offset_k2 = np.random.uniform(*FREQ_RANGE_TRAIN) * np.random.choice([-1, 1])
        init_phase1_k = np.random.uniform(*PHASE1_RANGE)
        init_phase2_k = np.random.uniform(*PHASE2_RANGE)

        t = np.arange(up_len) / fs
        tx1 = tx1 * np.exp(1j * (2 * np.pi * freq_offset_k1 * t + init_phase1_k))
        tx2 = tx2 * np.exp(1j * (2 * np.pi * freq_offset_k2 * t + init_phase2_k))

        # -------- 5) 合路 + AWGN（按全局 SNR） --------
        snr_db_k = np.random.uniform(*SNR_RANGE_TRAIN)
        rx_clean = tx1 + tx2
        rx = awgn_with_seed(rx_clean, snr_db_k, seed=None)

        # -------- 6) 记录样本 --------
        new_entry = {
            'mixsignal': rx,
            'rfsignal1': tx1,
            'rfsignal2': tx2,
            'params': (
                float(snr_db_k), float(amplititude_ratio), sps,
                f'f_off1={float(freq_offset_k1):.2f}Hz',
                f'f_off2={float(freq_offset_k2):.2f}Hz',
                f'phi1={float(init_phase1_k):.4f}rad',
                f'phi2={float(init_phase2_k):.4f}rad',
                f'delay1_samp={int(delay_samp1)}',
                f'delay2_samp={int(delay_samp2)}',
                'QPSK'
            ),
            'bits1': -1,   # train 集中不保存比特（只做盲分离）
            'bits2': -1,
            'origin_len': 1
        }
        shard_entries.append(new_entry)

        if (k + 1) % 1000 == 0 or (k + 1) == N_total:
            print(f"[train-N={N_total}] 进度 {k + 1}/{N_total}")

        if len(shard_entries) >= SHARD_SIZE:
            p = flush_current_shard(shard_entries, shard_idx)
            if p: saved_paths.append(p)
            shard_entries = []
            shard_idx += 1

    if shard_entries:
        p = flush_current_shard(shard_entries, shard_idx)
        if p: saved_paths.append(p)

    print(f"✅ N={N_total} 的训练集分片已全部保存，共 {len(saved_paths)} 片。"
          f"样例路径：{saved_paths[0] if saved_paths else '无'}")

def run_train_multiN():
    N_list = parse_train_sizes(args.train_sizes)
    SHARD_SIZE = max(1, int(args.shard_size))
    print(f"👉 模式: train | N 列表: {N_list} | 分片大小: {SHARD_SIZE} | "
          f"保存dtype: {'complex64' if args.save_complex64 else 'complex128'}")
    for N in N_list:
        gen_one_train_dataset(N_total=N, SHARD_SIZE=SHARD_SIZE)

# ============= 测试集（eqcfo） =============
def run_test_eqcfo():
    print("👉 模式: test_eqcfo")
    combos = list(itertools.product(FREQ_GRID, PHASE_GRID, AMP_GRID_EFF, DELAY_GRID_EFF))
    total = len(combos) * args.test_repeats
    print(f"[test_eqcfo] 固定网格组合数: {len(combos)}，每组重复 {args.test_repeats} 次，共 {total} 块")

    dataset = []
    global_idx = 0
    snr_db = TEST_SNR_DB
    for combo_idx, (freq_offset_k, init_phase_k, amplititude_ratio, delay_sym) in enumerate(combos):
        for r in range(args.test_repeats):
            combo_bytes = f"eq|f{freq_offset_k}|phi0{init_phase_k}|a{amplititude_ratio}|d{delay_sym}|rep{r}".encode()
            seed = int(hashlib.sha256(combo_bytes).hexdigest()[:8], 16)

            rng_bits = np.random.default_rng(seed)
            bits1 = rng_bits.integers(0, 2, bit_len)
            bits2 = rng_bits.integers(0, 2, bit_len)

            symbols1 = qpsk_mod(bits1)
            symbols2 = qpsk_mod(bits2)

            if int(delay_sym) != 0:
                symbols2 = np.roll(symbols2, int(delay_sym))

            up_len = len(symbols1) * sps
            symbols_up1 = np.zeros(up_len, dtype=complex)
            symbols_up2 = np.zeros(up_len, dtype=complex)
            symbols_up1[::sps] = symbols1
            symbols_up2[::sps] = symbols2 * float(amplititude_ratio)

            tx1 = convolve(symbols_up1, rc, mode='same')
            tx2 = convolve(symbols_up2, rc, mode='same')

            t = np.arange(up_len) / fs
            tx1 = tx1 * np.exp(1j * (2 * np.pi * float(freq_offset_k) * t))
            tx2 = tx2 * np.exp(1j * (2 * np.pi * float(freq_offset_k) * t + float(init_phase_k)))

            seed_rx = seed ^ 0xA5A5A5A5
            rx = awgn_with_seed(tx1 + tx2, snr_db, seed_rx)

            new_entry = {
                'mixsignal': rx,
                'rfsignal1': tx1,
                'rfsignal2': tx2,
                'params': (snr_db, float(amplititude_ratio), sps,
                           f'f_off={float(freq_offset_k):.2f}Hz',
                           f'phi0={float(init_phase_k):.4f}rad',
                           f'delay={int(delay_sym)}',
                           f'rep={r}', 'QPSK'),
                'bits1': bits1,
                'bits2': bits2,
                'origin_len': 1
            }
            dataset.append(new_entry)

            global_idx += 1
            if global_idx % 200 == 0:
                print(f"[test_eqcfo] 已生成 {global_idx}/{total} 块")

    save_path = build_save_path("test_eqcfo", extra_tag="")
    dataset_normed = energy_normalize_dataset(dataset)
    dataset_normed = [maybe_cast_complex64(e) for e in dataset_normed]
    torch.save(dataset_normed, save_path)
    print(f"✅ 数据集已保存至: {save_path}")

# ============= 测试集（diffcfo） =============
def run_test_diffcfo():
    print("👉 模式: test_diffcfo")
    FREQ1 = FREQ_GRID
    FREQ2 = FREQ_GRID
    combos = list(itertools.product(FREQ1, FREQ2, PHASE_GRID, AMP_GRID_EFF, DELAY_GRID_EFF))
    total = len(combos) * args.test_repeats
    print(f"[test_diffcfo] 固定网格组合数: {len(combos)}，每组重复 {args.test_repeats} 次，共 {total} 块")

    dataset = []
    global_idx = 0
    snr_db = TEST_SNR_DB
    for combo_idx, (freq_off1, freq_off2, init_phase_k, amplititude_ratio, delay_sym) in enumerate(combos):
        for r in range(args.test_repeats):
            combo_bytes = f"diff|f1{freq_off1}|f2{freq_off2}|phi0{init_phase_k}|a{amplititude_ratio}|d{delay_sym}|rep{r}".encode()
            seed = int(hashlib.sha256(combo_bytes).hexdigest()[:8], 16)

            rng_bits = np.random.default_rng(seed)
            bits1 = rng_bits.integers(0, 2, bit_len)
            bits2 = rng_bits.integers(0, 2, bit_len)

            symbols1 = qpsk_mod(bits1)
            symbols2 = qpsk_mod(bits2)

            if int(delay_sym) != 0:
                symbols2 = np.roll(symbols2, int(delay_sym))

            up_len = len(symbols1) * sps
            symbols_up1 = np.zeros(up_len, dtype=complex)
            symbols_up2 = np.zeros(up_len, dtype=complex)
            symbols_up1[::sps] = symbols1
            symbols_up2[::sps] = symbols2 * float(amplititude_ratio)

            tx1 = convolve(symbols_up1, rc, mode='same')
            tx2 = convolve(symbols_up2, rc, mode='same')

            t = np.arange(up_len) / fs
            tx1 = tx1 * np.exp(1j * (2 * np.pi * float(freq_off1) * t))
            tx2 = tx2 * np.exp(1j * (2 * np.pi * float(freq_off2) * t + float(init_phase_k)))

            seed_rx = seed ^ 0x5A5A5A5A
            rx = awgn_with_seed(tx1 + tx2, snr_db, seed_rx)

            new_entry = {
                'mixsignal': rx,
                'rfsignal1': tx1,
                'rfsignal2': tx2,
                'params': (
                    snr_db, float(amplititude_ratio), sps,
                    f'f_off1={float(freq_off1):.2f}Hz',
                    f'f_off2={float(freq_off2):.2f}Hz',
                    f'phi0={float(init_phase_k):.4f}rad',
                    f'delay={int(delay_sym)}',
                    f'rep={r}', 'QPSK'
                ),
                'bits1': bits1,
                'bits2': bits2,
                'origin_len': 1
            }
            dataset.append(new_entry)

            global_idx += 1
            if global_idx % 200 == 0:
                print(f"[test_diffcfo] 已生成 {global_idx}/{total} 块")

    save_path = build_save_path("test_diffcfo", extra_tag="")
    dataset_normed = energy_normalize_dataset(dataset)
    dataset_normed = [maybe_cast_complex64(e) for e in dataset_normed]
    torch.save(dataset_normed, save_path)
    print(f"✅ 数据集已保存至: {save_path}")

def run_test_all():
    print("👉 模式: test_all （两路 CFO/相位独立 + 幅度比 + SNR）")
    # 原来的六维网格：f1, f2, phi1, phi2, amp, snr
    combos = list(itertools.product(FREQ1_GRID, FREQ2_GRID,
                                    PHASE1_GRID, PHASE2_GRID,
                                    AMP_ALL_GRID, SNR_GRID))
    total = len(combos) * args.test_repeats
    print(f"[test_all] 无时延版本：网格组合数 {len(combos)}，每组重复 {args.test_repeats} 次，共 {total} 块")

    dataset = []          # 原始：无分数采样级时延
    dataset_delay = []    # 新增：两路均有 0,2,4,6 采样点的分数符号时延

    global_idx = 0
    for combo_idx, (f1, f2, phi1, phi2, amp, snr_val) in enumerate(combos):
        for r in range(args.test_repeats):
            # ----- 公共组合 + 固定 bits 随机种子 -----
            combo_bytes = f"all|f1{f1}|f2{f2}|phi1{phi1}|phi2{phi2}|a{amp}|snr{snr_val}|rep{r}".encode()
            seed = int(hashlib.sha256(combo_bytes).hexdigest()[:8], 16)

            rng_bits = np.random.default_rng(seed)
            bits1 = rng_bits.integers(0, 2, bit_len)
            bits2 = rng_bits.integers(0, 2, bit_len)

            symbols1 = qpsk_mod(bits1)
            symbols2 = qpsk_mod(bits2)

            up_len = len(symbols1) * sps
            t = np.arange(up_len) / fs

            # ========== 1) 原始版本：无分数采样时延 ==========
            symbols_up1 = np.zeros(up_len, dtype=complex)
            symbols_up2 = np.zeros(up_len, dtype=complex)
            symbols_up1[::sps] = symbols1
            symbols_up2[::sps] = symbols2 * float(amp)

            tx1 = convolve(symbols_up1, rc, mode='same')
            tx2 = convolve(symbols_up2, rc, mode='same')

            tx1 = tx1 * np.exp(1j * (2 * np.pi * float(f1) * t + float(phi1)))
            tx2 = tx2 * np.exp(1j * (2 * np.pi * float(f2) * t + float(phi2)))

            seed_rx = seed ^ 0x3C3C3C3C
            rx = awgn_with_seed(tx1 + tx2, float(snr_val), seed_rx)

            new_entry = {
                'mixsignal': rx,
                'rfsignal1': tx1,
                'rfsignal2': tx2,
                'params': (
                    float(snr_val), float(amp), sps,
                    f'f_off1={float(f1):.2f}Hz',
                    f'f_off2={float(f2):.2f}Hz',
                    f'phi1={float(phi1):.4f}rad',
                    f'phi2={float(phi2):.4f}rad',
                    f'rep={r}', 'QPSK'
                ),
                'bits1': bits1,
                'bits2': bits2,
                'origin_len': 1
            }
            dataset.append(new_entry)

            # ========== 2) 带 delay 版本：两路 delay ∈ {0,2,4,6} 采样点 ==========
            # 为了“均匀采样”，这里采用笛卡尔积网格而不是随机；
            # 但为了不把组合数放大 16 倍，我们这里在每个 combo 上只取一个 (d1,d2)，
            # 通过 seed 派生一个“确定但均匀”的选择。
            rng_delay = np.random.default_rng(seed ^ 0xD1E10FF)
            # 在 DELAY_SAMP_GRID 中均匀选一个索引
            idx1 = int(rng_delay.integers(0, len(DELAY_SAMP_GRID)))
            idx2 = int(rng_delay.integers(0, len(DELAY_SAMP_GRID)))
            delay_samp1 = int(DELAY_SAMP_GRID[idx1])
            delay_samp2 = int(DELAY_SAMP_GRID[idx2])

            symbols_up1_d = np.zeros(up_len, dtype=complex)
            symbols_up2_d = np.zeros(up_len, dtype=complex)
            symbols_up1_d[delay_samp1::sps] = symbols1
            symbols_up2_d[delay_samp2::sps] = symbols2 * float(amp)

            tx1_d = convolve(symbols_up1_d, rc, mode='same')
            tx2_d = convolve(symbols_up2_d, rc, mode='same')

            tx1_d = tx1_d * np.exp(1j * (2 * np.pi * float(f1) * t + float(phi1)))
            tx2_d = tx2_d * np.exp(1j * (2 * np.pi * float(f2) * t + float(phi2)))

            seed_rx_d = seed_rx ^ 0x11111111
            rx_d = awgn_with_seed(tx1_d + tx2_d, float(snr_val), seed_rx_d)

            new_entry_d = {
                'mixsignal': rx_d,
                'rfsignal1': tx1_d,
                'rfsignal2': tx2_d,
                'params': (
                    float(snr_val), float(amp), sps,
                    f'f_off1={float(f1):.2f}Hz',
                    f'f_off2={float(f2):.2f}Hz',
                    f'phi1={float(phi1):.4f}rad',
                    f'phi2={float(phi2):.4f}rad',
                    f'delay1_samp={delay_samp1}',
                    f'delay2_samp={delay_samp2}',
                    f'rep={r}', 'QPSK'
                ),
                'bits1': bits1,
                'bits2': bits2,
                'origin_len': 1
            }
            dataset_delay.append(new_entry_d)

            global_idx += 1
            if global_idx % 200 == 0:
                print(f"[test_all] 已生成 {global_idx}/{total} 组（每组包含：无delay + 有delay 各1块）")

    # # ----- 保存原始版本（保持不变，兼容以前脚本） -----
    # save_path = build_save_path("test_all", extra_tag="")
    # dataset_normed = energy_normalize_dataset(dataset)
    # dataset_normed = [maybe_cast_complex64(e) for e in dataset_normed]
    # torch.save(dataset_normed, save_path)
    # print(f"✅ 无分数时延版本已保存至: {save_path}")

    # ----- 保存带 delay 版本：两路 delay ∈ {0,2,4,6} 采样点 -----
    save_path_d = build_save_path("test_all", extra_tag="_delay4grid")
    dataset_d_normed = energy_normalize_dataset(dataset_delay)
    dataset_d_normed = [maybe_cast_complex64(e) for e in dataset_d_normed]
    torch.save(dataset_d_normed, save_path_d)
    print(f"✅ 带 delay∈{{0,2,4,6}} 采样点版本已保存至: {save_path_d}")

# ============= 主入口 =============
if __name__ == "__main__":
    if args.mode == "train":
        run_train_multiN()
    elif args.mode == "test_eqcfo":
        run_test_eqcfo()
    elif args.mode == "test_diffcfo":
        run_test_diffcfo()
    elif args.mode == "test_all":
        run_test_all()
