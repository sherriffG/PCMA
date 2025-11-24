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
    description="多调制数据集生成器：\n"
                "  train: 多N规模随机采样 + 分片保存（两路调制方式随机）；\n"
                "  test_all_qpsk: 原 test_all 的 QPSK 专用版本（含 delay 网格）；\n"
                "  test_all: 预留的新全集模式（暂未实现）。"
)
parser.add_argument("--mode", type=str, default="train",
                    choices=["train", "test_all_qpsk", "test_all"],
                    help="选择生成训练集或测试集")
parser.add_argument("--train_profile", type=str, default="robust",
                    choices=["aligned", "robust"],
                    help="train 模式下的超参数分布："
                         "aligned=贴近当前采集参数；robust=宽范围泛化（SNR U[8,22] 等）")
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
sps = 8   # 仿真统一使用 sps=8，采集的 16 可下采样到 8
fs = 12e6
num_taps = 64
input_len = 3072  # 每块样本点数（两路信号同长度）
assert input_len % sps == 0
num_syms = input_len // sps  # 每路符号数

# 各调制方式每符号 bit 数
BITS_PER_SYMBOL = {
    "QPSK": 2,
    "8PSK": 3,
    "16QAM": 4,
}

# 可选调制集合（train 时每路从这里随机选一种）
MOD_LIST = ["QPSK", "8PSK", "16QAM"]


def get_bit_len(modulation: str) -> int:
    """给定调制方式，返回每路比特长度。"""
    return num_syms * BITS_PER_SYMBOL[modulation.upper()]


# ============= 训练用随机区间（profile: aligned / robust） =============
def get_train_hyper_ranges(profile: str):
    """
    返回 (freq_range, phase1_range, phase2_range, amp_range, snr_range)
    freq_range: (0, 200) 表示绝对值范围，符号再随机 ±1
    这里不再依赖具体调制方式，因为两路调制本来就是随机混合。
    """
    profile = profile.lower()

    if profile == "robust":
        # 你的设定：统一大范围
        snr_range = (8.0, 22.0)      # dB
        freq_range = (0.0, 200.0)    # Hz（之后随机乘 ±1）
        amp_range = (0.2, 0.9)       # a = |s2|/|s1|
    else:
        # aligned：贴近当前采集数据的窄范围（所有调制共享一个“总体”区间）
        #   - SNR 大致在 12~18 dB
        #   - CFO 大致在 30~130 Hz，覆盖 53/107 附近
        snr_range = (12.0, 18.0)
        freq_range = (30.0, 130.0)
        amp_range = (0.4, 0.9)

    phase1_range = (0.0, 2 * np.pi)
    phase2_range = (0.0, 2 * np.pi)
    return freq_range, phase1_range, phase2_range, amp_range, snr_range


# ============= 测试集网格（QPSK 专用 test_all_qpsk 使用） =============
FREQ_GRID = np.linspace(0, 200, 10)                 # Hz
PHASE1_GRID = np.linspace(0.0, 2 * np.pi, 8, endpoint=False)
PHASE2_GRID = np.linspace(0.0, 2 * np.pi, 8, endpoint=False)
FREQ1_GRID = FREQ_GRID
FREQ2_GRID = FREQ_GRID
AMP_ALL_GRID = np.round(np.linspace(0.30, 0.90, 5), 2)
SNR_GRID = np.array([12.0, 18.0, 24.0, 30.0])
# sps = 8 时，对应 0, T/4, T/2, 3T/4 -> 0, 2, 4, 6 samples
DELAY_SAMP_GRID = np.array([0, 2, 4, 6], dtype=int)

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


# ============= 各调制方式的映射 =============
def qpsk_mod(bits: np.ndarray) -> np.ndarray:
    """QPSK Gray 映射，与原实现兼容。"""
    symbols = []
    for i in range(0, len(bits), 2):
        b1, b2 = bits[i], bits[i + 1]
        if b1 == 0 and b2 == 0:
            symbols.append(1 + 1j)
        elif b1 == 0 and b2 == 1:
            symbols.append(-1 + 1j)
        elif b1 == 1 and b2 == 0:
            symbols.append(1 - 1j)
        else:
            symbols.append(-1 - 1j)
    return np.array(symbols, dtype=complex) / np.sqrt(2)


def psk8_mod(bits: np.ndarray) -> np.ndarray:
    """
    8PSK 映射：
    - 每 3bit -> 一个符号，采用自然编码：k = b2*4 + b1*2 + b0
    - 符号 = exp(j*(2πk/8))，整体为 8 点均匀分布在单位圆上。
    """
    assert len(bits) % 3 == 0
    bits = bits.reshape(-1, 3)
    idx = bits[:, 0] * 4 + bits[:, 1] * 2 + bits[:, 2]
    phase = 2 * np.pi * idx / 8.0
    symbols = np.exp(1j * phase)
    return symbols.astype(complex)


def qam16_mod(bits: np.ndarray) -> np.ndarray:
    """
    16QAM 映射：
    - 每 4bit -> 2bit(I) + 2bit(Q)
    - 采用 Gray 编码 + 方阵 [-3,-1,1,3]，然后整体归一化到平均能量=1。
    """
    assert len(bits) % 4 == 0
    bits = bits.reshape(-1, 4)
    # Gray 2bit -> level 映射
    def gray2level(b0, b1):
        if b0 == 0 and b1 == 0:
            return -3
        elif b0 == 0 and b1 == 1:
            return -1
        elif b0 == 1 and b1 == 1:
            return 1
        else:  # b0==1 and b1==0
            return 3

    I = np.array([gray2level(b[0], b[1]) for b in bits], dtype=float)
    Q = np.array([gray2level(b[2], b[3]) for b in bits], dtype=float)
    symbols = I + 1j * Q
    symbols = symbols / np.sqrt(10.0)  # 平均能量归一化
    return symbols.astype(complex)


def modulate(bits: np.ndarray, modulation: str) -> np.ndarray:
    """统一入口：根据 modulation 调用对应的调制函数。"""
    modulation = modulation.upper()
    if modulation == "QPSK":
        return qpsk_mod(bits)
    elif modulation == "8PSK":
        return psk8_mod(bits)
    elif modulation == "16QAM":
        return qam16_mod(bits)
    else:
        raise ValueError(f"不支持的调制方式: {modulation}")


def rc_filter(beta, sps, num_taps):
    t = np.arange(-num_taps // 2, num_taps // 2) / sps
    with np.errstate(divide='ignore', invalid='ignore'):
        h = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t) ** 2)
        h[np.isnan(h)] = 1.0 - beta + (4 * beta / np.pi)
    h = h / np.sqrt(np.sum(h ** 2))
    return h


rc = rc_filter(beta, sps, num_taps)


def awgn_with_seed(signal, snr_db, seed=None):
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    noise = np.sqrt(noise_power / 2) * (
            rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal))
    )
    return signal + noise


def energy_normalize_dataset(dataset):
    energies = [np.mean(np.abs(e['mixsignal']) ** 2) for e in dataset]
    mean_e = np.mean(energies) if energies else 1.0
    scale = np.sqrt(mean_e)
    for e in dataset:
        e['mixsignal'] /= scale
        e['rfsignal1'] /= scale
        e['rfsignal2'] /= scale
    return dataset


def maybe_cast_complex64(entry):
    if args.save_complex64:
        entry['mixsignal'] = entry['mixsignal'].astype(np.complex64)
        entry['rfsignal1'] = entry['rfsignal1'].astype(np.complex64)
        entry['rfsignal2'] = entry['rfsignal2'].astype(np.complex64)
    return entry


def build_save_path(mode, extra_tag="", shard_idx=None, N_for_name=None,
                    train_profile: str = "robust"):
    """
    构造保存路径：
    - train: mixedmods_train_<profile>_rand_..._N..._shardXX.pth
    - test_all_qpsk: 仍沿用 qpsk_test_all_grid_... 的命名（便于兼容）
    - test_all: 预留
    """
    profile_tag = train_profile.lower()

    def fmt_range(tag, r, fmt='{:.2f}'):
        return f"{tag}U[{fmt.format(r[0])},{fmt.format(r[1])}]"

    if mode == "train":
        freq_range, phase1_range, phase2_range, amp_range, snr_range = get_train_hyper_ranges(
            train_profile
        )
        freq_tag = fmt_range('freq', freq_range, fmt='{:.0f}')
        phi1_tag = fmt_range('phi1', phase1_range, fmt='{:.4f}')
        phi2_tag = fmt_range('phi2', phase2_range, fmt='{:.4f}')
        amp_tag = fmt_range('amp', amp_range, fmt='{:.2f}')
        snr_tag = fmt_range('snr', snr_range, fmt='{:.0f}')
        dtype_tag = "_c64" if args.save_complex64 else "_c128"
        base = f"mixedmods_train_{profile_tag}_rand_{freq_tag}_{phi1_tag}_{phi2_tag}_{amp_tag}_{snr_tag}_N{N_for_name}{extra_tag}{dtype_tag}"
        if shard_idx is not None:
            return os.path.join(save_dir, f"{base}_shard{int(shard_idx)}.pth")
        else:
            return os.path.join(save_dir, f"{base}.pth")

    if mode == "test_all_qpsk":
        # F1/F2/P1/P2/A/S/R（不含 D）
        return os.path.join(
            save_dir,
            f'qpsk_test_all_grid_F1{len(FREQ1_GRID)}_F2{len(FREQ2_GRID)}_P1{len(PHASE1_GRID)}_P2{len(PHASE2_GRID)}_A{len(AMP_ALL_GRID)}_S{len(SNR_GRID)}_R{args.test_repeats}{extra_tag}.pth'
        )
    elif mode == "test_all":
        # 预留：将来可以为新的全集模式设计命名
        return os.path.join(save_dir, f"mixedmods_test_all_placeholder{extra_tag}.pth")
    else:
        raise ValueError(f"未知 mode: {mode}")


# ============= 训练生成（多N + 分片） =============
def gen_one_train_dataset(N_total, SHARD_SIZE, train_profile: str):
    """
    生成一个给定规模 N_total 的训练集，并按 shard 增量保存。

    特点：
      - 每个样本的两路调制方式都是随机的：
          mod1, mod2 分别从 MOD_LIST 中独立采样；
      - 超参数分布由 train_profile 决定（aligned/robust）。
    """
    train_profile = train_profile.lower()
    num_shards = (N_total + SHARD_SIZE - 1) // SHARD_SIZE
    shard_entries = []
    shard_idx = 1
    saved_paths = []

    freq_range, phase1_range, phase2_range, amp_range, snr_range = get_train_hyper_ranges(
        train_profile
    )

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
            N_for_name=N_total,
            train_profile=train_profile,
        )
        torch.save(entries_norm, path)
        print(f"📦 已保存分片 {shard_idx}/{num_shards}: {path} （样本数 {len(entries_norm)}）")
        return path

    for k in range(N_total):
        # -------- 0) 为每路随机选择调制方式 --------
        mod1 = np.random.choice(MOD_LIST)
        mod2 = np.random.choice(MOD_LIST)

        bit_len1 = get_bit_len(mod1)
        bit_len2 = get_bit_len(mod2)

        # -------- 1) 随机比特并调制映射 --------
        bits1 = np.random.randint(0, 2, bit_len1, dtype=np.int8)
        bits2 = np.random.randint(0, 2, bit_len2, dtype=np.int8)
        symbols1 = modulate(bits1, mod1)
        symbols2 = modulate(bits2, mod2)

        # 确认符号数均为 num_syms
        assert len(symbols1) == num_syms
        assert len(symbols2) == num_syms

        # -------- 2) 上采样 + 分数符号时延（0~T 区间，采样级） --------
        up_len = num_syms * sps
        symbols_up1 = np.zeros(up_len, dtype=complex)
        symbols_up2 = np.zeros(up_len, dtype=complex)

        # 每路在 [0, sps-1] 之间随机一个采样级时延
        delay_samp1 = np.random.randint(0, sps)  # 对应 τ1 ∈ [0, T)
        delay_samp2 = np.random.randint(0, sps)  # 对应 τ2 ∈ [0, T)

        amplitude_ratio = np.random.uniform(*amp_range)

        symbols_up1[delay_samp1::sps] = symbols1
        symbols_up2[delay_samp2::sps] = symbols2 * amplitude_ratio

        # -------- 3) RC 成型滤波 --------
        tx1 = convolve(symbols_up1, rc, mode='same')
        tx2 = convolve(symbols_up2, rc, mode='same')

        # -------- 4) 随机 CFO + 初相位 --------
        # 先在 [0, max] 中取绝对值，再随机乘 ±1
        freq_offset_k1 = np.random.uniform(*freq_range) * np.random.choice([-1, 1])
        freq_offset_k2 = np.random.uniform(*freq_range) * np.random.choice([-1, 1])
        init_phase1_k = np.random.uniform(*phase1_range)
        init_phase2_k = np.random.uniform(*phase2_range)

        t = np.arange(up_len) / fs
        tx1 = tx1 * np.exp(1j * (2 * np.pi * freq_offset_k1 * t + init_phase1_k))
        tx2 = tx2 * np.exp(1j * (2 * np.pi * freq_offset_k2 * t + init_phase2_k))

        # -------- 5) 合路 + AWGN（按全局 SNR） --------
        snr_db_k = np.random.uniform(*snr_range)
        rx_clean = tx1 + tx2
        rx = awgn_with_seed(rx_clean, snr_db_k, seed=None)

        # -------- 6) 记录样本 --------
        new_entry = {
            'mixsignal': rx,
            'rfsignal1': tx1,
            'rfsignal2': tx2,
            'params': (
                float(snr_db_k), float(amplitude_ratio), sps,
                f'f_off1={float(freq_offset_k1):.2f}Hz',
                f'f_off2={float(freq_offset_k2):.2f}Hz',
                f'phi1={float(init_phase1_k):.4f}rad',
                f'phi2={float(init_phase2_k):.4f}rad',
                f'delay1_samp={int(delay_samp1)}',
                f'delay2_samp={int(delay_samp2)}',
                f'mod1={mod1}',
                f'mod2={mod2}',
            ),
            # train 集默认不保存比特（只做盲分离）；如需保存可以改这里
            'bits1': -1,
            'bits2': -1,
            'origin_len': 1
        }
        shard_entries.append(new_entry)

        if (k + 1) % 1000 == 0 or (k + 1) == N_total:
            print(f"[train-mixedmods-N={N_total}] 进度 {k + 1}/{N_total}")

        if len(shard_entries) >= SHARD_SIZE:
            p = flush_current_shard(shard_entries, shard_idx)
            if p:
                saved_paths.append(p)
            shard_entries = []
            shard_idx += 1

    if shard_entries:
        p = flush_current_shard(shard_entries, shard_idx)
        if p:
            saved_paths.append(p)

    print(f"✅ mixedmods-N={N_total} 的训练集分片已全部保存，共 {len(saved_paths)} 片。"
          f"样例路径：{saved_paths[0] if saved_paths else '无'}")


def run_train_multiN():
    N_list = parse_train_sizes(args.train_sizes)
    SHARD_SIZE = max(1, int(args.shard_size))
    print(f"👉 模式: train | 调制: 每路随机于 {MOD_LIST} | profile: {args.train_profile} | "
          f"N 列表: {N_list} | 分片大小: {SHARD_SIZE} | "
          f"保存dtype: {'complex64' if args.save_complex64 else 'complex128'}")

    for N in N_list:
        gen_one_train_dataset(N_total=N,
                              SHARD_SIZE=SHARD_SIZE,
                              train_profile=args.train_profile)


# ============= 测试集：test_all_qpsk（原 test_all） =============
def run_test_all_qpsk():
    """
    原来的 test_all 逻辑：两路均为 QPSK，f1/f2/phi1/phi2/amp/SNR 走网格，
    另外为每个组合生成一个 delay 版本（delay1/2 ∈ {0,2,4,6} 采样点）。
    """
    modulation = "QPSK"
    bit_len = get_bit_len(modulation)
    print(f"👉 模式: test_all_qpsk （两路 QPSK，CFO/相位/幅度/SNR 网格 + delay）")

    combos = list(itertools.product(FREQ1_GRID, FREQ2_GRID,
                                    PHASE1_GRID, PHASE2_GRID,
                                    AMP_ALL_GRID, SNR_GRID))
    total = len(combos) * args.test_repeats
    print(f"[test_all_qpsk] 网格组合数 {len(combos)}，每组重复 {args.test_repeats} 次，共 {total} 组（每组含无delay+有delay 各1块）")

    dataset = []          # 原始：无分数采样级时延
    dataset_delay = []    # 两路 delay ∈ {0,2,4,6} 采样点

    global_idx = 0
    for combo_idx, (f1, f2, phi1, phi2, amp, snr_val) in enumerate(combos):
        for r in range(args.test_repeats):
            combo_bytes = f"allqpsk|f1{f1}|f2{f2}|phi1{phi1}|phi2{phi2}|a{amp}|snr{snr_val}|rep{r}".encode()
            seed = int(hashlib.sha256(combo_bytes).hexdigest()[:8], 16)

            rng_bits = np.random.default_rng(seed)
            bits1 = rng_bits.integers(0, 2, bit_len, dtype=np.int8)
            bits2 = rng_bits.integers(0, 2, bit_len, dtype=np.int8)

            symbols1 = modulate(bits1, modulation)
            symbols2 = modulate(bits2, modulation)

            up_len = len(symbols1) * sps
            t = np.arange(up_len) / fs

            # ========== 1) 无分数采样时延 ==========
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
                    f'rep={r}',
                    modulation
                ),
                'bits1': bits1,
                'bits2': bits2,
                'origin_len': 1
            }
            dataset.append(new_entry)

            # ========== 2) 带 delay 版本 ==========
            rng_delay = np.random.default_rng(seed ^ 0xD1E10FF)
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
                    f'rep={r}',
                    modulation
                ),
                'bits1': bits1,
                'bits2': bits2,
                'origin_len': 1
            }
            dataset_delay.append(new_entry_d)

            global_idx += 1
            if global_idx % 200 == 0:
                print(f"[test_all_qpsk] 已生成 {global_idx}/{total} 组（每组：无delay + 有delay 各1块）")

    # 只保存带 delay 的版本（和你现在用的 test_all_delay4grid 一致）
    save_path_d = build_save_path("test_all_qpsk", extra_tag="_delay4grid")
    dataset_d_normed = energy_normalize_dataset(dataset_delay)
    dataset_d_normed = [maybe_cast_complex64(e) for e in dataset_d_normed]
    torch.save(dataset_d_normed, save_path_d)
    print(f"✅ QPSK + delay 版本已保存至: {save_path_d}")


# ============= 预留：新的 test_all（暂不实现逻辑） =============
def run_test_all():
    """
    预留的新 test_all 模式：
      - 将来可以支持两路调制也随机、多调制混合的网格评测；
      - 目前只做占位，不生成数据。
    """
    print("👉 模式: test_all（预留）")
    raise NotImplementedError("test_all 模式尚未实现，请先使用 train 或 test_all_qpsk。")


# ============= 主入口 =============
if __name__ == "__main__":
    if args.mode == "train":
        run_train_multiN()
    elif args.mode == "test_all_qpsk":
        run_test_all_qpsk()
    elif args.mode == "test_all":
        run_test_all()
