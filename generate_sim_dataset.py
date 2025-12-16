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
    description=(
        "多调制数据集生成器：\n"
        "  train           : 多N规模随机采样 + 分片保存（两路调制方式随机）；\n"
        "  test_all_qpsk   : 原 test_all 的 QPSK 专用版本（含 CFO/相位/幅度/SNR 网格 + delay）；\n"
        "  test_all_8psk   : test_all_qpsk 的 8PSK 专用版本（参数网格完全一致，仅调制改为 8PSK）；\n"
        "  test_snr_amp    : 扫 SNR × AMP，四种调制组合（CFO=0, 无 delay）；\n"
        "  test_cfo_phase  : 扫 ΔCFO × Δphi，四种调制组合（固定 SNR/AMP, 无 delay）；\n"
        "  test_delay      : 扫 delay_diff，四种调制组合（固定 SNR/AMP, CFO=0）。"
    )
)

parser.add_argument(
    "--mode",
    type=str,
    default="train",
    choices=[
        "train",
        "test_all_qpsk", "test_all_8psk",
        "test_snr_amp", "test_cfo_phase", "test_delay",
        "test_snr_amp_8psk", "test_cfo_phase_8psk", "test_delay_8psk",
        "test_comparison",
    ],
    help="选择生成模式"
)

parser.add_argument(
    "--train_profile",
    type=str,
    default="robust",
    choices=["aligned", "robust"],
    help="train 模式下的超参数分布：aligned=贴近当前采集参数；robust=宽范围泛化（SNR U[8,22] 等）"
)
parser.add_argument("--test_repeats", type=int, default=5,
                    help="测试集中每个网格点重复的样本数（至少1个）")

parser.add_argument(
    "--auto_test_repeats",
    action="store_true",
    help=(
        "自动根据目标 BER 估算 test_repeats（按每个网格点的总比特数）"
        "；对 test_*_8psk / test_snr_amp / test_cfo_phase / test_delay 有效。"
    ),
)

parser.add_argument(
    "--target_ber",
    type=float,
    default=1e-4,
    help="用于 --auto_test_repeats 的目标 BER（默认 1e-4）",
)

parser.add_argument(
    "--min_expected_errors",
    type=int,
    default=20,
    help="用于 --auto_test_repeats：每个网格点期望的最少错误数（默认 20 => 总比特数≈20/BER）",
)

parser.add_argument("--shard_size", type=int, default=10000,
                    help="train 模式下每个分片包含的样本数（默认 10000）")

parser.add_argument(
    "--train_sizes",
    type=str,
    default="auto",
    help='仅用于 --mode=train。可选："auto"（默认，使用预设列表），'
         '或逗号分隔的数字/带k：如 "5k,10k,50k" 或 "5000,10000,50000"'
)

parser.add_argument(
    "--save_complex64",
    action="store_true",
    help="保存前将 complex 数据转换为 complex64（节省约一半空间）"
)

parser.add_argument(
    "--modulation_list",
    type=str,
    default=None,
    help='仅用于 --mode=train。指定训练集使用的调制方式，逗号分隔，如 "8PSK,16QAM" 或 "8PSK"。'
         '默认 None 表示使用所有调制方式 (QPSK,8PSK,16QAM)。'
         '有效值: QPSK, 8PSK, 16QAM'
)

args = parser.parse_args()

# ============= 通用参数 =============
beta = 0.33
sps = 8   # 仿真统一使用 sps=8
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

# 可选调制集合（train 与 test_* 中 mod1/mod2 都会用到）
MOD_LIST = ["QPSK", "8PSK", "16QAM"]


def get_bit_len(modulation: str) -> int:
    """给定调制方式，返回每路比特长度。"""
    return num_syms * BITS_PER_SYMBOL[modulation.upper()]


# ============= train 用随机区间（profile: aligned / robust） =============
def get_train_hyper_ranges(profile: str):
    """
    返回 (freq_range, phase1_range, phase2_range, amp_range, snr_range)
    freq_range: (0, 200) 表示绝对值范围，符号再随机 ±1
    """
    profile = profile.lower()

    if profile == "robust":
        # 你的设定：统一大范围
        # 原始 SNR 范围为 (8.0, 22.0) dB，这里根据需求提升到 (14.0, 20.0) dB
        snr_range = (14.0, 20.0)     # dB
        freq_range = (0.0, 200.0)    # Hz（之后随机乘 ±1）
        amp_range = (0.2, 0.9)       # a = |s2|/|s1|
    else:
        # aligned：贴近当前采集数据的窄范围（所有调制共享一个“总体”区间）
        #   - SNR 大致在 12~18 dB
        #   - CFO 大致在 30~130 Hz，覆盖 53/107 附近
        snr_range = (12.0, 18.0)
        freq_range = (0, 130.0)
        amp_range = (0.3, 0.9)

    phase1_range = (0.0, 2 * np.pi)
    phase2_range = (0.0, 2 * np.pi)
    return freq_range, phase1_range, phase2_range, amp_range, snr_range


# ============= 测试集用网格 =============

# 1) QPSK 专用 test_all_qpsk（原来的）
FREQ_GRID = np.linspace(0, 200, 10)                 # Hz
PHASE1_GRID = np.linspace(0.0, 2 * np.pi, 8, endpoint=False)
PHASE2_GRID = np.linspace(0.0, 2 * np.pi, 8, endpoint=False)
FREQ1_GRID = FREQ_GRID
FREQ2_GRID = FREQ_GRID
AMP_ALL_GRID_QPSK = np.round(np.linspace(0.30, 0.90, 5), 2)
SNR_GRID_QPSK = np.array([12.0, 18.0, 24.0, 30.0])
# sps = 8 时，对应 0, T/4, T/2, 3T/4 -> 0, 2, 4, 6 samples
DELAY_SAMP_GRID = np.array([0, 2, 4, 6], dtype=int)

# 2) test_snr_amp：只扫 SNR 和 AMP
SNR_GRID_SNR_AMP = np.array([8., 10., 12., 14., 16., 18., 20., 22.])
AMP_GRID_SNR_AMP = np.array([0.3, 0.5, 0.7, 0.9])

# 3) test_cfo_phase：扫 ΔCFO 和 Δphi（八个经典点）
CFO_GRID_CFO_PHASE = np.array([-200., -150., -100., -50., 0., 50., 100., 150., 200.])
PHASE_DIFF_GRID = np.array([
    0.0,
    np.pi / 2,
    np.pi,
    3 * np.pi / 2,
    np.pi / 4,
    3 * np.pi / 4,
    5 * np.pi / 4,
    7 * np.pi / 4,
])

# 4) test_delay：扫 delay_diff（采样级）
DELAY_DIFF_SAMP_GRID = DELAY_SAMP_GRID.copy()  # {0,2,4,6}

# 5) test_comparison：轻量级QPSK测试集，用于模型对比
SNR_GRID_COMPARISON = np.array([10., 12., 14., 16., 18., 20., 22.])  # 7个SNR点
AMP_GRID_COMPARISON = np.array([0.3, 0.5, 0.7, 0.9])  # 4个AMP点

# ============= 数据集保存目录 =============
# 原始路径：'/nas/datasets/yixin/PCMA/sim_data'
# 根据当前需求，将新生成的数据（尤其是 8PSK 相关训练集）保存到单独目录，便于管理
save_dir = '/nas/datasets/yixin/PCMA/8PSK'
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


def parse_modulation_list(s: str):
    """
    解析 --modulation_list 参数。
    支持 '8PSK,16QAM' 或 '8PSK' 等格式。
    返回验证后的调制方式列表。
    """
    if s is None or s.strip() == "":
        return MOD_LIST  # 默认使用所有调制方式
    
    parts = [p.strip().upper() for p in s.split(",") if p.strip()]
    valid_mods = []
    for mod in parts:
        if mod not in MOD_LIST:
            raise ValueError(f"不支持的调制方式: {mod}。有效值: {MOD_LIST}")
        valid_mods.append(mod)
    
    if not valid_mods:
        raise ValueError(f"至少需要指定一个调制方式。有效值: {MOD_LIST}")
    
    # 去重并保持顺序
    seen = set()
    result = []
    for mod in valid_mods:
        if mod not in seen:
            seen.add(mod)
            result.append(mod)
    
    return result


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
                    train_profile: str = "robust", mod_list=None):
    """
    构造保存路径：
    - train          : mixedmods_train_<profile>_rand_..._N..._shardXX.pth
    - test_all_qpsk  : qpsk_test_all_grid_F1.._F2.._P1.. 等命名（兼容旧代码）
    - test_snr_amp   : mixedmods_test_snr-amp_...
    - test_cfo_phase : mixedmods_test_cfo-phase_...
    - test_delay     : mixedmods_test_delay_...
    
    参数:
      - mod_list: 调制方式列表，如果指定且与默认不同，会在文件名中添加标记
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
        
        # 如果指定了调制列表且与默认不同，添加调制标记
        mod_tag = ""
        if mod_list is not None and set(mod_list) != set(MOD_LIST):
            mod_tag = "_mod" + "".join([m.replace("PSK", "P").replace("QAM", "Q") for m in sorted(mod_list)])
        
        base = (
            f"mixedmods_train_{profile_tag}_rand_"
            f"{freq_tag}_{phi1_tag}_{phi2_tag}_{amp_tag}_{snr_tag}_N{N_for_name}"
            f"{mod_tag}{extra_tag}{dtype_tag}"
        )
        if shard_idx is not None:
            return os.path.join(save_dir, f"{base}_shard{int(shard_idx)}.pth")
        else:
            return os.path.join(save_dir, f"{base}.pth")

    if mode == "test_all_qpsk":
        return os.path.join(
            save_dir,
            f'qpsk_test_all_grid_F1{len(FREQ1_GRID)}_F2{len(FREQ2_GRID)}'
            f'_P1{len(PHASE1_GRID)}_P2{len(PHASE2_GRID)}'
            f'_A{len(AMP_ALL_GRID_QPSK)}_S{len(SNR_GRID_QPSK)}'
            f'_R{args.test_repeats}{extra_tag}.pth'
        )
    if mode == "test_all_8psk":
        return os.path.join(
            save_dir,
            f'8psk_test_all_grid_F1{len(FREQ1_GRID)}_F2{len(FREQ2_GRID)}'
            f'_P1{len(PHASE1_GRID)}_P2{len(PHASE2_GRID)}'
            f'_A{len(AMP_ALL_GRID_QPSK)}_S{len(SNR_GRID_QPSK)}'
            f'_R{args.test_repeats}{extra_tag}.pth'
        )
    elif mode == "test_snr_amp":
        return os.path.join(
            save_dir,
            f'mixedmods_test_snr-amp_S{len(SNR_GRID_SNR_AMP)}_A{len(AMP_GRID_SNR_AMP)}'
            f'_R{args.test_repeats}{extra_tag}.pth'
        )
    elif mode == "test_snr_amp_8psk":
        return os.path.join(
            save_dir,
            f'8psk_test_snr-amp_S{len(SNR_GRID_SNR_AMP)}_A{len(AMP_GRID_SNR_AMP)}'
            f'_R{args.test_repeats}{extra_tag}.pth'
        )
    elif mode == "test_cfo_phase":
        return os.path.join(
            save_dir,
            f'mixedmods_test_cfo-phase_F{len(CFO_GRID_CFO_PHASE)}_P{len(PHASE_DIFF_GRID)}'
            f'_R{args.test_repeats}{extra_tag}.pth'
        )
    elif mode == "test_cfo_phase_8psk":
        return os.path.join(
            save_dir,
            f'8psk_test_cfo-phase_F{len(CFO_GRID_CFO_PHASE)}_P{len(PHASE_DIFF_GRID)}'
            f'_R{args.test_repeats}{extra_tag}.pth'
        )
    elif mode == "test_delay":
        return os.path.join(
            save_dir,
            f'mixedmods_test_delay_D{len(DELAY_DIFF_SAMP_GRID)}'
            f'_R{args.test_repeats}{extra_tag}.pth'
        )
    elif mode == "test_delay_8psk":
        return os.path.join(
            save_dir,
            f'8psk_test_delay_D{len(DELAY_DIFF_SAMP_GRID)}'
            f'_R{args.test_repeats}{extra_tag}.pth'
        )
    elif mode == "test_comparison":
        return os.path.join(
            save_dir,
            f'qpsk_test_comparison_S{len(SNR_GRID_COMPARISON)}_A{len(AMP_GRID_COMPARISON)}'
            f'_R{args.test_repeats}{extra_tag}.pth'
        )
    else:
        raise ValueError(f"未知 mode: {mode}")


# ============= 训练生成（多N + 分片） =============
def gen_one_train_dataset(N_total, SHARD_SIZE, train_profile: str, mod_list=None):
    """
    生成一个给定规模 N_total 的训练集，并按 shard 增量保存。

    特点：
      - 每个样本的两路调制方式都是随机的：mod1, mod2 ∈ mod_list（如果指定）或 MOD_LIST（默认）；
      - 超参数分布由 train_profile 决定（aligned/robust）。
    
    参数:
      - mod_list: 要使用的调制方式列表，如果为 None 则使用 MOD_LIST
    """
    train_profile = train_profile.lower()
    if mod_list is None:
        mod_list = MOD_LIST
    
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
        entries_norm = energy_normalize_dataset(entries)
        entries_norm = [maybe_cast_complex64(e) for e in entries_norm]
        path = build_save_path(
            "train",
            extra_tag="_varsnr_ampr_phi1phi2_delay0T",
            shard_idx=shard_idx,
            N_for_name=N_total,
            train_profile=train_profile,
            mod_list=mod_list,
        )
        torch.save(entries_norm, path)
        print(f"📦 已保存分片 {shard_idx}/{num_shards}: {path} （样本数 {len(entries_norm)}）")
        return path

    for k in range(N_total):
        # -------- 0) 为每路随机选择调制方式 --------
        mod1 = np.random.choice(mod_list)
        mod2 = np.random.choice(mod_list)

        bit_len1 = get_bit_len(mod1)
        bit_len2 = get_bit_len(mod2)

        # -------- 1) 随机比特并调制映射 --------
        bits1 = np.random.randint(0, 2, bit_len1, dtype=np.int8)
        bits2 = np.random.randint(0, 2, bit_len2, dtype=np.int8)
        symbols1 = modulate(bits1, mod1)
        symbols2 = modulate(bits2, mod2)

        assert len(symbols1) == num_syms
        assert len(symbols2) == num_syms

        # -------- 2) 上采样 + 分数符号时延（0~T 区间，采样级） --------
        up_len = num_syms * sps
        symbols_up1 = np.zeros(up_len, dtype=complex)
        symbols_up2 = np.zeros(up_len, dtype=complex)

        delay_samp1 = np.random.randint(0, sps)  # τ1 ∈ [0, T)
        delay_samp2 = np.random.randint(0, sps)  # τ2 ∈ [0, T)

        amplitude_ratio = np.random.uniform(*amp_range)

        symbols_up1[delay_samp1::sps] = symbols1
        symbols_up2[delay_samp2::sps] = symbols2 * amplitude_ratio

        # -------- 3) RC 成型滤波 --------
        tx1 = convolve(symbols_up1, rc, mode='same')
        tx2 = convolve(symbols_up2, rc, mode='same')

        # -------- 4) 随机 CFO + 初相位 --------
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
        # 之前为了节省空间，训练集中的 bits1/bits2 默认置为 -1。
        # 现在按需求改为保存真实比特串，便于后续做 BER/SER 分析和可微仿真对齐。
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
            'bits1': bits1,
            'bits2': bits2,
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
    
    # 解析调制列表
    mod_list = parse_modulation_list(args.modulation_list)
    
    print(
        f"👉 模式: train | 调制: 每路随机于 {mod_list} | profile: {args.train_profile} | "
        f"N 列表: {N_list} | 分片大小: {SHARD_SIZE} | "
        f"保存dtype: {'complex64' if args.save_complex64 else 'complex128'}"
    )

    for N in N_list:
        gen_one_train_dataset(N_total=N,
                              SHARD_SIZE=SHARD_SIZE,
                              train_profile=args.train_profile,
                              mod_list=mod_list)


# ============= 测试集：test_all_qpsk（原 test_all） =============
def _run_test_all_single_mod(modulation: str, save_mode: str):
    """
    test_all 逻辑：两路均为同一种调制（modulation），f1/f2/phi1/phi2/amp/SNR 走网格，
    另外为每个组合生成 delay 版本（delay1/2 ∈ {0,2,4,6} 采样点）。
    """
    modulation = modulation.upper()
    bit_len = get_bit_len(modulation)
    print(f"👉 模式: {save_mode} （两路 {modulation}，CFO/相位/幅度/SNR 网格 + delay）")

    combos = list(itertools.product(FREQ1_GRID, FREQ2_GRID,
                                    PHASE1_GRID, PHASE2_GRID,
                                    AMP_ALL_GRID_QPSK, SNR_GRID_QPSK))
    total = len(combos) * args.test_repeats
    print(
        f"[{save_mode}] 网格组合数 {len(combos)}，每组重复 {args.test_repeats} 次，"
        f"共 {total} 组（每组含无delay+有delay 各1块）"
    )

    dataset_delay = []    # 只保存带 delay 的版本（和你之前用的 _delay4grid 一致）

    global_idx = 0
    for combo_idx, (f1, f2, phi1, phi2, amp, snr_val) in enumerate(combos):
        for r in range(args.test_repeats):
            combo_bytes = (
                f"{save_mode}|mod{modulation}|f1{f1}|f2{f2}|phi1{phi1}|phi2{phi2}"
                f"|a{amp}|snr{snr_val}|rep{r}"
            ).encode()
            seed = int(hashlib.sha256(combo_bytes).hexdigest()[:8], 16)

            rng_bits = np.random.default_rng(seed)
            bits1 = rng_bits.integers(0, 2, bit_len, dtype=np.int8)
            bits2 = rng_bits.integers(0, 2, bit_len, dtype=np.int8)

            symbols1 = modulate(bits1, modulation)
            symbols2 = modulate(bits2, modulation)

            up_len = len(symbols1) * sps
            t = np.arange(up_len) / fs

            # ===== 只存带 delay 的版本 =====
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

            seed_rx_d = seed ^ 0x3C3C3C3C
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
                print(
                    f"[{save_mode}] 已生成 {global_idx}/{total} 组"
                    f"（每组：只存有delay版本1块）"
                )

    save_path_d = build_save_path(save_mode, extra_tag="_delay4grid")
    dataset_d_normed = energy_normalize_dataset(dataset_delay)
    dataset_d_normed = [maybe_cast_complex64(e) for e in dataset_d_normed]
    torch.save(dataset_d_normed, save_path_d)
    print(f"✅ {modulation} + delay 网格版本已保存至: {save_path_d}")


def run_test_all_qpsk():
    _run_test_all_single_mod(modulation="QPSK", save_mode="test_all_qpsk")


def run_test_all_8psk():
    _run_test_all_single_mod(modulation="8PSK", save_mode="test_all_8psk")


# ============= 测试集：test_snr_amp（CFO=0, 无 delay） =============
def run_test_snr_amp():
    """
    test_snr_amp：
      - 目的：画 BER–SNR per AMP 图；
      - 变量：
          * SNR ∈ SNR_GRID_SNR_AMP = {8,10,...,22} dB
          * AMP ∈ AMP_GRID_SNR_AMP = {0.3,0.5,0.7,0.9}
      - 固定：
          * CFO1 = 0 Hz, CFO2 = 0 Hz
          * phi1 = 0, phi2 = 0
          * delay1 = delay2 = 0（完全对齐）
      - 调制组合（四种）：
          * (QPSK, QPSK)
          * (8PSK, 8PSK)
          * (16QAM, 16QAM)
          * (QPSK, 16QAM)
    """
    print("👉 模式: test_snr_amp （CFO=0, 无 delay，四种调制组合）")

    mod_pairs = [
        ("QPSK", "QPSK"),
        ("8PSK", "8PSK"),
        ("16QAM", "16QAM"),
        ("QPSK", "16QAM"),
    ]

    # 所有网格组合：mod_pair × SNR × AMP
    combos = list(itertools.product(mod_pairs, SNR_GRID_SNR_AMP, AMP_GRID_SNR_AMP))
    total = len(combos) * args.test_repeats
    print(
        f"[test_snr_amp] 网格组合数 {len(combos)}，每组重复 {args.test_repeats} 次，"
        f"总样本数 ≈ {total}"
    )

    dataset = []
    global_idx = 0

    for combo_idx, ((mod1, mod2), snr_val, amp) in enumerate(combos):
        for r in range(args.test_repeats):
            combo_bytes = (
                f"snr-amp|mod1{mod1}|mod2{mod2}|snr{snr_val}|amp{amp}|rep{r}"
            ).encode()
            seed = int(hashlib.sha256(combo_bytes).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)

            bit_len1 = get_bit_len(mod1)
            bit_len2 = get_bit_len(mod2)

            bits1 = rng.integers(0, 2, bit_len1, dtype=np.int8)
            bits2 = rng.integers(0, 2, bit_len2, dtype=np.int8)

            symbols1 = modulate(bits1, mod1)
            symbols2 = modulate(bits2, mod2)

            assert len(symbols1) == num_syms
            assert len(symbols2) == num_syms

            up_len = num_syms * sps
            t = np.arange(up_len) / fs

            # delay 全 0
            delay1_samp = 0
            delay2_samp = 0
            amp_float = float(amp)

            symbols_up1 = np.zeros(up_len, dtype=complex)
            symbols_up2 = np.zeros(up_len, dtype=complex)
            symbols_up1[delay1_samp::sps] = symbols1
            symbols_up2[delay2_samp::sps] = symbols2 * amp_float

            tx1 = convolve(symbols_up1, rc, mode='same')
            tx2 = convolve(symbols_up2, rc, mode='same')

            # CFO=0, phi=0
            tx1 = tx1  # * exp(j*0)
            tx2 = tx2

            seed_rx = seed ^ 0x12345678
            rx = awgn_with_seed(tx1 + tx2, float(snr_val), seed_rx)

            new_entry = {
                'mixsignal': rx,
                'rfsignal1': tx1,
                'rfsignal2': tx2,
                'params': (
                    float(snr_val), amp_float, sps,
                    'f_off1=0.00Hz',
                    'f_off2=0.00Hz',
                    'phi1=0.0000rad',
                    'phi2=0.0000rad',
                    f'delay1_samp={delay1_samp}',
                    f'delay2_samp={delay2_samp}',
                    f'mod1={mod1}',
                    f'mod2={mod2}',
                    f'rep={r}',
                ),
                'bits1': bits1,
                'bits2': bits2,
                'origin_len': 1
            }
            dataset.append(new_entry)

            global_idx += 1
            if global_idx % 200 == 0:
                print(f"[test_snr_amp] 已生成 {global_idx}/{total} 块")

    save_path = build_save_path("test_snr_amp", extra_tag="")
    dataset_normed = energy_normalize_dataset(dataset)
    dataset_normed = [maybe_cast_complex64(e) for e in dataset_normed]
    torch.save(dataset_normed, save_path)
    print(f"✅ test_snr_amp 数据集已保存至: {save_path}")


def _auto_set_test_repeats(mod1: str, mod2: str, label: str):
    """
    根据目标 BER 估算每个网格点需要的 repeats：
      total_bits_per_point ≈ min_expected_errors / target_ber
      bits_per_sample_point = num_syms * (bps(mod1) + bps(mod2))
      repeats = ceil(total_bits_per_point / bits_per_sample_point)
    """
    if not args.auto_test_repeats:
        return
    mod1 = mod1.upper()
    mod2 = mod2.upper()
    bps_total = BITS_PER_SYMBOL[mod1] + BITS_PER_SYMBOL[mod2]
    bits_per_sample = int(num_syms * bps_total)
    target_bits = int(np.ceil(float(args.min_expected_errors) / float(args.target_ber)))
    repeats = int(np.ceil(target_bits / max(1, bits_per_sample)))
    repeats = max(1, repeats)
    old = args.test_repeats
    args.test_repeats = repeats
    print(
        f"[auto_test_repeats] {label}: target_ber={args.target_ber:g}, "
        f"min_expected_errors={args.min_expected_errors} => target_bits≈{target_bits}；"
        f"bits/sample={bits_per_sample} (num_syms={num_syms}, bps_total={bps_total})；"
        f"test_repeats: {old} -> {args.test_repeats}"
    )


def run_test_snr_amp_8psk():
    """
    8PSK-only 版本：只生成 (8PSK, 8PSK) 的 SNR×AMP 网格，避免 mixedmods 多组合导致文件膨胀。
    """
    print("👉 模式: test_snr_amp_8psk （CFO=0, 无 delay，仅 (8PSK,8PSK)）")
    mod1, mod2 = "8PSK", "8PSK"
    _auto_set_test_repeats(mod1, mod2, label="test_snr_amp_8psk")

    combos = list(itertools.product(SNR_GRID_SNR_AMP, AMP_GRID_SNR_AMP))
    total = len(combos) * args.test_repeats
    print(
        f"[test_snr_amp_8psk] 网格组合数 {len(combos)}，每组重复 {args.test_repeats} 次，"
        f"总样本数 ≈ {total}"
    )

    dataset = []
    global_idx = 0
    for (snr_val, amp) in combos:
        for r in range(args.test_repeats):
            combo_bytes = (
                f"snr-amp-8psk|mod1{mod1}|mod2{mod2}|snr{snr_val}|amp{amp}|rep{r}"
            ).encode()
            seed = int(hashlib.sha256(combo_bytes).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)

            bit_len1 = get_bit_len(mod1)
            bit_len2 = get_bit_len(mod2)
            bits1 = rng.integers(0, 2, bit_len1, dtype=np.int8)
            bits2 = rng.integers(0, 2, bit_len2, dtype=np.int8)
            symbols1 = modulate(bits1, mod1)
            symbols2 = modulate(bits2, mod2)

            up_len = num_syms * sps
            symbols_up1 = np.zeros(up_len, dtype=complex)
            symbols_up2 = np.zeros(up_len, dtype=complex)
            symbols_up1[0::sps] = symbols1
            symbols_up2[0::sps] = symbols2 * float(amp)
            tx1 = convolve(symbols_up1, rc, mode='same')
            tx2 = convolve(symbols_up2, rc, mode='same')

            seed_rx = seed ^ 0x12345678
            rx = awgn_with_seed(tx1 + tx2, float(snr_val), seed_rx)

            dataset.append({
                'mixsignal': rx,
                'rfsignal1': tx1,
                'rfsignal2': tx2,
                'params': (
                    float(snr_val), float(amp), sps,
                    'f_off1=0.00Hz',
                    'f_off2=0.00Hz',
                    'phi1=0.0000rad',
                    'phi2=0.0000rad',
                    'delay1_samp=0',
                    'delay2_samp=0',
                    f'mod1={mod1}',
                    f'mod2={mod2}',
                    f'rep={r}',
                ),
                'bits1': bits1,
                'bits2': bits2,
                'origin_len': 1
            })
            global_idx += 1
            if global_idx % 200 == 0:
                print(f"[test_snr_amp_8psk] 已生成 {global_idx}/{total} 块")

    save_path = build_save_path("test_snr_amp_8psk", extra_tag="")
    dataset_normed = energy_normalize_dataset(dataset)
    dataset_normed = [maybe_cast_complex64(e) for e in dataset_normed]
    torch.save(dataset_normed, save_path)
    print(f"✅ test_snr_amp_8psk 数据集已保存至: {save_path}")


# ============= 测试集：test_cfo_phase（扫 ΔCFO × Δphi） =============
def run_test_cfo_phase():
    """
    test_cfo_phase：
      - 目的：画 BER–ΔCFO per Δphi 图；
      - 变量：
          * CFO1 = 0，CFO2 ∈ CFO_GRID_CFO_PHASE
          * Δphi ∈ PHASE_DIFF_GRID（八个经典点）
      - 固定：
          * SNR = 16 dB
          * AMP = 0.5
          * delay1 = delay2 = 0
      - 调制组合：同样四种 (QPSK,QPSK),(8PSK,8PSK),(16QAM,16QAM),(QPSK,16QAM)
    """
    print("👉 模式: test_cfo_phase （扫 ΔCFO × Δphi，四种调制组合）")

    mod_pairs = [
        ("QPSK", "QPSK"),
        ("8PSK", "8PSK"),
        ("16QAM", "16QAM"),
        ("QPSK", "16QAM"),
    ]
    snr_val = 16.0
    amp = 0.5

    combos = list(itertools.product(mod_pairs, CFO_GRID_CFO_PHASE, PHASE_DIFF_GRID))
    total = len(combos) * args.test_repeats
    print(
        f"[test_cfo_phase] 网格组合数 {len(combos)}，每组重复 {args.test_repeats} 次，"
        f"总样本数 ≈ {total}"
    )

    dataset = []
    global_idx = 0

    for combo_idx, ((mod1, mod2), cfo2, dphi) in enumerate(combos):
        for r in range(args.test_repeats):
            combo_bytes = (
                f"cfo-phase|mod1{mod1}|mod2{mod2}|cfo2{cfo2}|dphi{dphi}|rep{r}"
            ).encode()
            seed = int(hashlib.sha256(combo_bytes).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)

            bit_len1 = get_bit_len(mod1)
            bit_len2 = get_bit_len(mod2)

            bits1 = rng.integers(0, 2, bit_len1, dtype=np.int8)
            bits2 = rng.integers(0, 2, bit_len2, dtype=np.int8)

            symbols1 = modulate(bits1, mod1)
            symbols2 = modulate(bits2, mod2)

            assert len(symbols1) == num_syms
            assert len(symbols2) == num_syms

            up_len = num_syms * sps
            t = np.arange(up_len) / fs

            delay1_samp = 0
            delay2_samp = 0
            amp_float = float(amp)

            symbols_up1 = np.zeros(up_len, dtype=complex)
            symbols_up2 = np.zeros(up_len, dtype=complex)
            symbols_up1[delay1_samp::sps] = symbols1
            symbols_up2[delay2_samp::sps] = symbols2 * amp_float

            tx1 = convolve(symbols_up1, rc, mode='same')
            tx2 = convolve(symbols_up2, rc, mode='same')

            # CFO1=0, CFO2=cfo2；phi1=0, phi2=Δphi
            tx1 = tx1
            tx2 = tx2 * np.exp(1j * (2 * np.pi * float(cfo2) * t + float(dphi)))

            seed_rx = seed ^ 0x87654321
            rx = awgn_with_seed(tx1 + tx2, float(snr_val), seed_rx)

            new_entry = {
                'mixsignal': rx,
                'rfsignal1': tx1,
                'rfsignal2': tx2,
                'params': (
                    float(snr_val), amp_float, sps,
                    'f_off1=0.00Hz',
                    f'f_off2={float(cfo2):.2f}Hz',
                    'phi1=0.0000rad',
                    f'phi2={float(dphi):.4f}rad',
                    f'delay1_samp={delay1_samp}',
                    f'delay2_samp={delay2_samp}',
                    f'mod1={mod1}',
                    f'mod2={mod2}',
                    f'rep={r}',
                ),
                'bits1': bits1,
                'bits2': bits2,
                'origin_len': 1
            }
            dataset.append(new_entry)

            global_idx += 1
            if global_idx % 200 == 0:
                print(f"[test_cfo_phase] 已生成 {global_idx}/{total} 块")

    save_path = build_save_path("test_cfo_phase", extra_tag="")
    dataset_normed = energy_normalize_dataset(dataset)
    dataset_normed = [maybe_cast_complex64(e) for e in dataset_normed]
    torch.save(dataset_normed, save_path)
    print(f"✅ test_cfo_phase 数据集已保存至: {save_path}")


def run_test_cfo_phase_8psk():
    """
    8PSK-only 版本：只生成 (8PSK, 8PSK) 的 ΔCFO×Δphi 网格（SNR/AMP 固定，无 delay）。
    """
    print("👉 模式: test_cfo_phase_8psk （扫 ΔCFO × Δphi，仅 (8PSK,8PSK)）")
    mod1, mod2 = "8PSK", "8PSK"
    _auto_set_test_repeats(mod1, mod2, label="test_cfo_phase_8psk")

    snr_val = 16.0
    amp = 0.5
    combos = list(itertools.product(CFO_GRID_CFO_PHASE, PHASE_DIFF_GRID))
    total = len(combos) * args.test_repeats
    print(
        f"[test_cfo_phase_8psk] 网格组合数 {len(combos)}，每组重复 {args.test_repeats} 次，"
        f"总样本数 ≈ {total}"
    )

    dataset = []
    global_idx = 0
    for (cfo2, dphi) in combos:
        for r in range(args.test_repeats):
            combo_bytes = (
                f"cfo-phase-8psk|mod1{mod1}|mod2{mod2}|cfo2{cfo2}|dphi{dphi}|rep{r}"
            ).encode()
            seed = int(hashlib.sha256(combo_bytes).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)

            bit_len1 = get_bit_len(mod1)
            bit_len2 = get_bit_len(mod2)
            bits1 = rng.integers(0, 2, bit_len1, dtype=np.int8)
            bits2 = rng.integers(0, 2, bit_len2, dtype=np.int8)
            symbols1 = modulate(bits1, mod1)
            symbols2 = modulate(bits2, mod2)

            up_len = num_syms * sps
            t = np.arange(up_len) / fs
            symbols_up1 = np.zeros(up_len, dtype=complex)
            symbols_up2 = np.zeros(up_len, dtype=complex)
            symbols_up1[0::sps] = symbols1
            symbols_up2[0::sps] = symbols2 * float(amp)
            tx1 = convolve(symbols_up1, rc, mode='same')
            tx2 = convolve(symbols_up2, rc, mode='same')

            tx2 = tx2 * np.exp(1j * (2 * np.pi * float(cfo2) * t + float(dphi)))
            seed_rx = seed ^ 0x87654321
            rx = awgn_with_seed(tx1 + tx2, float(snr_val), seed_rx)

            dataset.append({
                'mixsignal': rx,
                'rfsignal1': tx1,
                'rfsignal2': tx2,
                'params': (
                    float(snr_val), float(amp), sps,
                    'f_off1=0.00Hz',
                    f'f_off2={float(cfo2):.2f}Hz',
                    'phi1=0.0000rad',
                    f'phi2={float(dphi):.4f}rad',
                    'delay1_samp=0',
                    'delay2_samp=0',
                    f'mod1={mod1}',
                    f'mod2={mod2}',
                    f'rep={r}',
                ),
                'bits1': bits1,
                'bits2': bits2,
                'origin_len': 1
            })
            global_idx += 1
            if global_idx % 200 == 0:
                print(f"[test_cfo_phase_8psk] 已生成 {global_idx}/{total} 块")

    save_path = build_save_path("test_cfo_phase_8psk", extra_tag="")
    dataset_normed = energy_normalize_dataset(dataset)
    dataset_normed = [maybe_cast_complex64(e) for e in dataset_normed]
    torch.save(dataset_normed, save_path)
    print(f"✅ test_cfo_phase_8psk 数据集已保存至: {save_path}")


# ============= 测试集：test_delay（扫 delay_diff） =============
def run_test_delay():
    """
    test_delay：
      - 目的：画 BER–delay_diff 图；
      - 变量：
          * delay1_samp = 0
          * delay2_samp ∈ DELAY_DIFF_SAMP_GRID = {0,2,4,6}
      - 固定：
          * SNR = 16 dB
          * AMP = 0.5
          * CFO1 = CFO2 = 0
          * phi1 = phi2 = 0
      - 调制组合：四种 (QPSK,QPSK),(8PSK,8PSK),(16QAM,16QAM),(QPSK,16QAM)
    """
    print("👉 模式: test_delay （扫 delay_diff，四种调制组合）")

    mod_pairs = [
        ("QPSK", "QPSK"),
        ("8PSK", "8PSK"),
        ("16QAM", "16QAM"),
        ("QPSK", "16QAM"),
    ]
    snr_val = 16.0
    amp = 0.5

    combos = list(itertools.product(mod_pairs, DELAY_DIFF_SAMP_GRID))
    total = len(combos) * args.test_repeats
    print(
        f"[test_delay] 网格组合数 {len(combos)}，每组重复 {args.test_repeats} 次，"
        f"总样本数 ≈ {total}"
    )

    dataset = []
    global_idx = 0

    for combo_idx, ((mod1, mod2), delay2_samp) in enumerate(combos):
        for r in range(args.test_repeats):
            combo_bytes = (
                f"delay|mod1{mod1}|mod2{mod2}|d2{delay2_samp}|rep{r}"
            ).encode()
            seed = int(hashlib.sha256(combo_bytes).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)

            bit_len1 = get_bit_len(mod1)
            bit_len2 = get_bit_len(mod2)

            bits1 = rng.integers(0, 2, bit_len1, dtype=np.int8)
            bits2 = rng.integers(0, 2, bit_len2, dtype=np.int8)

            symbols1 = modulate(bits1, mod1)
            symbols2 = modulate(bits2, mod2)

            assert len(symbols1) == num_syms
            assert len(symbols2) == num_syms

            up_len = num_syms * sps
            t = np.arange(up_len) / fs

            delay1_samp = 0
            amp_float = float(amp)

            symbols_up1 = np.zeros(up_len, dtype=complex)
            symbols_up2 = np.zeros(up_len, dtype=complex)
            symbols_up1[delay1_samp::sps] = symbols1
            symbols_up2[int(delay2_samp)::sps] = symbols2 * amp_float

            tx1 = convolve(symbols_up1, rc, mode='same')
            tx2 = convolve(symbols_up2, rc, mode='same')

            # CFO=0, phi=0
            tx1 = tx1
            tx2 = tx2

            seed_rx = seed ^ 0xABCDEF01
            rx = awgn_with_seed(tx1 + tx2, float(snr_val), seed_rx)

            delay_diff = int(delay2_samp - delay1_samp)

            new_entry = {
                'mixsignal': rx,
                'rfsignal1': tx1,
                'rfsignal2': tx2,
                'params': (
                    float(snr_val), amp_float, sps,
                    'f_off1=0.00Hz',
                    'f_off2=0.00Hz',
                    'phi1=0.0000rad',
                    'phi2=0.0000rad',
                    f'delay1_samp={delay1_samp}',
                    f'delay2_samp={int(delay2_samp)}',
                    f'delay_diff_samp={delay_diff}',
                    f'mod1={mod1}',
                    f'mod2={mod2}',
                    f'rep={r}',
                ),
                'bits1': bits1,
                'bits2': bits2,
                'origin_len': 1
            }
            dataset.append(new_entry)

            global_idx += 1
            if global_idx % 200 == 0:
                print(f"[test_delay] 已生成 {global_idx}/{total} 块")

    save_path = build_save_path("test_delay", extra_tag="")
    dataset_normed = energy_normalize_dataset(dataset)
    dataset_normed = [maybe_cast_complex64(e) for e in dataset_normed]
    torch.save(dataset_normed, save_path)
    print(f"✅ test_delay 数据集已保存至: {save_path}")


def run_test_delay_8psk():
    """
    8PSK-only 版本：只生成 (8PSK, 8PSK) 的 delay_diff 网格（SNR/AMP 固定，CFO=0, phi=0）。
    """
    print("👉 模式: test_delay_8psk （扫 delay_diff，仅 (8PSK,8PSK)）")
    mod1, mod2 = "8PSK", "8PSK"
    _auto_set_test_repeats(mod1, mod2, label="test_delay_8psk")

    snr_val = 16.0
    amp = 0.5
    combos = list(DELAY_DIFF_SAMP_GRID)
    total = len(combos) * args.test_repeats
    print(
        f"[test_delay_8psk] 网格组合数 {len(combos)}，每组重复 {args.test_repeats} 次，"
        f"总样本数 ≈ {total}"
    )

    dataset = []
    global_idx = 0
    for delay2_samp in combos:
        for r in range(args.test_repeats):
            combo_bytes = (
                f"delay-8psk|mod1{mod1}|mod2{mod2}|d2{delay2_samp}|rep{r}"
            ).encode()
            seed = int(hashlib.sha256(combo_bytes).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)

            bit_len1 = get_bit_len(mod1)
            bit_len2 = get_bit_len(mod2)
            bits1 = rng.integers(0, 2, bit_len1, dtype=np.int8)
            bits2 = rng.integers(0, 2, bit_len2, dtype=np.int8)
            symbols1 = modulate(bits1, mod1)
            symbols2 = modulate(bits2, mod2)

            up_len = num_syms * sps
            symbols_up1 = np.zeros(up_len, dtype=complex)
            symbols_up2 = np.zeros(up_len, dtype=complex)
            symbols_up1[0::sps] = symbols1
            symbols_up2[int(delay2_samp)::sps] = symbols2 * float(amp)
            tx1 = convolve(symbols_up1, rc, mode='same')
            tx2 = convolve(symbols_up2, rc, mode='same')

            seed_rx = seed ^ 0xABCDEF01
            rx = awgn_with_seed(tx1 + tx2, float(snr_val), seed_rx)

            dataset.append({
                'mixsignal': rx,
                'rfsignal1': tx1,
                'rfsignal2': tx2,
                'params': (
                    float(snr_val), float(amp), sps,
                    'f_off1=0.00Hz',
                    'f_off2=0.00Hz',
                    'phi1=0.0000rad',
                    'phi2=0.0000rad',
                    'delay1_samp=0',
                    f'delay2_samp={int(delay2_samp)}',
                    f'delay_diff_samp={int(delay2_samp)}',
                    f'mod1={mod1}',
                    f'mod2={mod2}',
                    f'rep={r}',
                ),
                'bits1': bits1,
                'bits2': bits2,
                'origin_len': 1
            })
            global_idx += 1
            if global_idx % 200 == 0:
                print(f"[test_delay_8psk] 已生成 {global_idx}/{total} 块")

    save_path = build_save_path("test_delay_8psk", extra_tag="")
    dataset_normed = energy_normalize_dataset(dataset)
    dataset_normed = [maybe_cast_complex64(e) for e in dataset_normed]
    torch.save(dataset_normed, save_path)
    print(f"✅ test_delay_8psk 数据集已保存至: {save_path}")


# ============= 测试集：test_comparison（轻量级QPSK对比测试集） =============
def run_test_comparison():
    """
    test_comparison：
      - 目的：生成轻量级QPSK测试集，用于新旧模型快速对比；
      - 变量：
          * SNR ∈ SNR_GRID_COMPARISON = {10,12,14,16,18,20,22} dB
          * AMP ∈ AMP_GRID_COMPARISON = {0.3,0.5,0.7,0.9}
      - 固定：
          * 两路均为 QPSK
          * CFO1 = CFO2 = 0 Hz
          * phi1 = phi2 = 0
          * delay1 = delay2 = 0（完全对齐）
      - 特点：样本数适中，便于快速测试和对比
    """
    print("👉 模式: test_comparison （轻量级QPSK对比测试集）")
    
    modulation = "QPSK"
    bit_len = get_bit_len(modulation)
    
    combos = list(itertools.product(SNR_GRID_COMPARISON, AMP_GRID_COMPARISON))
    total = len(combos) * args.test_repeats
    print(
        f"[test_comparison] 网格组合数 {len(combos)}，每组重复 {args.test_repeats} 次，"
        f"总样本数 = {total}"
    )
    
    dataset = []
    global_idx = 0
    
    for combo_idx, (snr_val, amp) in enumerate(combos):
        for r in range(args.test_repeats):
            combo_bytes = (
                f"comparison|QPSK|QPSK|snr{snr_val}|amp{amp}|rep{r}"
            ).encode()
            seed = int(hashlib.sha256(combo_bytes).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            
            bits1 = rng.integers(0, 2, bit_len, dtype=np.int8)
            bits2 = rng.integers(0, 2, bit_len, dtype=np.int8)
            
            symbols1 = modulate(bits1, modulation)
            symbols2 = modulate(bits2, modulation)
            
            assert len(symbols1) == num_syms
            assert len(symbols2) == num_syms
            
            up_len = num_syms * sps
            t = np.arange(up_len) / fs
            
            # 固定参数：delay=0, CFO=0, phase=0
            delay1_samp = 0
            delay2_samp = 0
            amp_float = float(amp)
            
            symbols_up1 = np.zeros(up_len, dtype=complex)
            symbols_up2 = np.zeros(up_len, dtype=complex)
            symbols_up1[delay1_samp::sps] = symbols1
            symbols_up2[delay2_samp::sps] = symbols2 * amp_float
            
            tx1 = convolve(symbols_up1, rc, mode='same')
            tx2 = convolve(symbols_up2, rc, mode='same')
            
            # CFO=0, phi=0
            tx1 = tx1
            tx2 = tx2
            
            seed_rx = seed ^ 0xC0FFEE01
            rx = awgn_with_seed(tx1 + tx2, float(snr_val), seed_rx)
            
            new_entry = {
                'mixsignal': rx,
                'rfsignal1': tx1,
                'rfsignal2': tx2,
                'params': (
                    float(snr_val), amp_float, sps,
                    'f_off1=0.00Hz',
                    'f_off2=0.00Hz',
                    'phi1=0.0000rad',
                    'phi2=0.0000rad',
                    f'delay1_samp={delay1_samp}',
                    f'delay2_samp={delay2_samp}',
                    f'mod1={modulation}',
                    f'mod2={modulation}',
                    f'rep={r}',
                ),
                'bits1': bits1,
                'bits2': bits2,
                'origin_len': 1
            }
            dataset.append(new_entry)
            
            global_idx += 1
            if global_idx % 50 == 0:
                print(f"[test_comparison] 已生成 {global_idx}/{total} 块")
    
    save_path = build_save_path("test_comparison", extra_tag="")
    dataset_normed = energy_normalize_dataset(dataset)
    dataset_normed = [maybe_cast_complex64(e) for e in dataset_normed]
    torch.save(dataset_normed, save_path)
    print(f"✅ test_comparison 数据集已保存至: {save_path}")
    print(f"   总样本数: {len(dataset_normed)}")
    print(f"   网格: SNR={len(SNR_GRID_COMPARISON)}点 × AMP={len(AMP_GRID_COMPARISON)}点 × 重复{args.test_repeats}次")


# ============= 主入口 =============
if __name__ == "__main__":
    if args.mode == "train":
        run_train_multiN()
    elif args.mode == "test_all_qpsk":
        run_test_all_qpsk()
    elif args.mode == "test_all_8psk":
        run_test_all_8psk()
    elif args.mode == "test_snr_amp":
        run_test_snr_amp()
    elif args.mode == "test_cfo_phase":
        run_test_cfo_phase()
    elif args.mode == "test_delay":
        run_test_delay()
    elif args.mode == "test_snr_amp_8psk":
        run_test_snr_amp_8psk()
    elif args.mode == "test_cfo_phase_8psk":
        run_test_cfo_phase_8psk()
    elif args.mode == "test_delay_8psk":
        run_test_delay_8psk()
    elif args.mode == "test_comparison":
        run_test_comparison()
