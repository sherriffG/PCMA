"""
Pure-PyTorch differentiable signal generation utilities.

用法示例（放在你的训练/实验脚本里）::

    import torch
    from torch_signal_sim import simulate_two_user_mix

    # 基本配置
    num_syms = 384       # 符号数，例如 input_len=3072, sps=8 => 384
    BPS_8PSK = 3

    device = "cuda"
    bits1 = torch.randint(0, 2, (num_syms * BPS_8PSK,), dtype=torch.int64, device=device)
    bits2 = torch.randint(0, 2, (num_syms * BPS_8PSK,), dtype=torch.int64, device=device)

    # 连续参数可以是可学习的 Tensor
    amp_ratio = torch.tensor(0.5, device=device, requires_grad=True)
    f1 = torch.tensor(50.0, device=device, requires_grad=True)
    f2 = torch.tensor(-80.0, device=device, requires_grad=True)
    phi1 = torch.tensor(0.3, device=device, requires_grad=True)
    phi2 = torch.tensor(1.0, device=device, requires_grad=True)
    snr_db = torch.tensor(16.0, device=device, requires_grad=True)

    # 生成两路混合波形
    rx, tx1, tx2 = simulate_two_user_mix(
        bits1, bits2,
        mod1="8PSK", mod2="8PSK",
        amp_ratio=amp_ratio,
        freq_offset1_hz=f1, freq_offset2_hz=f2,
        phase1=phi1, phase2=phi2,
        snr_db=snr_db,
        sps=8, beta=0.33, num_taps=64,
        fs=12e6,
        delay1_samp=0, delay2_samp=0,
    )

    # 举例：对一个简单的 loss 反向传播，验证梯度
    loss = rx.real.pow(2).mean()
    loss.backward()
    print("grad_amp_ratio =", amp_ratio.grad)
    print("grad_f1        =", f1.grad)
    print("grad_snr_db    =", snr_db.grad)

主要用途：
- 在 PyTorch 计算图中生成两路调制信号混合的波形；
- 输入参数（幅度比、CFO、相位、SNR 等）可以是带 requires_grad 的标量 Tensor；
- 输出波形（混合信号、单路信号）是可反向传播的 torch.Tensor。

注意：
- 随机比特与噪声本身不是可微的（离散 & 随机），但相对于“连续参数”的梯度是可传播的。
"""

import math
from typing import Tuple

import torch
import torch.nn.functional as F


# ========================
# 工具函数
# ========================

def _to_tensor(x, device, dtype=torch.float32) -> torch.Tensor:
    """保持已有 Tensor 的 requires_grad 属性，否则创建新 Tensor。"""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def create_rrc_filter(beta: float = 0.33, sps: int = 8, num_taps: int = 64, device=None) -> torch.Tensor:
    """
    Root-Raised-Cosine (RRC) 滤波器（与 generate_sim_dataset.py 中 rc_filter 行为对齐）。

    返回形状: (num_taps,) 的 1D Tensor，可用于 conv1d.
    """
    device = device or "cpu"
    # 注意：这里使用与 numpy 版本一致的近似公式
    t = torch.arange(-num_taps // 2, num_taps // 2, device=device, dtype=torch.float32) / float(sps)

    # sinc(x) = sin(pi x) / (pi x)
    pi_t = math.pi * t
    sinc_val = torch.sin(pi_t) / (pi_t + 1e-12)
    sinc_val[torch.isinf(sinc_val)] = 1.0
    sinc_val[torch.isnan(sinc_val)] = 1.0

    num = sinc_val * torch.cos(math.pi * beta * t)
    den = 1.0 - (2 * beta * t) ** 2
    h = num / (den + 1e-12)

    # 处理数值异常
    nan_mask = torch.isnan(h)
    if nan_mask.any():
        h[nan_mask] = 1.0 - beta + (4 * beta / math.pi)

    h = h / torch.sqrt(torch.sum(h**2) + 1e-12)
    return h


# ========================
# 调制映射（Torch 版本）
# ========================

def qpsk_mod_torch(bits: torch.Tensor) -> torch.Tensor:
    """
    QPSK Gray 映射（与 numpy 版 qpsk_mod 一致）。
    bits: (num_bits,) 0/1 Tensor
    返回: (num_syms,) complex64 Tensor
    """
    device = bits.device
    bits = bits.to(dtype=torch.int64)
    assert bits.numel() % 2 == 0, "QPSK 需要偶数长度 bit 序列"
    b = bits.view(-1, 2)
    b1 = b[:, 0].float()
    b2 = b[:, 1].float()
    I = 1.0 - 2.0 * b1  # 0->+1, 1->-1
    Q = 1.0 - 2.0 * b2
    s = torch.complex(I, Q) / math.sqrt(2.0)
    return s


def psk8_mod_torch(bits: torch.Tensor) -> torch.Tensor:
    """
    8PSK 映射：
    - 每 3bit -> 一个符号，采用自然编码：k = b0*4 + b1*2 + b2（与 numpy 版 psk8_mod 对齐）
    - 符号 = exp(j*(2πk/8))
    """
    device = bits.device
    bits = bits.to(dtype=torch.int64)
    assert bits.numel() % 3 == 0, "8PSK 需要 3 的倍数长度 bit 序列"
    b = bits.view(-1, 3)
    idx = b[:, 0] * 4 + b[:, 1] * 2 + b[:, 2]
    phase = 2 * math.pi * idx.float() / 8.0
    s = torch.exp(1j * phase)
    return s


def qam16_mod_torch(bits: torch.Tensor) -> torch.Tensor:
    """
    16QAM Gray 映射（保持与 numpy 版 qam16_mod 一致）。
    bits: (num_bits,) 0/1 Tensor
    返回: (num_syms,) complex64 Tensor
    """
    bits = bits.to(dtype=torch.int64)
    assert bits.numel() % 4 == 0, "16QAM 需要 4 的倍数长度 bit 序列"
    b = bits.view(-1, 4)

    def gray2level_torch(b0, b1):
        # 逐元素计算 Gray 映射：(-3,-1,1,3)
        out = torch.empty_like(b0, dtype=torch.float32)
        mask = (b0 == 0) & (b1 == 0)
        out[mask] = -3.0
        mask = (b0 == 0) & (b1 == 1)
        out[mask] = -1.0
        mask = (b0 == 1) & (b1 == 1)
        out[mask] = 1.0
        mask = (b0 == 1) & (b1 == 0)
        out[mask] = 3.0
        return out

    I = gray2level_torch(b[:, 0], b[:, 1])
    Q = gray2level_torch(b[:, 2], b[:, 3])
    s = torch.complex(I, Q) / math.sqrt(10.0)
    return s


def modulate_torch(bits: torch.Tensor, modulation: str) -> torch.Tensor:
    """
    通用调制入口（Torch 版），返回 complex Tensor.
    """
    modulation = modulation.upper()
    if modulation == "QPSK":
        return qpsk_mod_torch(bits)
    elif modulation == "8PSK":
        return psk8_mod_torch(bits)
    elif modulation == "16QAM":
        return qam16_mod_torch(bits)
    else:
        raise ValueError(f"不支持的调制方式: {modulation}")


# ========================
# AWGN + 上采样 + 成型 + 混合
# ========================

def awgn_torch(signal: torch.Tensor, snr_db: torch.Tensor, rng=None) -> torch.Tensor:
    """
    添加复高斯白噪声（Torch 版），与 numpy 版 awgn_with_seed 行为一致。

    Args:
        signal: (T,) complex Tensor
        snr_db: 标量 Tensor 或 float
    """
    device = signal.device
    snr_db = _to_tensor(snr_db, device=device)
    power = torch.mean(signal.real**2 + signal.imag**2)
    snr_linear = torch.pow(10.0, snr_db / 10.0)
    noise_power = power / (snr_linear + 1e-12)
    std = torch.sqrt(noise_power / 2.0)
    # 随机噪声本身不可微，但对 signal / snr_db 的梯度是可传播的
    noise_real = torch.randn_like(signal.real)
    noise_imag = torch.randn_like(signal.imag)
    noise = std * torch.complex(noise_real, noise_imag)
    return signal + noise


def upsample_and_filter(
    symbols: torch.Tensor,
    sps: int = 8,
    beta: float = 0.33,
    num_taps: int = 64,
    delay_samp: int = 0,
    device=None,
) -> torch.Tensor:
    """
    上采样 + RRC 成型滤波（Torch 版）。

    Args:
        symbols: (num_syms,) complex Tensor
        sps: 每符号采样数
        delay_samp: 整体采样级延时（用于两路 delay 差）
    Returns:
        tx: (num_syms * sps,) complex Tensor
    """
    device = device or symbols.device
    num_syms = symbols.numel()
    up_len = num_syms * sps

    # 上采样并插入延时
    real_up = torch.zeros(up_len, device=device, dtype=torch.float32)
    imag_up = torch.zeros(up_len, device=device, dtype=torch.float32)

    idx = torch.arange(num_syms, device=device) * sps + int(delay_samp)
    mask = (idx >= 0) & (idx < up_len)
    idx = idx[mask]
    real_up[idx] = symbols.real[mask]
    imag_up[idx] = symbols.imag[mask]

    rrc = create_rrc_filter(beta=beta, sps=sps, num_taps=num_taps, device=device)
    rrc = rrc.view(1, 1, -1)

    real_up_ = real_up.view(1, 1, -1)
    imag_up_ = imag_up.view(1, 1, -1)
    real_f = F.conv1d(real_up_, rrc, padding=num_taps // 2).view(-1)[:up_len]
    imag_f = F.conv1d(imag_up_, rrc, padding=num_taps // 2).view(-1)[:up_len]

    return torch.complex(real_f, imag_f)


def apply_cfo_phase(
    tx: torch.Tensor,
    freq_offset_hz: torch.Tensor,
    init_phase: torch.Tensor,
    fs: float = 12e6,
) -> torch.Tensor:
    """
    应用 CFO + 初始相位（Torch 版）。

    Args:
        tx: (T,) complex Tensor
        freq_offset_hz: 标量 Tensor 或 float
        init_phase: 标量 Tensor 或 float
    """
    device = tx.device
    T = tx.numel()
    freq_offset_hz = _to_tensor(freq_offset_hz, device=device)
    init_phase = _to_tensor(init_phase, device=device)
    t = torch.arange(T, device=device, dtype=torch.float32) / float(fs)
    phase = 2 * math.pi * freq_offset_hz * t + init_phase
    rot = torch.exp(1j * phase)
    return tx * rot


def simulate_two_user_mix(
    bits1: torch.Tensor,
    bits2: torch.Tensor,
    mod1: str,
    mod2: str,
    amp_ratio: torch.Tensor,
    freq_offset1_hz: torch.Tensor,
    freq_offset2_hz: torch.Tensor,
    phase1: torch.Tensor,
    phase2: torch.Tensor,
    snr_db: torch.Tensor,
    *,
    sps: int = 8,
    beta: float = 0.33,
    num_taps: int = 64,
    fs: float = 12e6,
    delay1_samp: int = 0,
    delay2_samp: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    纯 Torch 版两路信号混合仿真（与 generate_sim_dataset 中 train 模式逻辑等价的核心路径）。

    输入参数可以是标量 Tensor（支持 requires_grad=True），输出:
      - mix: (T,) complex Tensor
      - tx1: (T,) complex Tensor
      - tx2: (T,) complex Tensor
    """
    device = bits1.device

    # 调制
    s1 = modulate_torch(bits1, mod1)
    s2 = modulate_torch(bits2, mod2)

    # 路径 1/2 的幅度比：|s2| = amp_ratio * |s1|
    amp_ratio = _to_tensor(amp_ratio, device=device)
    s2 = s2 * amp_ratio

    # 上采样 + 成型 + delay
    tx1 = upsample_and_filter(s1, sps=sps, beta=beta, num_taps=num_taps, delay_samp=delay1_samp, device=device)
    tx2 = upsample_and_filter(s2, sps=sps, beta=beta, num_taps=num_taps, delay_samp=delay2_samp, device=device)

    # CFO + 初相位
    tx1 = apply_cfo_phase(tx1, freq_offset1_hz, phase1, fs=fs)
    tx2 = apply_cfo_phase(tx2, freq_offset2_hz, phase2, fs=fs)

    # 合路 + AWGN
    rx_clean = tx1 + tx2
    rx = awgn_torch(rx_clean, snr_db)
    return rx, tx1, tx2


