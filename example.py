"""
最小示例：信号生成 -> 模型推理 -> SER评估 -> 信号重建
完整流程演示，包含可视化
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy.signal import convolve

# 设置中文字体，避免警告
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
from model_complex import SignalSeparator

# ==================== 参数设置 ====================
BETA = 0.33
SPS = 8
FS = 12e6
NUM_TAPS = 64
INPUT_LEN = 3072  # 每块样本点数
NUM_SYMS = INPUT_LEN // SPS  # 每路符号数

BITS_PER_SYMBOL = {
    "QPSK": 2,
    "8PSK": 3,
    "16QAM": 4,
}

# ==================== RRC滤波器 ====================
def rc_filter(beta, sps, num_taps):
    """创建RRC滤波器"""
    t = np.arange(-num_taps//2, num_taps//2) / sps
    with np.errstate(divide='ignore', invalid='ignore'):
        h = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t) ** 2)
        h[np.isnan(h)] = 1.0 - beta + (4 * beta / np.pi)
    h = h / np.sqrt(np.sum(h**2))
    return h

rc = rc_filter(BETA, SPS, NUM_TAPS)

# ==================== 调制函数（与generate_sim_dataset.py完全一致）====================
def qpsk_mod(bits):
    """QPSK Gray 映射，与generate_sim_dataset.py完全一致"""
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

def psk8_mod(bits):
    """8PSK调制，与generate_sim_dataset.py完全一致"""
    assert len(bits) % 3 == 0
    bits = bits.reshape(-1, 3)
    idx = bits[:, 0] * 4 + bits[:, 1] * 2 + bits[:, 2]
    phase = 2 * np.pi * idx / 8.0
    symbols = np.exp(1j * phase)
    return symbols.astype(complex)

def qam16_mod(bits):
    """16QAM调制，与generate_sim_dataset.py完全一致"""
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

def modulate(bits, modulation):
    """根据调制方式调制"""
    mod = modulation.upper()
    if mod == "QPSK":
        return qpsk_mod(bits)
    elif mod == "8PSK":
        return psk8_mod(bits)
    elif mod == "16QAM":
        return qam16_mod(bits)
    else:
        raise ValueError(f"未知调制方式: {modulation}")

# ==================== 解调函数 ====================
def qpsk_demod(symbols):
    """QPSK解调"""
    bits = []
    sym = symbols * np.sqrt(2)
    for s in sym:
        if s.real >= 0 and s.imag >= 0: b1, b2 = 0, 0
        elif s.real < 0 and s.imag >= 0: b1, b2 = 0, 1
        elif s.real >= 0 and s.imag < 0: b1, b2 = 1, 0
        else: b1, b2 = 1, 1
        bits.extend([b1, b2])
    return np.array(bits, dtype=np.int8)

def psk8_demod(symbols):
    """8PSK解调"""
    angles = np.angle(symbols)
    angles = np.mod(angles, 2*np.pi)
    step = 2*np.pi / 8.0
    k = np.round(angles / step).astype(int) % 8
    bits = []
    for val in k:
        b0 = (val >> 2) & 1
        b1 = (val >> 1) & 1
        b2 = val & 1
        bits.extend([b0, b1, b2])
    return np.array(bits, dtype=np.int8)

def qam16_demod(symbols):
    """16QAM解调"""
    levels = np.array([-3., -1., 1., 3.]) / np.sqrt(10.0)
    bits = []
    for s in symbols:
        I = s.real
        Q = s.imag
        idx_I = np.argmin((I - levels)**2)
        idx_Q = np.argmin((Q - levels)**2)
        level_I = levels[idx_I]
        level_Q = levels[idx_Q]

        if level_I < (-2/np.sqrt(10)):   bi0, bi1 = 0, 0
        elif level_I < 0:                bi0, bi1 = 0, 1
        elif level_I > (2/np.sqrt(10)):  bi0, bi1 = 1, 0
        else:                            bi0, bi1 = 1, 1

        if level_Q < (-2/np.sqrt(10)):   bq0, bq1 = 0, 0
        elif level_Q < 0:                bq0, bq1 = 0, 1
        elif level_Q > (2/np.sqrt(10)):  bq0, bq1 = 1, 0
        else:                            bq0, bq1 = 1, 1

        bits.extend([bi0, bi1, bq0, bq1])
    return np.array(bits, dtype=np.int8)

def demodulate(symbols, modulation):
    """根据调制方式解调"""
    mod = modulation.upper()
    if mod == "QPSK":
        return qpsk_demod(symbols)
    elif mod == "8PSK":
        return psk8_demod(symbols)
    elif mod == "16QAM":
        return qam16_demod(symbols)
    else:
        raise ValueError(f"未知调制方式: {modulation}")

# ==================== AWGN噪声 ====================
def awgn_with_seed(signal, snr_db, seed=None):
    """添加AWGN噪声"""
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal))
    )
    return signal + noise

# ==================== 信号生成（与generate_sim_dataset.py完全一致）====================
def generate_signal_pair(bits1, bits2, modulation1, modulation2, 
                        amp_ratio=1.0, freq_offset1_hz=0.0, freq_offset2_hz=0.0,
                        phase1_rad=0.0, phase2_rad=0.0, delay1_samp=0, delay2_samp=0,
                        snr_db=None, seed=None):
    """
    生成两路信号并混合（与generate_sim_dataset.py的test_snr_amp流程完全一致）
    
    Args:
        bits1, bits2: 比特序列
        modulation1, modulation2: 调制方式
        amp_ratio: 信号2相对于信号1的幅度比
        freq_offset1_hz, freq_offset2_hz: 频偏（Hz）
        phase1_rad, phase2_rad: 相偏（弧度）
        delay1_samp, delay2_samp: 时延（采样点数）
        snr_db: 信噪比（针对混合信号）
        seed: 随机种子（用于噪声生成）
    
    Returns:
        mixsignal: 混合信号（已添加噪声）
        rfsignal1: 信号1（未添加噪声）
        rfsignal2: 信号2（未添加噪声）
        symbols1, symbols2: 原始符号
    """
    # 调制
    symbols1 = modulate(bits1, modulation1)
    symbols2 = modulate(bits2, modulation2)
    
    # 上采样
    up_len = len(symbols1) * SPS
    assert len(symbols1) == len(symbols2), "符号数必须相同"
    
    symbols_up1 = np.zeros(up_len, dtype=complex)
    symbols_up2 = np.zeros(up_len, dtype=complex)
    symbols_up1[delay1_samp::SPS] = symbols1
    symbols_up2[delay2_samp::SPS] = symbols2 * amp_ratio
    
    # RRC滤波
    tx1 = convolve(symbols_up1, rc, mode='same')
    tx2 = convolve(symbols_up2, rc, mode='same')
    
    # CFO和相位（与generate_sim_dataset.py一致：先滤波，后应用CFO/相位）
    t = np.arange(up_len) / FS
    tx1 = tx1 * np.exp(1j * (2 * np.pi * freq_offset1_hz * t + phase1_rad))
    tx2 = tx2 * np.exp(1j * (2 * np.pi * freq_offset2_hz * t + phase2_rad))
    
    # 混合并添加噪声（与generate_sim_dataset.py一致）
    mixsignal_clean = tx1 + tx2
    if snr_db is not None:
        seed_rx = seed ^ 0x12345678 if seed is not None else None
        mixsignal = awgn_with_seed(mixsignal_clean, snr_db, seed_rx)
    else:
        mixsignal = mixsignal_clean
    
    return mixsignal, tx1, tx2, symbols1, symbols2

# ==================== 匹配滤波和符号抽取 ====================
def find_best_offset(y_mf, sps):
    """找到最佳符号抽取offset"""
    best_off = 0
    best_eng = -1.0
    for off in range(sps):
        sym = y_mf[off::sps]
        if len(sym) > 0:
            eng = np.mean(np.abs(sym)**2)
            if eng > best_eng:
                best_eng = eng
                best_off = off
    return best_off

def mf_and_sample(wave, sps, rc, num_taps, guard_sym=None, offset=None):
    """
    匹配滤波并抽取符号（与test_sim_SignalSeparator.py一致）
    包含幅度归一化
    """
    if guard_sym is None:
        guard_sym = num_taps // sps  # 64/8=8 符号
    
    if wave is None or len(wave) == 0:
        return np.zeros(0, dtype=np.complex64)
    
    # 匹配滤波
    y_mf = convolve(wave, rc, mode='same')
    
    # 找到最佳offset
    if offset is None:
        off = find_best_offset(y_mf, sps)
    else:
        off = offset
    
    # 抽取符号
    syms = y_mf[off::sps]
    
    # 去除guard符号
    if len(syms) <= 2 * guard_sym:
        return np.zeros(0, dtype=np.complex64)
    syms = syms[guard_sym:-guard_sym]
    
    # 幅度归一化（与test_sim_SignalSeparator.py一致）
    m = np.mean(np.abs(syms))
    if m > 0:
        syms = syms / m
    
    return syms.astype(np.complex64)

# ==================== 相位对齐 ====================
def align_phase(ref, est):
    """
    相位对齐函数（与test_sim_SignalSeparator.py一致）
    估计全局相位偏移并旋转est使其对齐到ref
    """
    c = np.mean(ref * np.conj(est) + 1e-12)
    a = np.angle(c)
    return est * np.exp(-1j * a)

# ==================== 符号时序对齐 ====================
def align_symbols_sequence(pred_symbols, true_symbols, max_shift=5):
    """
    通过互相关找到最佳对齐位置，解决符号时序偏移问题
    """
    if len(pred_symbols) == 0 or len(true_symbols) == 0:
        return pred_symbols, true_symbols
    
    min_len = min(len(pred_symbols), len(true_symbols))
    if min_len < max_shift * 2:
        return pred_symbols[:min_len], true_symbols[:min_len]
    
    # 归一化符号以便比较
    pred_power = np.mean(np.abs(pred_symbols)**2)
    true_power = np.mean(np.abs(true_symbols)**2)
    
    if pred_power > 0:
        pred_norm = pred_symbols / np.sqrt(pred_power)
    else:
        pred_norm = pred_symbols
    
    if true_power > 0:
        true_norm = true_symbols / np.sqrt(true_power)
    else:
        true_norm = true_symbols
    
    # 计算互相关，找到最佳对齐位置
    best_shift = 0
    best_corr = -1.0
    
    for shift in range(-max_shift, max_shift + 1):
        if shift == 0:
            pred_shifted = pred_norm[:min_len]
            true_shifted = true_norm[:min_len]
        elif shift > 0:
            if len(pred_norm) > shift and len(true_norm) >= min_len:
                pred_shifted = pred_norm[shift:shift+min_len]
                true_shifted = true_norm[:min_len]
            else:
                continue
        else:  # shift < 0
            if len(true_norm) > -shift and len(pred_norm) >= min_len:
                pred_shifted = pred_norm[:min_len]
                true_shifted = true_norm[-shift:-shift+min_len]
            else:
                continue
        
        if len(pred_shifted) == len(true_shifted) and len(pred_shifted) > 0:
            corr = np.abs(np.mean(pred_shifted * np.conj(true_shifted)))
            if corr > best_corr:
                best_corr = corr
                best_shift = shift
    
    # 应用最佳偏移
    if best_shift == 0:
        aligned_pred = pred_symbols[:min_len]
        aligned_true = true_symbols[:min_len]
    elif best_shift > 0:
        aligned_pred = pred_symbols[best_shift:best_shift+min_len]
        aligned_true = true_symbols[:min_len]
    else:  # best_shift < 0
        aligned_pred = pred_symbols[:min_len]
        aligned_true = true_symbols[-best_shift:-best_shift+min_len]
    
    return aligned_pred, aligned_true

# ==================== SER计算（与test_sim_SignalSeparator.py完全一致）====================
def calculate_ser_from_bits(pred_symbols, true_symbols, true_bits_full, modulation):
    """
    计算误符号率（SER），与test_sim_SignalSeparator.py完全一致
    使用真实的比特序列作为参考，而不是从符号重新解调
    """
    if len(pred_symbols) == 0 or len(true_symbols) == 0:
        return 1.0
    
    # 相位对齐（在符号级别）
    pred_symbols_aligned = align_phase(true_symbols, pred_symbols)
    
    # 解调预测符号
    pred_bits = demodulate(pred_symbols_aligned, modulation)
    
    # 计算SER：按符号分组比较比特组合（与test_sim_SignalSeparator.py一致）
    bits_per_sym = BITS_PER_SYMBOL[modulation.upper()]
    
    # 确保比特序列长度是bps的倍数
    L_pred = len(pred_bits)
    L_true = len(true_bits_full)
    L_pred = (L_pred // bits_per_sym) * bits_per_sym
    L_true = (L_true // bits_per_sym) * bits_per_sym
    min_len = min(L_pred, L_true)
    min_len = (min_len // bits_per_sym) * bits_per_sym
    
    if min_len <= 0:
        return 1.0
    
    # 将比特序列按符号分组（每bps个比特为一个符号）
    n_syms = min_len // bits_per_sym
    
    # 将比特序列重塑为 (n_syms, bps) 形状
    pred_bits_syms = pred_bits[:min_len].reshape(n_syms, bits_per_sym)
    true_bits_syms = true_bits_full[:min_len].reshape(n_syms, bits_per_sym)
    
    # 比较每个符号的比特组合是否相同
    sym_errors = np.any(pred_bits_syms != true_bits_syms, axis=1)
    
    ser = float(np.mean(sym_errors)) if n_syms > 0 else 1.0
    return ser

# ==================== 可视化 ====================
def plot_results(mix_signal, true_signal1, true_signal2, 
                pred_signal1, pred_signal2,
                true_symbols1, true_symbols2,
                pred_symbols1, pred_symbols2,
                ser1, ser2, modulation1, modulation2,
                amp_ratio=0.5,
                save_path="example_results.png"):
    """绘制结果对比图"""
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 时域信号对比
    ax1 = plt.subplot(3, 3, 1)
    t = np.arange(len(mix_signal)) / FS * 1e6  # 转换为微秒
    ax1.plot(t[:500], mix_signal[:500].real, 'b-', alpha=0.7, label='混合信号(实部)')
    ax1.plot(t[:500], true_signal1[:500].real, 'g--', alpha=0.7, label='真实信号1(实部)')
    ax1.plot(t[:500], pred_signal1[:500].real, 'r:', alpha=0.7, label='预测信号1(实部)')
    ax1.set_xlabel('时间 (μs)')
    ax1.set_ylabel('幅度')
    ax1.set_title('时域信号对比 (前500点)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 信号1星座图
    ax2 = plt.subplot(3, 3, 2)
    ax2.scatter(true_symbols1.real, true_symbols1.imag, c='g', marker='o', 
                alpha=0.5, s=20, label='真实符号')
    ax2.scatter(pred_symbols1.real, pred_symbols1.imag, c='r', marker='x', 
                alpha=0.5, s=20, label='预测符号')
    ax2.set_xlabel('I')
    ax2.set_ylabel('Q')
    ax2.set_title(f'信号1星座图 ({modulation1}, SER={ser1:.4f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # 3. 信号2星座图
    ax3 = plt.subplot(3, 3, 3)
    ax3.scatter(true_symbols2.real, true_symbols2.imag, c='g', marker='o', 
                alpha=0.5, s=20, label='真实符号')
    ax3.scatter(pred_symbols2.real, pred_symbols2.imag, c='r', marker='x', 
                alpha=0.5, s=20, label='预测符号')
    ax3.set_xlabel('I')
    ax3.set_ylabel('Q')
    ax3.set_title(f'信号2星座图 ({modulation2}, SER={ser2:.4f})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # 4. 频域对比（信号1）
    ax4 = plt.subplot(3, 3, 4)
    freq = np.fft.fftfreq(len(true_signal1), 1/FS) / 1e6  # MHz
    true_psd1 = np.abs(np.fft.fft(true_signal1))**2
    pred_psd1 = np.abs(np.fft.fft(pred_signal1))**2
    ax4.plot(freq[:len(freq)//2], 10*np.log10(true_psd1[:len(freq)//2] + 1e-10), 
             'g-', alpha=0.7, label='真实信号1')
    ax4.plot(freq[:len(freq)//2], 10*np.log10(pred_psd1[:len(freq)//2] + 1e-10), 
             'r--', alpha=0.7, label='预测信号1')
    ax4.set_xlabel('频率 (MHz)')
    ax4.set_ylabel('功率谱密度 (dB)')
    ax4.set_title('信号1功率谱')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 频域对比（信号2）
    ax5 = plt.subplot(3, 3, 5)
    freq = np.fft.fftfreq(len(true_signal2), 1/FS) / 1e6  # MHz
    true_psd2 = np.abs(np.fft.fft(true_signal2))**2
    pred_psd2 = np.abs(np.fft.fft(pred_signal2))**2
    ax5.plot(freq[:len(freq)//2], 10*np.log10(true_psd2[:len(freq)//2] + 1e-10), 
             'g-', alpha=0.7, label='真实信号2')
    ax5.plot(freq[:len(freq)//2], 10*np.log10(pred_psd2[:len(freq)//2] + 1e-10), 
             'r--', alpha=0.7, label='预测信号2')
    ax5.set_xlabel('频率 (MHz)')
    ax5.set_ylabel('功率谱密度 (dB)')
    ax5.set_title('信号2功率谱')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 重建信号对比（信号1）
    ax6 = plt.subplot(3, 3, 6)
    # 用预测符号重建信号（直接使用符号，不需要再解调）
    rebuild_signal1 = np.zeros(len(true_signal1), dtype=complex)
    up_len = len(pred_symbols1) * SPS
    if up_len <= len(rebuild_signal1):
        symbols_up1 = np.zeros(up_len, dtype=complex)
        symbols_up1[::SPS] = pred_symbols1[:up_len//SPS]
        rebuild_signal1[:up_len] = convolve(symbols_up1, rc, mode='same')[:up_len]
    t = np.arange(len(true_signal1)) / FS * 1e6
    ax6.plot(t[:500], true_signal1[:500].real, 'g-', alpha=0.7, label='原始信号1')
    ax6.plot(t[:500], rebuild_signal1[:500].real, 'r--', alpha=0.7, label='重建信号1')
    ax6.set_xlabel('时间 (μs)')
    ax6.set_ylabel('幅度')
    ax6.set_title('信号1重建对比')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. 重建信号对比（信号2）
    ax7 = plt.subplot(3, 3, 7)
    rebuild_signal2 = np.zeros(len(true_signal2), dtype=complex)
    up_len = len(pred_symbols2) * SPS
    if up_len <= len(rebuild_signal2):
        symbols_up2 = np.zeros(up_len, dtype=complex)
        symbols_up2[::SPS] = pred_symbols2[:up_len//SPS] * amp_ratio
        rebuild_signal2[:up_len] = convolve(symbols_up2, rc, mode='same')[:up_len]
    t = np.arange(len(true_signal2)) / FS * 1e6
    ax7.plot(t[:500], true_signal2[:500].real, 'g-', alpha=0.7, label='原始信号2')
    ax7.plot(t[:500], rebuild_signal2[:500].real, 'r--', alpha=0.7, label='重建信号2')
    ax7.set_xlabel('时间 (μs)')
    ax7.set_ylabel('幅度')
    ax7.set_title('信号2重建对比')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. SER汇总
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')
    ser_text = f"""
    SER evaluation results:
    
    Sig1 ({modulation1}):
      SER = {ser1:.6f}
      ({int(ser1 * len(true_symbols1))}/{len(true_symbols1)} wrong symbols)
    
    Sig2 ({modulation2}):
      SER = {ser2:.6f}
      ({int(ser2 * len(true_symbols2))}/{len(true_symbols2)} wrong symbols)
    
    Avg SER = {(ser1 + ser2) / 2:.6f}
    """
    ax8.text(0.1, 0.5, ser_text, fontsize=12, verticalalignment='center',
             family='monospace')
    
    # 9. 混合信号频谱
    ax9 = plt.subplot(3, 3, 9)
    freq = np.fft.fftfreq(len(mix_signal), 1/FS) / 1e6  # MHz
    mix_psd = np.abs(np.fft.fft(mix_signal))**2
    ax9.plot(freq[:len(freq)//2], 10*np.log10(mix_psd[:len(freq)//2] + 1e-10), 
             'b-', alpha=0.7, label='混合信号')
    ax9.set_xlabel('频率 (MHz)')
    ax9.set_ylabel('功率谱密度 (dB)')
    ax9.set_title('混合信号功率谱')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"结果图已保存到: {save_path}")
    plt.close()

# ==================== 主函数 ====================
def main():
    print("="*60)
    print("最小示例：信号生成 -> 模型推理 -> SER评估 -> 信号重建")
    print("="*60)
    
    # 参数设置
    modulation1 = "QPSK"
    modulation2 = "8PSK"
    amp_ratio = 0.5
    snr_db = 15.0
    freq_offset1 = 20  # Hz
    freq_offset2 = -30  # Hz
    phase1 = 0  # rad
    phase2 = 0  # rad
    delay1 = 0
    delay2 = 0
    
    # 模型检查点路径（需要根据实际情况修改）
    ckpt_path = "/nas/datasets/yixin/PCMA/src/check_points/mixedmod/signal_separator_mixedmods_train_aligned_rand_freqU[0,130]_phi1U[0.0000,6.2832]_phi2U[0.0000,6.2832]_ampU[0.30,0.90]_snrU[12,18]_N100000_varsnr_ampr_phi1phi2_delay0T_c64_latest_epoch180.pth"
    
    print(f"\n参数设置:")
    print(f"  信号1: {modulation1}, 频偏={freq_offset1}Hz, 相偏={phase1:.3f}rad")
    print(f"  信号2: {modulation2}, 幅度比={amp_ratio}, 频偏={freq_offset2}Hz, 相偏={phase2:.3f}rad")
    print(f"  SNR: {snr_db}dB")
    
    # 1. 生成比特序列（使用固定种子，与generate_sim_dataset.py一致）
    print(f"\n步骤1: 生成比特序列...")
    bit_len1 = NUM_SYMS * BITS_PER_SYMBOL[modulation1]
    bit_len2 = NUM_SYMS * BITS_PER_SYMBOL[modulation2]
    
    # 使用与generate_sim_dataset.py相同的种子生成方式
    combo_bytes = f"snr-amp|mod1{modulation1}|mod2{modulation2}|snr{snr_db}|amp{amp_ratio}|rep0".encode()
    import hashlib
    seed = int(hashlib.sha256(combo_bytes).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    
    bits1 = rng.integers(0, 2, bit_len1, dtype=np.int8)
    bits2 = rng.integers(0, 2, bit_len2, dtype=np.int8)
    print(f"  信号1比特数: {bit_len1}")
    print(f"  信号2比特数: {bit_len2}")
    print(f"  使用种子: {seed}")
    
    # 2. 生成两路信号并混合（与generate_sim_dataset.py完全一致）
    print(f"\n步骤2: 生成两路信号并混合...")
    mix_signal, signal1, signal2, symbols1, symbols2 = generate_signal_pair(
        bits1, bits2, modulation1, modulation2,
        amp_ratio=amp_ratio,
        freq_offset1_hz=freq_offset1, freq_offset2_hz=freq_offset2,
        phase1_rad=phase1, phase2_rad=phase2,
        delay1_samp=delay1, delay2_samp=delay2,
        snr_db=snr_db, seed=seed
    )
    print(f"  混合信号长度: {len(mix_signal)}")
    
    # 3. 能量归一化（与generate_sim_dataset.py一致）
    print(f"\n步骤3: 能量归一化...")
    mix_energy = np.mean(np.abs(mix_signal) ** 2)
    scale = np.sqrt(mix_energy)
    mix_signal = mix_signal / scale
    signal1 = signal1 / scale
    signal2 = signal2 / scale
    print(f"  归一化因子: {scale:.6f}")
    
    # 4. 准备模型输入（转换为模型需要的格式）
    print(f"\n步骤4: 准备模型输入...")
    # 模型输入格式: (B, 2, T)，其中第0维是实部，第1维是虚部
    mixsignal_ri = torch.stack([
        torch.from_numpy(mix_signal.real.astype(np.float32)),
        torch.from_numpy(mix_signal.imag.astype(np.float32))
    ], dim=0).unsqueeze(0)  # (1, 2, T)
    
    # 5. 加载模型并推理
    print(f"\n步骤5: 加载模型并推理...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  使用设备: {device}")
    
    model = SignalSeparator()
    try:
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"  ✓ 模型加载成功: {ckpt_path}")
    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        print(f"  使用随机初始化的模型（仅用于演示）")
    
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        mixsignal_ri = mixsignal_ri.to(device)
        
        # 模型返回一个列表，包含4个元素: [pred1_real, pred1_imag, pred2_real, pred2_imag]
        # 每个元素的形状是 (B, 1, T)
        output = model(mixsignal_ri)
        
        # 如果返回的是列表，提取各个部分
        if isinstance(output, (list, tuple)):
            pred1_real = output[0]  # (B, 1, T)
            pred1_imag = output[1]  # (B, 1, T)
            pred2_real = output[2]  # (B, 1, T)
            pred2_imag = output[3]  # (B, 1, T)
        else:
            # 如果返回的是拼接后的张量 (B, 4, T)
            pred1_real = output[:, 0:1, :]
            pred1_imag = output[:, 1:2, :]
            pred2_real = output[:, 2:3, :]
            pred2_imag = output[:, 3:4, :]
        
        # 转换为numpy并组合成复数信号
        pred_signal1 = (pred1_real[0, 0].cpu().numpy() + 1j * pred1_imag[0, 0].cpu().numpy())
        pred_signal2 = (pred2_real[0, 0].cpu().numpy() + 1j * pred2_imag[0, 0].cpu().numpy())
    
    print(f"  预测信号1长度: {len(pred_signal1)}")
    print(f"  预测信号2长度: {len(pred_signal2)}")
    
    # 6. 频偏和相偏补偿（与test_sim_SignalSeparator.py完全一致）
    print(f"\n步骤6: 频偏和相偏补偿...")
    n = np.arange(len(pred_signal1))
    t = n / FS
    
    # 对预测信号和真实信号都进行CFO和相位补偿（与test_sim_SignalSeparator.py一致）
    pred_signal1_comp = pred_signal1 * np.exp(-1j * (2 * np.pi * float(freq_offset1) * t + float(phase1)))
    pred_signal2_comp = pred_signal2 * np.exp(-1j * (2 * np.pi * float(freq_offset2) * t + float(phase2)))
    true_signal1_comp = signal1 * np.exp(-1j * (2 * np.pi * float(freq_offset1) * t + float(phase1)))
    true_signal2_comp = signal2 * np.exp(-1j * (2 * np.pi * float(freq_offset2) * t + float(phase2)))
    
    print(f"  已补偿信号1: 频偏={freq_offset1}Hz, 相偏={phase1:.3f}rad")
    print(f"  已补偿信号2: 频偏={freq_offset2}Hz, 相偏={phase2:.3f}rad")
    
    # 7. 从补偿后的信号中提取符号
    print(f"\n步骤7: 从补偿后的信号中提取符号...")
    pred_symbols1 = mf_and_sample(pred_signal1_comp, SPS, rc, NUM_TAPS)
    pred_symbols2 = mf_and_sample(pred_signal2_comp, SPS, rc, NUM_TAPS)
    
    # 从真实信号中提取符号（用于对比）
    true_symbols1 = mf_and_sample(true_signal1_comp, SPS, rc, NUM_TAPS)
    true_symbols2 = mf_and_sample(true_signal2_comp, SPS, rc, NUM_TAPS)
    
    print(f"  真实符号1数量: {len(true_symbols1)}")
    print(f"  预测符号1数量: {len(pred_symbols1)}")
    print(f"  真实符号2数量: {len(true_symbols2)}")
    print(f"  预测符号2数量: {len(pred_symbols2)}")
    
    # 8. 相位对齐（在符号级别，与test_sim_SignalSeparator.py一致）
    print(f"\n步骤8: 符号级别相位对齐...")
    pred_symbols1_aligned = align_phase(true_symbols1, pred_symbols1)
    pred_symbols2_aligned = align_phase(true_symbols2, pred_symbols2)
    
    # 9. 计算SER（与test_sim_SignalSeparator.py完全一致）
    print(f"\n步骤9: 计算误符号率...")
    
    # 使用真实的bits1和bits2作为参考（与test_sim_SignalSeparator.py一致）
    # 需要根据实际使用的符号数量来切片bits
    def slice_bits_to_match_syms(bits_full, n_syms_used, bits_per_sym):
        """与test_sim_SignalSeparator.py一致"""
        if len(bits_full) == 0 or n_syms_used <= 0 or bits_per_sym <= 0:
            return np.zeros(0, dtype=np.int8)
        n_sym_total = len(bits_full) // bits_per_sym
        n_syms_used = min(n_syms_used, n_sym_total)
        if n_sym_total <= n_syms_used:
            return bits_full[:bits_per_sym * n_syms_used]
        guard_sym = max((n_sym_total - n_syms_used) // 2, 0)
        start = guard_sym * bits_per_sym
        end = start + bits_per_sym * n_syms_used
        end = min(end, len(bits_full))
        return bits_full[start:end]
    
    bps1 = BITS_PER_SYMBOL[modulation1.upper()]
    bps2 = BITS_PER_SYMBOL[modulation2.upper()]
    
    b1_ref = slice_bits_to_match_syms(bits1, len(true_symbols1), bps1)
    b2_ref = slice_bits_to_match_syms(bits2, len(true_symbols2), bps2)
    
    # 解调预测符号
    b1_hat = demodulate(pred_symbols1_aligned, modulation1)
    b2_hat = demodulate(pred_symbols2_aligned, modulation2)
    
    # 计算SER：按符号分组比较比特组合（与test_sim_SignalSeparator.py一致）
    Lb1 = min(len(b1_hat), len(b1_ref))
    Lb2 = min(len(b2_hat), len(b2_ref))
    Lb1 = (Lb1 // bps1) * bps1
    Lb2 = (Lb2 // bps2) * bps2
    
    if Lb1 > 0 and Lb2 > 0:
        n_syms1 = Lb1 // bps1
        n_syms2 = Lb2 // bps2
        b1_ref_syms = b1_ref[:Lb1].reshape(n_syms1, bps1)
        b1_hat_syms = b1_hat[:Lb1].reshape(n_syms1, bps1)
        b2_ref_syms = b2_ref[:Lb2].reshape(n_syms2, bps2)
        b2_hat_syms = b2_hat[:Lb2].reshape(n_syms2, bps2)
        sym_errors1 = np.any(b1_ref_syms != b1_hat_syms, axis=1)
        sym_errors2 = np.any(b2_ref_syms != b2_hat_syms, axis=1)
        ser1 = float(np.mean(sym_errors1)) if n_syms1 > 0 else 1.0
        ser2 = float(np.mean(sym_errors2)) if n_syms2 > 0 else 1.0
    else:
        ser1 = 1.0
        ser2 = 1.0
    
    ser = 0.5 * (ser1 + ser2)
    
    print(f"  信号1 ({modulation1}) SER: {ser1:.6f}")
    print(f"  信号2 ({modulation2}) SER: {ser2:.6f}")
    print(f"  平均 SER: {ser:.6f}")
    
    # 10. 可视化（使用对齐后的符号）
    print(f"\n步骤10: 生成可视化...")
    plot_results(
        mix_signal, signal1, signal2,
        pred_signal1, pred_signal2,
        true_symbols1, true_symbols2,
        pred_symbols1_aligned, pred_symbols2_aligned,
        ser1, ser2, modulation1, modulation2,
        amp_ratio=amp_ratio,
        save_path="example_results.png"
    )
    
    print(f"\n" + "="*60)
    print("完成！")
    print("="*60)
    print(f"结果图: example_results.png")
    print(f"信号1 SER: {ser1:.6f}")
    print(f"信号2 SER: {ser2:.6f}")

if __name__ == '__main__':
    main()

