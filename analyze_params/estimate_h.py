import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from scipy import signal as scipy_signal
from scipy import stats
import os
from compensation import costas_loop, constellation_analysis
from utils_compensation import evaluate_clustering_quality, compare_psd, compare_evm, analyze_error_signal,estimate_pulse_response_ls
import math
from scipy.optimize import curve_fit

# 参数设置：在此处填写每种调制对应的数据文件路径和采样参数
# 例：data_configs = {
#   '8PSK': {'file': '/path/to/8psk.dat', 'fs': 12e6, 'sps': 16},
#   'QPSK': {'file': '/path/to/qpsk.dat', 'fs': 12e6, 'sps': 16},
# }
data_configs = {
    # 填入你的文件路径和采样参数
    '8PSK': {'file': '/nas/datasets/yixin/PCMA/sim_data/splited_data_8psk.pth', 'fs': 12e6, 'sps': 16},
    'QPSK': {'file': '/nas/datasets/yixin/PCMA/sim_data/splited_data_qpsk.pth', 'fs': 12e6, 'sps': 8},
    '16QAM': {'file': '/nas/datasets/yixin/PCMA/sim_data/splited_data_16qam.pth', 'fs': 12e6, 'sps': 16},
    'QPSK_10':{'file': '/nas/datasets/yixin/PCMA/sim_data/splited_data_qpsk_10x3072.pth', 'fs': 12e6, 'sps': 8},
    'QPSK_100':{'file': '/nas/datasets/yixin/PCMA/sim_data/splited_data_qpsk_100x3072.pth', 'fs': 12e6, 'sps': 8},
    'QPSK_500':{'file': '/nas/datasets/yixin/PCMA/sim_data/splited_data_qpsk_500x3072.pth', 'fs': 12e6, 'sps': 8},
    'QPSK_1000':{'file': '/nas/datasets/yixin/PCMA/sim_data/splited_data_qpsk_1000x3072.pth', 'fs': 12e6, 'sps': 8},
    '8PSK_10':{'file': '/nas/datasets/yixin/PCMA/sim_data/splited_data_8psk_10x3072.pth', 'fs': 12e6, 'sps': 16},
    '16QAM_10':{'file': '/nas/datasets/yixin/PCMA/sim_data/splited_data_16qam_10x3072.pth', 'fs': 12e6, 'sps': 16},
    'QPSK_50':{'file': '/nas/datasets/yixin/PCMA/sim_data/splited_data_qpsk_50x3072.pth', 'fs': 12e6, 'sps': 8},
    '8PSK_50':{'file': '/nas/datasets/yixin/PCMA/sim_data/splited_data_8psk_50x3072.pth', 'fs': 12e6, 'sps': 16},
    '8PSK_500':{'file': '/nas/datasets/yixin/PCMA/sim_data/splited_data_8psk_500x3072.pth', 'fs': 12e6, 'sps': 16},
    '16QAM_50':{'file': '/nas/datasets/yixin/PCMA/sim_data/splited_data_16qam_50x3072.pth', 'fs': 12e6, 'sps': 16},
    '16QAM_500':{'file': '/nas/datasets/yixin/PCMA/sim_data/splited_data_16qam_500x3072.pth', 'fs': 12e6, 'sps': 16}
}

# 选择本次要处理的调制类型（请改为你想处理的键，例如 '8PSK'）
selected_mod = '8PSK_500'

if selected_mod not in data_configs:
    raise ValueError(f"selected_mod '{selected_mod}' not found in data_configs. Please update the config at top of the script.")

cfg = data_configs[selected_mod]
file = cfg['file']
fs = cfg.get('fs', 12e6)  # 采样频率
sps = cfg.get('sps', 16)

# 输出目录（按调制方式分组）
out_dir = os.path.join('./src/estimate_h', selected_mod)
os.makedirs(out_dir, exist_ok=True)

data = np.fromfile(file, dtype=np.int16)
complex_data = data[0::2].astype(np.float32) + 1j * data[1::2].astype(np.float32)

complex_data_compensated, phase_history = costas_loop(complex_data, loop_bandwidth=0.00001, sps=sps)


best_offset = -1
best_score = -np.inf
best_history = None



print("开始评估不同offset的聚类效果...")

for offset in range(sps):
    symbol_compensated = complex_data_compensated[offset::sps]
    
    # --- 新增：计算当前offset的得分 ---
    current_score = evaluate_clustering_quality(symbol_compensated)
    print(f"Offset {offset}: 聚类质量得分 = {current_score:.4f}")

    # --- 新增：更新最佳offset ---
    if current_score > best_score:
        best_score = current_score
        best_offset = offset
        best_history = phase_history
    # --- 原有的绘图代码 ---
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.scatter(symbol_compensated.real, symbol_compensated.imag, alpha=0.5)
    plt.title(f'Compensated Constellation Diagram (Offset {offset}, Score: {current_score:.2f})')
    plt.xlabel('In-phase')
    plt.ylabel('Quadrature')
    plt.grid(True)
    plt.axis('equal')

    plt.subplot(2, 1, 2)
    plt.plot(phase_history)
    plt.title('Phase History')
    plt.xlabel('Sample Index')
    plt.ylabel('Phase (radians)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'compesation_quadrature_offset{offset}.png'))
    plt.close() # 建议加上plt.close()，防止内存泄漏

print(f"\n评估完成！最佳 offset 是: {best_offset}，对应得分: {best_score:.4f}")

# 现在，你可以使用 best_offset 来获取最佳符号序列
best_symbols = complex_data_compensated[best_offset::sps]

# 假设 phase_history 是一个长度为 N 的数组
time_axis = np.arange(len(best_history))

# 使用 np.polyfit 进行一次多项式拟合（即直线拟合）
# 它会返回斜率和截距
slope, intercept = np.polyfit(time_axis, best_history, 1)

print(f"Phase History 的线性拟合结果:")
print(f"斜率 (弧度/样本): {slope}")
print(f"截距 (弧度): {intercept}")

# 你也可以计算出对应的频率偏移
# 角频率偏移 rad/sample
angular_freq_offset = slope
# 频率偏移 Hz
# fs 是采样频率
freq_offset_Hz = slope * fs / (2 * np.pi)
print(f"\n估算出的频率偏移: {freq_offset_Hz:.2f} Hz")

known_freq_offset_Hz = freq_offset_Hz

print(f"已知的频率偏移: {known_freq_offset_Hz} Hz")
print(f"采样频率: {fs/1e6} MHz")

# 1. 获取初始相位
# 我们从 phase_history 的第一个点获取。在真实场景中，这就是 costas_loop 的输出。
# 在我们的模拟中，phase_history 就是 phi_to_apply
phase_history = best_history
initial_phase = phase_history[0]
print(f"获取的初始相位: {initial_phase:.4f} 弧度")

# 2. 创建时间轴
num_samples = len(complex_data_compensated)
time_axis = np.arange(num_samples)

# 3. 计算需要重新引入的相位序列
# 这就是我们想要“加回去”的相位
phi_to_reintroduce = 2 * np.pi * known_freq_offset_Hz * time_axis / fs + initial_phase

# 4. 构建复指数校正因子
correction_factor = np.exp(1j * phi_to_reintroduce)

# 5. 执行逆补偿，恢复原始信号
reconstructed_complex_data = complex_data_compensated * correction_factor

# ==============================================================================
#                          验证结果
# ==============================================================================

print("\n--- 验证恢复效果 ---")


plt.figure(figsize=(15, 10))

# 1. 对比原始信号和恢复信号的频谱（应该几乎重合）
plt.subplot(3, 1, 1)
frequencies = np.fft.fftfreq(num_samples, 1/fs)
original_spectrum = np.fft.fft(complex_data)
reconstructed_spectrum = np.fft.fft(reconstructed_complex_data)

plt.plot(frequencies[:num_samples//2]/1e3, 20*np.log10(np.abs(original_spectrum[:num_samples//2])), label='Original Signal Spectrum')
plt.plot(frequencies[:num_samples//2]/1e3, 20*np.log10(np.abs(reconstructed_spectrum[:num_samples//2])), '--', label='Reconstructed Signal Spectrum', alpha=0.8)
plt.title('Spectrum Comparison')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.grid(True)

# 2. 对比补偿后的信号频谱（应该是一个干净的尖峰在0Hz附近）
plt.subplot(3, 1, 2)
compensated_spectrum = np.fft.fft(complex_data_compensated)
plt.plot(frequencies[:num_samples//2]/1e3, 20*np.log10(np.abs(compensated_spectrum[:num_samples//2])), label='Compensated Signal Spectrum')
plt.title('Compensated Signal Spectrum (Should be centered at 0 Hz)')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.grid(True)

# 3. 对比星座图（只取一个offset的点）
offset = best_offset # 假设最佳offset是0
original_symbols_plot = complex_data[offset::sps]
reconstructed_symbols_plot = reconstructed_complex_data[offset::sps]
compensated_symbols_plot = complex_data_compensated[offset::sps]

plt.subplot(3, 1, 3)
plt.scatter(original_symbols_plot.real, original_symbols_plot.imag, alpha=0.5, label='Original Symbols', s=10)
plt.scatter(reconstructed_symbols_plot.real, reconstructed_symbols_plot.imag, marker='x', alpha=0.7, label='Reconstructed Symbols', s=15)
plt.title('Constellation Comparison (Original vs Reconstructed)')
plt.xlabel('In-phase')
plt.ylabel('Quadrature')
plt.legend()
plt.grid(True)
plt.axis('equal')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'reconstruction_verification.png'))

freq_error, psd_nmse = compare_psd(complex_data, reconstructed_complex_data, fs)

print(f"\n频谱对比结果:")
print(f"频率偏移误差: {freq_error:.2f} Hz")
print(f"主瓣归一化均方误差 (NMSE): {psd_nmse:.6f}")

evm_orig, evm_recon = compare_evm(complex_data, reconstructed_complex_data, sps, best_offset)
print(f"\nEVM 对比结果:")
print(f"原始信号 EVM: {evm_orig:.2f} %")
print(f"恢复信号 EVM: {evm_recon:.2f} %")

# 估计成型滤波器的脉冲响应
# ==============================================================================
#                      1. 符号判决
# ==============================================================================
print("步骤 1: 从补偿后的信号中判决符号并针对不同调制方式处理...")

# 从补偿后的数据中抽取符号（统一使用best_offset和sps）
symbols_rx = complex_data_compensated[best_offset::sps]
symbols_rx_normed = symbols_rx / np.sqrt(np.mean(np.abs(symbols_rx)**2))

def get_constellation(mod_type):
    """返回给定调制方式的理想星座点。"""
    if mod_type.upper() == 'QPSK' or mod_type.upper() == 'QPSK_10' or mod_type.upper() == 'QPSK_100' or mod_type.upper() == 'QPSK_500' or mod_type.upper() == 'QPSK_1000' or mod_type.upper() == 'QPSK_50':
        pts = np.array([1+1j, 1-1j, -1+1j, -1-1j], dtype=np.complex64)
    elif mod_type.upper() == '8PSK' or mod_type.upper() == '8-PSK' or mod_type.upper() == '8PSK_10' or mod_type.upper() == '8PSK_50' or mod_type.upper() == '8PSK_500':
        angles = 2 * np.pi * np.arange(8) / 8.0 + np.pi/8  # 相位偏移 pi/8
        pts = np.exp(1j * angles)
    elif mod_type.upper() == '16QAM' or mod_type.upper() == '16-QAM' or mod_type.upper() == '16QAM_10' or mod_type.upper() == '16QAM_50' or mod_type.upper() == '16QAM_500':
        levels = np.array([-3, -1, 1, 3], dtype=np.float32)
        xv, yv = np.meshgrid(levels, levels)
        pts = (xv.flatten() + 1j * yv.flatten()).astype(np.complex64)
        # 归一化到均方值为1
        pts = pts / np.sqrt(np.mean(np.abs(pts)**2))
    else:
        raise ValueError(f'Unsupported modulation: {mod_type}')
    # 归一化为单位平均功率（除非是PSK已经在单位圆上）
    if mod_type.upper() == 'QPSK' or 'PSK' in mod_type.upper():
        pts = pts / np.sqrt(np.mean(np.abs(pts)**2))
    return pts

def decide_symbols(symbols, constellation_pts):
    """对接收符号进行最近邻判决，返回判决后的符号序列。"""
    # 将符号与每个星座点的距离计算并取最小
    pts = constellation_pts.reshape((-1, 1))
    # distances: (num_pts, num_symbols)
    # 使用广播计算
    distances = np.abs(symbols.reshape((1, -1)) - pts)
    idx = np.argmin(distances, axis=0)
    decided = pts[idx].flatten()
    return decided

mod = selected_mod
print(f"\n处理调制方式: {mod}")
out_dir = os.path.join('./src/estimate_h', mod)
os.makedirs(out_dir, exist_ok=True)

constellation_pts = get_constellation(mod)
symbols_decided = decide_symbols(symbols_rx_normed, constellation_pts)
print(f"判决了 {len(symbols_decided)} 个符号（{mod}）。")

# 可视化判决前后的星座图
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(symbols_rx_normed.real, symbols_rx_normed.imag, alpha=0.6, label='Received Symbols')
plt.title('Constellation Before Decision')
plt.xlabel('In-phase')
plt.ylabel('Quadrature')
plt.grid(True)
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.scatter(symbols_decided.real, symbols_decided.imag, alpha=0.8, c='red', marker='x', label='Decided Symbols')
plt.scatter(constellation_pts.real, constellation_pts.imag, c='k', marker='o', s=40, label='Ideal Constellation')
plt.title('Constellation After Decision')
plt.xlabel('In-phase')
plt.ylabel('Quadrature')
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'symbol_decision_comparison.png'))
plt.close()

# 2. 生成理想方波信号（根据判决符号）
ideal_signal = np.zeros_like(complex_data_compensated)
ideal_signal[best_offset::sps] = symbols_decided

# 小段可视化
plt.figure(figsize=(15, 6))
plot_range = slice(0, 80)
plt.plot(ideal_signal[plot_range].real, '-o', label='Ideal Signal (Real Part)')
plt.plot(complex_data_compensated[plot_range].real/np.mean(np.abs(complex_data_compensated)), '-x', label='Received Signal (Real Part)')
plt.title(f'Ideal vs. Received Signal (Time Domain Snapshot) - {mod}')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(out_dir, 'ideal_vs_received_time_domain.png'))
plt.close()

# 3. 反卷积估计脉冲响应
print("\n步骤 3: 通过时域反卷积估计脉冲响应...")
N_filter = 8 * sps
A_rows = len(complex_data_compensated) - N_filter + 1
A = np.zeros((A_rows, N_filter), dtype=np.complex64)
for n in range(A_rows):
    A[n, :] = ideal_signal[n:n + N_filter]

AA = np.vstack([np.real(A), np.imag(A)])
xxI = np.real(complex_data_compensated)[N_filter//2 : N_filter//2 + A_rows]
xxQ = np.imag(complex_data_compensated)[N_filter//2 : N_filter//2 + A_rows]
xxxx = np.hstack([xxI, xxQ])

# 求解最小二乘问题
h = np.linalg.lstsq(AA, xxxx, rcond=None)[0]
h_estimated_ls = np.flipud(h)
print(f"估计出长度为 {len(h_estimated_ls)} 的脉冲响应。")

def theoretical_rc(num_taps=64, alpha=0.3, sps_local=8):
    t = np.arange(-num_taps//2, num_taps//2) / sps_local
    h = np.sinc(t) * np.cos(np.pi * alpha * t) / (1 - (2 * alpha * t)**2)
    h[t == 0] = 1.0
    h[np.abs(1 - (2 * alpha * t)**2) < 1e-6] = np.pi/4 * np.sinc(1/(2*alpha))
    return h / np.max(np.abs(h))

if len(h_estimated_ls) > 0:
    print(f"成功估计出长度为 {len(h_estimated_ls)} 的脉冲响应（{mod}）。")

    # 绘图与分析
    w, h_response_ls = scipy_signal.freqz(h_estimated_ls, worN=4096, fs=fs)
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(h_estimated_ls)
    plt.title('Estimated Impulse Response (Time-Domain LS, Normalized)')
    plt.xlabel('Taps')
    plt.ylabel('Amplitude')
    plt.grid(True)

    num_taps = len(h_estimated_ls)
    # 归一化已估计的脉冲响应
    h_measured = h_estimated_ls.astype(np.complex128)
    if np.max(np.abs(h_measured)) != 0:
        h_measured_norm = h_measured / np.max(np.abs(h_measured))
    else:
        h_measured_norm = h_measured

    # 增加: 定义 RRC 生成函数（时域表达式），并对 RC / RRC 的 roll-off beta 进行扫描
    def theoretical_rrc(num_taps=64, alpha=0.3, sps_local=8):
        # t 以符号为单位（samples/sps）中心对称
        t = np.arange(-num_taps//2, num_taps//2) / float(sps_local)
        h = np.zeros_like(t, dtype=np.float64)
        pi = np.pi
        for i, ti in enumerate(t):
            if abs(ti) < 1e-12:
                h[i] = 1.0 - alpha + (4*alpha/pi)
            elif abs(abs(4*alpha*ti) - 1.0) < 1e-12:
                # 处理分母接近0的位置
                # 根据极限值公式
                h[i] = (alpha/np.sqrt(2)) * ((1 + 2/pi) * np.sin(pi/(4*alpha)) + (1 - 2/pi) * np.cos(pi/(4*alpha)))
            else:
                num = np.sin(pi*ti*(1-alpha)) + 4*alpha*ti*np.cos(pi*ti*(1+alpha))
                den = pi*ti*(1 - (4*alpha*ti)**2)
                h[i] = num/den
        # 归一化幅值到峰值为1，方便比较形状
        if np.max(np.abs(h)) != 0:
            h = h / np.max(np.abs(h))
        return h

    # 复用已有的 theoretical_rc，但保证对特殊点处理更稳定
    def theoretical_rc_safe(num_taps=64, alpha=0.3, sps_local=8):
        t = np.arange(-num_taps//2, num_taps//2) / float(sps_local)
        h = np.zeros_like(t, dtype=np.float64)
        for i, ti in enumerate(t):
            denom = 1 - (2 * alpha * ti) ** 2
            if abs(ti) < 1e-12:
                h[i] = 1.0
            elif abs(denom) < 1e-8:
                # 极限值
                h[i] = (np.pi/4) * np.sinc(1.0/(2*alpha))
            else:
                h[i] = np.sinc(ti) * np.cos(np.pi * alpha * ti) / denom
        if np.max(np.abs(h)) != 0:
            h = h / np.max(np.abs(h))
        return h

    def align_and_compute_nmse(reference, measured):
        # reference, measured: 1D arrays
        # 通过互相关寻找最佳延迟，再进行线性缩放拟合，返回最小 NMSE 和对应延迟与缩放
        ref = np.asarray(reference)
        meas = np.asarray(measured)
        # 互相关（绝对值），full 模式
        corr = np.abs(np.correlate(meas, ref, mode='full'))
        peak_idx = np.argmax(corr)
        # 计算延迟，使得 ref 被移位后与 meas 对齐
        delay = peak_idx - (len(ref) - 1)

        # 将 reference 移动到 measured 的坐标中，并截取/填零以匹配长度
        if delay >= 0:
            ref_shifted = np.concatenate([np.zeros(delay, dtype=ref.dtype), ref])
            ref_shifted = ref_shifted[:len(meas)]
        else:
            # delay < 0: reference 向右移动（对 meas 来说就是提前）
            ref_shifted = ref[-delay:]
            if len(ref_shifted) < len(meas):
                ref_shifted = np.concatenate([ref_shifted, np.zeros(len(meas)-len(ref_shifted), dtype=ref.dtype)])
            else:
                ref_shifted = ref_shifted[:len(meas)]

        # 最优线性缩放（最小二乘）： scale = <meas,ref'> / <ref',ref'>
        denom = np.vdot(ref_shifted, ref_shifted)
        if np.abs(denom) < 1e-12:
            scale = 0.0
        else:
            scale = np.vdot(meas, ref_shifted) / denom

        residual = meas - scale * ref_shifted
        nmse = np.linalg.norm(residual) ** 2 / (np.linalg.norm(meas) ** 2 + 1e-18)
        return nmse, delay, scale, ref_shifted

    # 扫描 beta 值
    beta_list = np.linspace(0.01, 0.99, 99)
    nmse_rc = []
    nmse_rrc = []
    detailed_rc = {}
    detailed_rrc = {}

    for beta in beta_list:
        h_rc = theoretical_rc_safe(num_taps, beta, sps)
        nmse_val, delay_val, scale_val, ref_shifted = align_and_compute_nmse(h_rc, h_measured_norm)
        nmse_rc.append(nmse_val)
        detailed_rc[beta] = (nmse_val, delay_val, scale_val, ref_shifted)

        h_rrc = theoretical_rrc(num_taps, beta, sps)
        nmse_val2, delay_val2, scale_val2, ref_shifted2 = align_and_compute_nmse(h_rrc, h_measured_norm)
        nmse_rrc.append(nmse_val2)
        detailed_rrc[beta] = (nmse_val2, delay_val2, scale_val2, ref_shifted2)

    nmse_rc = np.array(nmse_rc)
    nmse_rrc = np.array(nmse_rrc)

    # 选择最小 NMSE 的 beta
    best_idx_rc = np.argmin(nmse_rc)
    best_beta_rc = beta_list[best_idx_rc]
    best_nmse_rc = nmse_rc[best_idx_rc]
    best_info_rc = detailed_rc[best_beta_rc]

    best_idx_rrc = np.argmin(nmse_rrc)
    best_beta_rrc = beta_list[best_idx_rrc]
    best_nmse_rrc = nmse_rrc[best_idx_rrc]
    best_info_rrc = detailed_rrc[best_beta_rrc]

    print(f"RC best beta: {best_beta_rc:.3f}, NMSE: {best_nmse_rc:.6e}, delay: {best_info_rc[1]}, scale: {best_info_rc[2]:.3f}")
    print(f"RRC best beta: {best_beta_rrc:.3f}, NMSE: {best_nmse_rrc:.6e}, delay: {best_info_rrc[1]}, scale: {best_info_rrc[2]:.3f}")

    # ==================== RC/RRC NMSE 曲线 & 时域对比 ====================
    plt.figure(figsize=(12, 6))
    plt.plot(beta_list, nmse_rc, label='RC NMSE')
    plt.plot(beta_list, nmse_rrc, label='RRC NMSE')
    plt.xlabel('Roll-off (beta)')
    plt.ylabel('NMSE')
    plt.title('RC vs RRC NMSE over beta')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'rc_rrc_beta_scan_nmse.png'))
    plt.close()

    # 绘制最佳 RC/RRC 与测量滤波器对比
    plt.figure(figsize=(12, 6))
    # RC best
    h_rc_best = theoretical_rc_safe(num_taps, best_beta_rc, sps)
    _, _, scale_rc, ref_shifted_rc = best_info_rc
    # RRC best
    h_rrc_best = theoretical_rrc(num_taps, best_beta_rrc, sps)
    _, _, scale_rrc, ref_shifted_rrc = best_info_rrc

    # 绘制测量与最佳理论（经缩放、对齐）
    plt.subplot(2, 1, 1)
    plt.plot(np.real(h_measured_norm), label='Measured (real)')
    plt.plot(np.real(scale_rc * ref_shifted_rc), '--', label=f'Best RC beta={best_beta_rc:.3f}')
    plt.plot(np.real(scale_rrc * ref_shifted_rrc), ':', label=f'Best RRC beta={best_beta_rrc:.3f}')
    plt.title('Impulse Response Comparison (Real Part)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(np.imag(h_measured_norm), label='Measured (imag)')
    plt.plot(np.imag(scale_rc * ref_shifted_rc), '--', label=f'Best RC beta={best_beta_rc:.3f}')
    plt.plot(np.imag(scale_rrc * ref_shifted_rrc), ':', label=f'Best RRC beta={best_beta_rrc:.3f}')
    plt.title('Impulse Response Comparison (Imag Part)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'best_rc_rrc_comparison.png'))
    plt.close()

    # ======================================================
    # 从这里开始：使用 NMSE 最小的 RRC 滤波器做重建和噪声估计
    # ======================================================

    # 构造用于重建/噪声估计的 RRC 滤波器（峰值已归一）
    h_rrc_used = theoretical_rrc(num_taps, best_beta_rrc, sps)

    def find_optimal_delay(reference, measured, max_delay=10):
        correlations = []
        for delay in range(-max_delay, max_delay + 1):
            if delay >= 0:
                ref_shifted = reference[delay:]
                meas_trimmed = measured[:len(ref_shifted)]
            else:
                ref_shifted = reference[:delay] if delay != 0 else reference
                meas_trimmed = measured[-delay:len(ref_shifted) - delay] if delay != 0 else measured
            if len(ref_shifted) > 0 and len(ref_shifted) == len(meas_trimmed):
                correlation = np.corrcoef(ref_shifted, meas_trimmed)[0, 1]
                correlations.append((delay, correlation))
        optimal_delay, max_corr = max(correlations, key=lambda x: abs(x[1]))
        return optimal_delay

    def apply_delay_signal(signal, delay):
        if delay > 0:
            return np.concatenate([np.zeros(delay, dtype=signal.dtype), signal[:-delay]])
        elif delay < 0:
            return signal[-delay:]
        else:
            return signal

    # 对齐 ideal_signal 与接收信号
    optimal_delay = find_optimal_delay(ideal_signal, complex_data_compensated)
    print(f"最优延迟量: {optimal_delay} 个样本")
    ideal_signal_aligned = apply_delay_signal(ideal_signal, optimal_delay)

    # 用 RRC 滤波器卷积得到重建信号（先不缩放）
    recon_tmp = scipy_signal.convolve(ideal_signal_aligned, h_rrc_used, mode='same')

    # 用最小二乘求一个复数增益，使 recon_tmp 尽量拟合接收信号
    denom = np.vdot(recon_tmp, recon_tmp)
    if np.abs(denom) < 1e-12:
        gain = 1.0 + 0j
    else:
        gain = np.vdot(complex_data_compensated, recon_tmp) / denom

    reconstructed_rx_signal = gain * recon_tmp

    # 将之前估计出的 CFO/初始相位应用到重构信号上，以便与之前重新引入相位的接收信号对齐
    # 如果 correction_factor 已定义（脚本前面有计算），则直接使用；否则按估计频偏和初始相位重建
    try:
        cf = correction_factor
    except NameError:
        cf = np.exp(1j * (2 * np.pi * known_freq_offset_Hz * np.arange(len(reconstructed_rx_signal)) / fs + initial_phase))

    # 确保长度匹配
    if len(cf) != len(reconstructed_rx_signal):
        cf = np.exp(1j * (2 * np.pi * known_freq_offset_Hz * np.arange(len(reconstructed_rx_signal)) / fs + initial_phase))

    reconstructed_rx_signal_cfo = reconstructed_rx_signal * cf

    # 用带 CFO 的重构信号与带 CFO 的重建接收信号比较以得到噪声
    if 'reconstructed_complex_data' in globals():
        ref_rx = reconstructed_complex_data
    else:
        ref_rx = complex_data_compensated * cf

    noise = reconstructed_rx_signal_cfo - ref_rx

    snr = 10 * math.log10(
        np.mean(np.abs(reconstructed_rx_signal_cfo) ** 2) / (np.mean(np.abs(noise) ** 2) + 1e-18)
    )
    print(f'SNR (using best-RRC h, with CFO applied): {snr:.2f} dB')

    # ========== 验证 & 噪声分析（下面这部分基本沿用你原来的，只是 h 换成了 RRC） ==========
    plot_range = slice(0, 160)
    plt.figure(figsize=(12, 8))

    plt.subplot(1, 2, 1)
    plt.plot(ref_rx[plot_range].real, label='Original Received Signal (Real, with CFO)', linewidth=2)
    plt.plot(reconstructed_rx_signal_cfo[plot_range].real, '--', label='Reconstructed Signal (Real, with CFO)', linewidth=1.5)
    plt.plot(noise[plot_range].real, label='Noise (Real)')
    plt.title('Validation: Reconstructed vs Original (Real Part)')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    # 4. 分析信道噪声的统计分布（整体）
    print("\n分析信道噪声的统计分布...")

    # 分离I路和Q路噪声
    noise_I = noise.real
    noise_Q = noise.imag

    plt.subplot(1, 2, 2)
    num_bins = 30
    alpha = 0.6

    plt.hist(noise_I, bins=num_bins, density=True, alpha=alpha, label='I路')
    plt.hist(noise_Q, bins=num_bins, density=True, alpha=alpha, label='Q路')

    mu_I, std_I = stats.norm.fit(noise_I)
    mu_Q, std_Q = stats.norm.fit(noise_Q)

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p_I = stats.norm.pdf(x, mu_I, std_I)
    p_Q = stats.norm.pdf(x, mu_Q, std_Q)

    plt.plot(x, p_I, 'k-', linewidth=2, label=f'I路拟合 (μ={mu_I:.2f}, σ={std_I:.2f})')
    plt.plot(x, p_Q, 'k--', linewidth=2, label=f'Q路拟合 (μ={mu_Q:.2f}, σ={std_Q:.2f})')

    plt.title('信道噪声分布', fontsize=16)
    plt.xlabel('噪声幅度', fontsize=12)
    plt.ylabel('概率密度', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pulse_response_estimation_ls.png'))
    plt.close()

    # ==========================
    # 噪声分布与功率谱分析（整体）
    # ==========================
    print("\n进一步分析噪声的分布和功率谱...")

    plt.figure(figsize=(12, 5))

    # 1. 整体噪声直方图
    plt.subplot(1, 3, 1)
    plt.hist(noise_I, bins=num_bins, density=True, alpha=alpha, label='I路')
    plt.hist(noise_Q, bins=num_bins, density=True, alpha=alpha, label='Q路')

    mu_I, std_I = stats.norm.fit(noise_I)
    mu_Q, std_Q = stats.norm.fit(noise_Q)

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p_I = stats.norm.pdf(x, mu_I, std_I)
    p_Q = stats.norm.pdf(x, mu_Q, std_Q)

    plt.plot(x, p_I, 'k-', linewidth=2, label=f'I路拟合 (μ={mu_I:.2f}, σ={std_I:.2f})')
    plt.plot(x, p_Q, 'k--', linewidth=2, label=f'Q路拟合 (μ={mu_Q:.2f}, σ={std_Q:.2f})')

    plt.title('信道噪声分布', fontsize=16)
    plt.xlabel('噪声幅度', fontsize=12)
    plt.ylabel('概率密度', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # 2. 噪声功率谱
    plt.subplot(1, 3, 2)
    noise_psd_freq, noise_psd = scipy_signal.welch(noise, fs=fs, nperseg=4096)
    plt.semilogy(noise_psd_freq / 1e3, noise_psd, label='Noise PSD')
    plt.title('噪声功率谱')
    plt.xlabel('频率 (kHz)')
    plt.ylabel('功率谱密度')
    plt.legend()
    plt.grid(True)

    # 3. 信号功率谱
    plt.subplot(1, 3, 3)
    signal_psd_freq, signal_psd = scipy_signal.welch(complex_data_compensated, fs=fs, nperseg=4096)
    plt.semilogy(signal_psd_freq / 1e3, signal_psd, label='Signal PSD', color='g')
    plt.title('信号功率谱')
    plt.xlabel('频率 (kHz)')
    plt.ylabel('功率谱密度')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'noise_distribution_and_psd.png'))
    plt.close()

    # ==========================================================
    # 16QAM 情况下，按星座点分类统计噪声直方图（保持不变）
    # ==========================================================
    if selected_mod.upper() in ['16QAM', '16-QAM']:
        print("\n16QAM：按星座点分类统计符号处噪声分布...")

        noise_symbols = noise[best_offset::sps]
        assert len(noise_symbols) == len(symbols_decided), \
            "noise_symbols 与 symbols_decided 长度不一致，请检查 sps / offset 对齐。"

        pts = constellation_pts
        num_pts = len(pts)

        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.ravel()

        for i, pt in enumerate(pts):
            ax = axes[i]
            idx = (symbols_decided == pt)
            n_this = np.sum(idx)
            if n_this == 0:
                ax.set_title(f'Pt {i}: ({pt.real:.1f},{pt.imag:.1f})\nN=0')
                ax.grid(True, linestyle='--', alpha=0.4)
                continue

            noise_i = noise_symbols[idx].real
            noise_q = noise_symbols[idx].imag

            ax.hist(noise_i, bins=20, density=True, alpha=0.5, label='I')
            ax.hist(noise_q, bins=20, density=True, alpha=0.5, label='Q')

            mu_i, std_i = stats.norm.fit(noise_i)
            mu_q, std_q = stats.norm.fit(noise_q)

            xmin = min(noise_i.min(), noise_q.min())
            xmax = max(noise_i.max(), noise_q.max())
            x = np.linspace(xmin, xmax, 100)
            p_i = stats.norm.pdf(x, mu_i, std_i)
            p_q = stats.norm.pdf(x, mu_q, std_q)

            ax.plot(x, p_i, 'k-', linewidth=1.5)
            ax.plot(x, p_q, 'k--', linewidth=1.5)

            ax.set_title(
                f'Pt {i}: ({pt.real:.1f},{pt.imag:.1f}), N={n_this}\n'
                f'I: μ={mu_i:.2f},σ={std_i:.2f}; Q: μ={mu_q:.2f},σ={std_q:.2f}',
                fontsize=9
            )
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.set_xlabel('噪声幅度')
            ax.set_ylabel('概率密度')

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        fig.suptitle('16QAM 符号采样点噪声分布（按星座点分类）', fontsize=16)

        plt.tight_layout(rect=[0, 0, 0.98, 0.96])
        plt.savefig(os.path.join(out_dir, 'noise_hist_per_constellation_16qam.png'))
        plt.close()
