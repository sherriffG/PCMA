import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from scipy import stats
from compensation import costas_loop, constellation_analysis
from utils_compensation import evaluate_clustering_quality, compare_psd, compare_evm, analyze_error_signal,estimate_pulse_response_ls
import math
from scipy.optimize import curve_fit
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False
# 参数设置
path = '/nas/datasets/LYX/PCMA/QPSK_16/1050/'
filename = 'splited_data.dat'

file = path + filename
file = '/nas/datasets/yixin/PCMA/sim_data/splited_data_qpsk_1000x3072.pth'

fs = 12e6  # 采样频率
sps = 8

data = np.fromfile(file, dtype=np.int16)
complex_data = data[0::2].astype(np.float32) + 1j * data[1::2].astype(np.float32)

complex_data_compensated, phase_history = costas_loop(complex_data, loop_bandwidth=0.00001, sps=8)


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
    plt.savefig(f'./src/splited_data/compesation_quadrature_offset{offset}.png')
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
sps = 8
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
plt.savefig('./src/splited_data/reconstruction_verification.png')
plt.show()

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
print("步骤 1: 从补偿后的信号中判决符号...")

# 从补偿后的数据中抽取符号
symbols_rx = complex_data_compensated[best_offset::sps]

# 定义理想的QPSK星座点
ideal_qpsk_constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j])

# 对每个接收到的符号进行判决
symbols_decided = np.zeros_like(symbols_rx)
for i, sym in enumerate(symbols_rx):
    real = sym.real
    imag = sym.imag
    if real >= 0 and imag >= 0:
        symbols_decided[i] = ideal_qpsk_constellation[0]
    elif real >= 0 and imag < 0:
        symbols_decided[i] = ideal_qpsk_constellation[1]
    elif real < 0 and imag >= 0:
        symbols_decided[i] = ideal_qpsk_constellation[2]
    elif real < 0 and imag < 0:
        symbols_decided[i] = ideal_qpsk_constellation[3]

print(f"判决了 {len(symbols_decided)} 个符号。")

# 可视化判决前后的星座图
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(symbols_rx.real, symbols_rx.imag, alpha=0.6, label='Received Symbols')
plt.title('Constellation Before Decision')
plt.xlabel('In-phase')
plt.ylabel('Quadrature')
plt.grid(True)
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.scatter(symbols_decided.real, symbols_decided.imag, alpha=0.8, c='red', marker='x', label='Decided Symbols')
plt.title('Constellation After Decision')
plt.xlabel('In-phase')
plt.ylabel('Quadrature')
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.savefig('./src/splited_data/symbol_decision_comparison.png')


# ==============================================================================
#                      2. 生成理想QPSK方波信号
# ==============================================================================
print("\n步骤 2: 根据判决符号生成理想方波信号...")

# 创建一个与原始信号等长的全零数组
ideal_signal = np.zeros_like(complex_data_compensated)

# 将判决出的符号填充到对应的采样点上
ideal_signal[best_offset::sps] = symbols_decided

print(f"生成长度为 {len(ideal_signal)} 的理想信号。")

# 可视化理想信号和实际信号的一小段
plt.figure(figsize=(15, 6))
plot_range = slice(0, 80) # 绘制前80个采样点
plt.plot(ideal_signal[plot_range].real, '-o', label='Ideal Signal (Real Part)')
plt.plot(complex_data_compensated[plot_range].real/np.mean(np.abs(complex_data_compensated)), '-x', label='Received Signal (Real Part)')
plt.title('Ideal vs. Received Signal (Time Domain Snapshot)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.savefig('./src/splited_data/ideal_vs_received_time_domain.png')


# ==============================================================================
#                      3. 反卷积估计脉冲响应
# ==============================================================================
print("\n步骤 3: 通过时域反卷积估计脉冲响应...")

# 参数设置
N_filter = 8 * sps
# 构建矩阵A
A_rows = len(complex_data_compensated) - N_filter + 1
A = np.zeros((A_rows, N_filter))

for n in range(A_rows):
    A[n, :] = ideal_signal[n:n + N_filter]

# 构建AA矩阵（实部和虚部堆叠）
AA = np.vstack([np.real(A), np.imag(A)],dtype=complex)

# 提取对应的实部和虚部数据
xxI = np.real(complex_data_compensated)[N_filter//2 : N_filter//2 + A_rows]
xxQ = np.imag(complex_data_compensated)[N_filter//2 : N_filter//2 + A_rows]

# 构建目标向量xxxx
xxxx = np.hstack([xxI, xxQ])

# 求解最小二乘问题
h = np.linalg.lstsq(AA, xxxx, rcond=None)[0]

# 翻转h向量
h_estimated_ls = np.flipud(h)

print(f"估计出长度为 {len(h_estimated_ls)} 的脉冲响应。")

def theoretical_rc(num_taps=64, alpha=0.3, sps=8):
    t = np.arange(-num_taps//2, num_taps//2) / sps
    h = np.sinc(t) * np.cos(np.pi * alpha * t) / (1 - (2 * alpha * t)**2)
    h[t == 0] = 1.0  # 处理t=0时的特殊情况
    h[np.abs(1 - (2 * alpha * t)**2) < 1e-6] = np.pi/4 * np.sinc(1/(2*alpha))  # 处理分母接近0的情况
    return h / np.max(np.abs(h))  # 归一化

if len(h_estimated_ls) > 0:
    print(f"成功估计出长度为 {len(h_estimated_ls)} 的脉冲响应。")

    # --- 【新增】步骤 4: 结果分析与可视化 ---
    print("\n步骤 4: 分析和可视化时域LS估计结果...")

    # 计算频率响应
    w, h_response_ls = scipy_signal.freqz(h_estimated_ls, worN=4096, fs=fs)

    # --- 绘图 ---
    plt.figure(figsize=(15, 10))

    # 1. 估计出的脉冲响应
    plt.subplot(2, 2, 1)
    plt.plot(h_estimated_ls)
    plt.title('Estimated Impulse Response (Time-Domain LS, Normalized)')
    plt.xlabel('Taps')
    plt.ylabel('Amplitude')
    plt.grid(True)

    num_taps = 64
    best_beta = 0.33
    T = sps / fs
    t = np.arange(len(h_estimated_ls)) - len(h_estimated_ls)//2
    t_seconds = t / fs
    
    # 生成理论响应
    h_theoretical = theoretical_rc(num_taps,best_beta,sps)
    # 归一化对比
    h_measured_norm = h_estimated_ls / np.max(np.abs(h_estimated_ls))
    h_theoretical_norm = h_theoretical / np.max(np.abs(h_theoretical))
    
    # 计算误差指标
    mse = np.mean((h_measured_norm - h_theoretical_norm)**2)
    correlation = np.corrcoef(h_measured_norm, h_theoretical_norm)[0,1]

    # 2. 估计出的频率响应幅度
    plt.subplot(2, 2, 2)
    plt.plot(h_theoretical_norm, '--',label=f'Generated RC (rolloff={best_beta})')
    plt.plot(h_measured_norm, label='Original Estimated Filter')
    plt.title('RC Filter Comparison')
    plt.xlabel('Taps')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    # 3. 验证：用估计的滤波器对理想信号进行卷积
    def find_optimal_delay(reference, measured, max_delay=10):
        """找到最优延迟量"""
        correlations = []
        for delay in range(-max_delay, max_delay+1):
            if delay >= 0:
                ref_shifted = reference[delay:]
                meas_trimmed = measured[:len(ref_shifted)]
            else:
                ref_shifted = reference[:delay] if delay != 0 else reference
                meas_trimmed = measured[-delay:len(ref_shifted)-delay] if delay != 0 else measured
                
            if len(ref_shifted) > 0 and len(ref_shifted) == len(meas_trimmed):
                correlation = np.corrcoef(ref_shifted, meas_trimmed)[0, 1]
                correlations.append((delay, correlation))
        
        # 找到相关性最大的延迟
        optimal_delay, max_corr = max(correlations, key=lambda x: abs(x[1]))
        return optimal_delay

    def apply_delay_signal(signal, delay):
        """应用延迟到信号"""
        if delay > 0:
            # 正延迟：在开头添加零
            return np.concatenate([np.zeros(delay, dtype=signal.dtype), signal[:-delay]])
        elif delay < 0:
            # 负延迟：去掉开头的样本
            return signal[-delay:]
        else:
            return signal

    # 使用更精确的延迟估计方法
    optimal_delay = find_optimal_delay(ideal_signal, complex_data_compensated)
    print(f"最优延迟量: {optimal_delay} 个样本")

    # 应用精确延迟
    ideal_signal_aligned = apply_delay_signal(ideal_signal, optimal_delay)

    # 然后进行卷积
    reconstructed_rx_signal = scipy_signal.convolve(ideal_signal_aligned, h_estimated_ls, mode='same')

    noise = reconstructed_rx_signal - complex_data_compensated

    snr = 10*math.log10(np.mean(np.abs(reconstructed_rx_signal)**2) / np.mean(np.abs(noise)**2))
    print(f'SNR:{snr}dB')

    plot_range = slice(0, 160)
    plt.subplot(2, 2, 3)
    plt.plot(complex_data_compensated[plot_range].real, label='Original Received Signal (Real)', linewidth=2)
    plt.plot(reconstructed_rx_signal[plot_range].real, '--', label='Final Reconstructed Signal (Real)', linewidth=1.5)
    plt.plot(noise[plot_range].real,label='Noise')
    plt.title('Validation: Final Reconstructed vs. Original (Amplitude Aligned)')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    # 4. 分析噪声的功率谱
    print("\n分析信道噪声的统计分布...")

    # 1. 分离I路和Q路噪声
    noise_I = noise.real
    noise_Q = noise.imag

    plt.subplot(2, 2, 4)
    num_bins = 30  # MATLAB代码中的30个bins
    alpha = 0.6    # 设置透明度，以便在重叠时能看到两个分布

    # --- 创建图形 ---

    # `density=True` 将直方图归一化为概率密度，便于比较和拟合
    # `alpha` 设置透明度
    plt.hist(noise_I, bins=num_bins, density=True, alpha=alpha, label='I路')
    plt.hist(noise_Q, bins=num_bins, density=True, alpha=alpha, label='Q路')

    # 【可选】拟合并绘制高斯分布曲线，以验证噪声是否为高斯分布
    # 计算I路和Q路噪声的均值和标准差
    mu_I, std_I = stats.norm.fit(noise_I)
    mu_Q, std_Q = stats.norm.fit(noise_Q)

    # 为拟合的高斯分布生成x轴点
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    
    # 计算高斯概率密度函数
    p_I = stats.norm.pdf(x, mu_I, std_I)
    p_Q = stats.norm.pdf(x, mu_Q, std_Q)

    # 绘制拟合曲线
    plt.plot(x, p_I, 'k-', linewidth=2, label=f'I路拟合 (μ={mu_I:.2f}, σ={std_I:.2f})')
    plt.plot(x, p_Q, 'k--', linewidth=2, label=f'Q路拟合 (μ={mu_Q:.2f}, σ={std_Q:.2f})')

    plt.title('信道噪声分布', fontsize=16)
    plt.xlabel('噪声幅度', fontsize=12)
    plt.ylabel('概率密度', fontsize=12) # 因为用了density=True，所以Y轴是概率密度
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.box(True) # 对应MATLAB的 box on

    plt.savefig('./src/splited_data/pulse_response_estimation_ls.png')
    # ==========================
    # 噪声分布与功率谱分析
    # ==========================
    print("\n进一步分析噪声的分布和功率谱...")

    plt.figure(figsize=(12, 5))

    # 1. 绘制噪声的幅度分布直方图
    plt.subplot(1,3, 1)
    num_bins = 30  # MATLAB代码中的30个bins
    alpha = 0.6    # 设置透明度，以便在重叠时能看到两个分布

    # --- 创建图形 ---

    # `density=True` 将直方图归一化为概率密度，便于比较和拟合
    # `alpha` 设置透明度
    plt.hist(noise_I, bins=num_bins, density=True, alpha=alpha, label='I路')
    plt.hist(noise_Q, bins=num_bins, density=True, alpha=alpha, label='Q路')

    # 【可选】拟合并绘制高斯分布曲线，以验证噪声是否为高斯分布
    # 计算I路和Q路噪声的均值和标准差
    mu_I, std_I = stats.norm.fit(noise_I)
    mu_Q, std_Q = stats.norm.fit(noise_Q)

    # 为拟合的高斯分布生成x轴点
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    
    # 计算高斯概率密度函数
    p_I = stats.norm.pdf(x, mu_I, std_I)
    p_Q = stats.norm.pdf(x, mu_Q, std_Q)

    # 绘制拟合曲线
    plt.plot(x, p_I, 'k-', linewidth=2, label=f'I路拟合 (μ={mu_I:.2f}, σ={std_I:.2f})')
    plt.plot(x, p_Q, 'k--', linewidth=2, label=f'Q路拟合 (μ={mu_Q:.2f}, σ={std_Q:.2f})')

    plt.title('信道噪声分布', fontsize=16)
    plt.xlabel('噪声幅度', fontsize=12)
    plt.ylabel('概率密度', fontsize=12) # 因为用了density=True，所以Y轴是概率密度
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.box(True) # 对应MATLAB的 box on

    # 2. 绘制噪声的功率谱
    plt.subplot(1, 3, 2)
    noise_psd_freq, noise_psd = scipy_signal.welch(noise, fs=fs, nperseg=4096)
    plt.semilogy(noise_psd_freq/1e3, noise_psd, label='Noise PSD')
    plt.title('噪声功率谱')
    plt.xlabel('频率 (kHz)')
    plt.ylabel('功率谱密度')
    plt.legend()
    plt.grid(True)

    # 3. 绘制原始信号的功率谱
    plt.subplot(1, 3, 3)
    signal_psd_freq, signal_psd = scipy_signal.welch(complex_data_compensated, fs=fs, nperseg=4096)
    plt.semilogy(signal_psd_freq/1e3, signal_psd, label='Signal PSD', color='g')
    plt.title('信号功率谱')
    plt.xlabel('频率 (kHz)')
    plt.ylabel('功率谱密度')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('./src/splited_data/noise_distribution_and_psd.png')
    plt.show()
else:
    print("脉冲响应估计失败。")