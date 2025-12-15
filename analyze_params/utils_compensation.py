import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


def evaluate_clustering_quality(symbols):
    """
    评估QPSK星座图的聚类质量。
    通过计算符号点到理想QPSK点的平均距离的倒数来评估。
    得分越高，聚类效果越好。

    Args:
        symbols (np.ndarray): 复数符号序列。

    Returns:
        float: 聚类质量得分。
    """
    if len(symbols) == 0:
        return -np.inf # 如果没有数据，返回负无穷

    # 1. 定义理想的QPSK星座点 (归一化到单位圆上)
    ideal_qpsk = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)

    # 2. 归一化输入符号的能量，使其平均功率为1
    # 这一步很重要，因为它消除了信号幅度对距离计算的影响
    power = np.mean(np.abs(symbols)**2)
    if power == 0:
        return -np.inf
    normalized_symbols = symbols / np.sqrt(power)

    # 3. 计算每个符号到所有理想点的距离
    # 使用广播机制高效计算
    distances = np.abs(normalized_symbols[:, np.newaxis] - ideal_qpsk)

    # 4. 找到每个符号最近的理想点及其距离
    min_distances = np.min(distances, axis=1)

    # 5. 计算平均最小距离
    mean_error_distance = np.mean(min_distances)

    # 6. 返回平均距离的倒数作为得分。距离越小，得分越高。
    # 加上一个很小的数防止除以零
    return 1.0 / (mean_error_distance + 1e-9)

def compare_psd(original, reconstructed, fs, title_prefix="linear_compensation"):
    """
    计算并对比两个信号的功率谱密度，返回定量指标。
    """
    nfft = 4096
    # 使用 Welch 方法计算PSD，能提供更平滑、更稳健的估计
    f_orig, Pxx_orig = signal.welch(original, fs=fs, nperseg=nfft, return_onesided=False)
    f_recon, Pxx_recon = signal.welch(reconstructed, fs=fs, nperseg=nfft, return_onesided=False)

    # 将频率轴从 [-fs/2, fs/2] 移动到 [0, fs] 方便观察
    f_orig = np.fft.fftshift(f_orig)
    Pxx_orig = np.fft.fftshift(Pxx_orig)
    f_recon = np.fft.fftshift(f_recon)
    Pxx_recon = np.fft.fftshift(Pxx_recon)

    # --- 定量指标计算 ---
    # 1. 找到主瓣峰值频率
    peak_idx_orig = np.argmax(Pxx_orig)
    peak_freq_orig = f_orig[peak_idx_orig]
    
    peak_idx_recon = np.argmax(Pxx_recon)
    peak_freq_recon = f_recon[peak_idx_recon]
    
    freq_offset_error = abs(peak_freq_recon - peak_freq_orig)
    
    # 2. 计算主瓣部分的NMSE (例如，取峰值周围的一小段区域)
    # 定义主瓣范围，例如峰值左右10kHz
    search_range = 10e3 
    mask = (f_orig > peak_freq_orig - search_range) & (f_orig < peak_freq_orig + search_range)
    
    Pxx_orig_mainlobe = Pxx_orig[mask]
    Pxx_recon_mainlobe = Pxx_recon[mask]
    
    # 归一化
    Pxx_orig_norm = Pxx_orig_mainlobe / np.max(Pxx_orig_mainlobe)
    Pxx_recon_norm = Pxx_recon_mainlobe / np.max(Pxx_recon_mainlobe)
    
    nmse = np.mean((Pxx_recon_norm - Pxx_orig_norm)**2)
    
    print(f"--- {title_prefix}PSD 对比结果 ---")
    print(f"原始信号峰值频率: {peak_freq_orig/1000:.2f} kHz")
    print(f"恢复信号峰值频率: {peak_freq_recon/1000:.2f} kHz")
    print(f"峰值频率误差: {freq_offset_error:.2f} Hz")
    print(f"主瓣归一化均方误差: {nmse:.6f}")

    # --- 绘图 ---
    plt.figure(figsize=(12, 6))
    plt.plot(f_orig/1000, 10*np.log10(Pxx_orig), label='Original Signal PSD')
    plt.plot(f_recon/1000, 10*np.log10(Pxx_recon), '--', label='Reconstructed Signal PSD', alpha=0.8)
    plt.title(f'{title_prefix} Power Spectral Density Comparison')
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./src/estimate_h/{title_prefix}_psd_comparison.png')
    
    return freq_offset_error, nmse

def calculate_evm(symbols, ideal_constellation):
    """
    计算符号序列相对于理想星座的EVM。
    """
    if len(symbols) == 0:
        return np.inf
        
    # 1. 归一化符号能量
    power = np.mean(np.abs(symbols)**2)
    if power == 0:
        return np.inf
    normalized_symbols = symbols / np.sqrt(power)
    
    # 2. 为每个符号找到最近的理想星座点
    distances = np.abs(normalized_symbols[:, np.newaxis] - ideal_constellation)
    closest_ideal_points = ideal_constellation[np.argmin(distances, axis=1)]
    
    # 3. 计算误差矢量
    error_vectors = normalized_symbols - closest_ideal_points
    
    # 4. 计算EVM (RMS)
    # EVM = sqrt( mean(|error_vector|^2) / mean(|ideal_symbol|^2) )
    # 因为理想星座点能量已归一化，mean(|ideal_symbol|^2) = 1
    evm_rms = np.sqrt(np.mean(np.abs(error_vectors)**2))
    
    # 转换为百分比
    return evm_rms * 100

def compare_evm(original_data, reconstructed_data, sps, best_offset,title_prefix="EVM_Comparison"):
    """
    对比原始信号和恢复信号的EVM。
    """
    # 理想QPSK星座点
    ideal_qpsk = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    
    # 抽取符号
    original_symbols = original_data[best_offset::sps]
    reconstructed_symbols = reconstructed_data[best_offset::sps]
    
    # 计算EVM
    evm_original = calculate_evm(original_symbols, ideal_qpsk)
    evm_reconstructed = calculate_evm(reconstructed_symbols, ideal_qpsk)
    
    print(f"\n--- EVM 对比结果 ---")
    print(f"原始信号 EVM (RMS): {evm_original:.4f}%")
    print(f"恢复信号 EVM (RMS): {evm_reconstructed:.4f}%")
    
    # 绘制星座图对比
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(original_symbols.real, original_symbols.imag, alpha=0.5, s=10)
    plt.scatter(ideal_qpsk.real, ideal_qpsk.imag, c='red', marker='x', s=100, label='Ideal Points')
    plt.title(f'Original Symbols (EVM: {evm_original:.2f}%)')
    plt.xlabel('In-phase')
    plt.ylabel('Quadrature')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(reconstructed_symbols.real, reconstructed_symbols.imag, alpha=0.5, s=10)
    plt.scatter(ideal_qpsk.real, ideal_qpsk.imag, c='red', marker='x', s=100, label='Ideal Points')
    plt.title(f'Reconstructed Symbols (EVM: {evm_reconstructed:.2f}%)')
    plt.xlabel('In-phase')
    plt.ylabel('Quadrature')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'./src/estimate_h/{title_prefix}_constellation_comparison.png')
    
    return evm_original, evm_reconstructed

def analyze_error_signal(original, reconstructed):
    """
    分析原始信号与恢复信号之间的误差。
    """
    error_signal = reconstructed - original
    
    # --- 定量指标 ---
    mean_error_real = np.mean(error_signal.real)
    mean_error_imag = np.mean(error_signal.imag)
    std_error_real = np.std(error_signal.real)
    std_error_imag = np.std(error_signal.imag)
    
    print(f"\n--- 误差信号分析 ---")
    print(f"误差实部均值: {mean_error_real:.6f}")
    print(f"误差虚部均值: {mean_error_imag:.6f}")
    print(f"误差实部标准差: {std_error_real:.6f}")
    print(f"误差虚部标准差: {std_error_imag:.6f}")
    
    # --- 绘图 ---
    plt.figure(figsize=(15, 10))
    
    # 1. 误差信号时域图
    plt.subplot(2, 2, 1)
    plt.plot(error_signal.real[:1000], label='Real Part')
    plt.title('Error Signal (First 1000 samples) Real Part')
    plt.xlabel('Sample Index')
    plt.ylabel('Error Amplitude')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(error_signal.imag[:1000], label='Imaginary Part', color='orange')
    plt.title('Error Signal (First 1000 samples) Imaginary Part')
    plt.xlabel('Sample Index')
    plt.ylabel('Error Amplitude')
    plt.legend()
    plt.grid(True)

    # 2. 误差信号实部直方图
    plt.subplot(2, 2, 3)
    plt.hist(error_signal.real, bins=100, density=True, alpha=0.7, label='Error Real Part')
    # 拟合一个高斯分布
    mu, sigma = mean_error_real, std_error_real
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2) ), 'r-', linewidth=2, label='Fitted Gaussian')
    plt.title('Histogram of Error Real Part')
    plt.xlabel('Error Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)

    # 3. 误差信号虚部直方图
    plt.subplot(2, 2, 4)
    plt.hist(error_signal.imag, bins=100, density=True, alpha=0.7, label='Error Imag Part')
    mu, sigma = mean_error_imag, std_error_imag
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2) ), 'r-', linewidth=2, label='Fitted Gaussian')
    plt.title('Histogram of Error Imaginary Part')
    plt.xlabel('Error Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./src/estimate_h/error_signal_analysis.png')

    snr = 10 * np.log10(np.mean(np.abs(original)**2) / np.mean(np.abs(error_signal)**2))
    print(f"信噪比 (SNR): {snr:.2f} dB")

    return snr



def estimate_pulse_response_ls(ideal_signal, received_signal, sps, filter_length_in_symbols=8):
    """
    使用时域最小二乘法估计成型滤波器的脉冲响应。

    Args:
        ideal_signal (np.ndarray): 理想的冲激串信号 (发送端)。
        received_signal (np.ndarray): 接收到的、经过滤波的信号。
        sps (int): 每符号采样点数。
        filter_length_in_symbols (int): 期望估计的滤波器长度（以符号为单位）。

    Returns:
        np.ndarray: 估计出的实系数脉冲响应。
    """
    N_filter = filter_length_in_symbols * sps
    L_x = len(ideal_signal)
    
    # 1. 构建实数卷积矩阵 A_real
    # 我们将问题建模为 real(A)*h = real(x) 和 imag(A)*h = imag(x)
    # 然后合并成 [real(A); imag(A)] * h = [real(x); imag(x)]
    # 由于h是实数，这样更稳健
    
    # 确定输出信号的长度，并考虑群延迟对齐
    # 群延迟约为 (N_filter - 1) / 2
    group_delay = (N_filter) // 2
    
    # 接收信号的有效长度，用于构建方程组
    # L_effective = L_x - N_filter + 1
    # 接收信号的切片需要与这个长度对齐
    start_idx_rx = group_delay
    end_idx_rx = start_idx_rx + (L_x - N_filter + 1)
    
    if end_idx_rx > len(received_signal):
        print("警告：信号长度不足以进行估计，请减小滤波器长度或增加信号长度。")
        return np.array([])

    # 构建卷积矩阵 A (实数部分)
    A = np.zeros((L_x - N_filter + 1, N_filter))
    for i in range(A.shape[0]):
        A[i, :] = ideal_signal[i : i + N_filter].real # 使用实部构建矩阵，因为h是实数

    # 准备扩展的矩阵 AA 和向量 xx
    AA = np.vstack([A, A]) # [real(A); imag(A)] 等效于 [A; A] 因为A是实的
    xx_real = received_signal[start_idx_rx:end_idx_rx].real
    xx_imag = received_signal[start_idx_rx:end_idx_rx].imag
    xx = np.concatenate([xx_real, xx_imag])

    # 2. 求解最小二乘问题 h = (AA^T * AA)^-1 * AA^T * xx
    # 使用 np.linalg.lstsq 进行稳定求解
    try:
        h, residuals, rank, s = np.linalg.lstsq(AA, xx, rcond=None)
    except np.linalg.LinAlgError:
        print("错误：最小二乘求解失败。矩阵可能是奇异的。")
        return np.array([])

    # 3. 翻转滤波器系数以修正卷积方向
    h_estimated = np.flip(h)

    return h_estimated