import numpy as np
from scipy import signal as sci_signal
import matplotlib.pyplot as plt

def costas_loop(signal_complex, loop_bandwidth, sps):
    """
    简单的Costas环实现用于QPSK
    """
    # 初始化
    phase_estimate = 0
    phase_history = []
    corrected_signal = []
    
    # 环路滤波器参数
    alpha = loop_bandwidth  # 比例项
    beta = alpha * alpha   # 积分项（经验值）
    
    integrator = 0
    
    for i in range(len(signal_complex)):
        # 相位旋转补偿
        current_sample = signal_complex[i] * np.exp(-1j * phase_estimate)
        corrected_signal.append(current_sample)
        
        # QPSK的相位误差检测
        # 使用决策导向的误差检测
        real_part = np.real(current_sample)
        imag_part = np.imag(current_sample)
        
        # 计算误差（简化版）
        error = np.sign(real_part) * imag_part - np.sign(imag_part) * real_part
        
        # 环路滤波
        integrator = integrator + beta * error
        phase_step = alpha * error + integrator
        
        # 更新相位估计
        phase_estimate += phase_step
        phase_history.append(phase_estimate)
    
    return np.array(corrected_signal), np.array(phase_history)
def constellation_analysis(signal, sps):
    """
    分析星座图以检测相位和幅度问题
    """
    # 在每个符号的中心采样
    center_offset = int(sps / 2)
    symbol_samples = signal[center_offset::sps]
    
    plt.figure(figsize=(10, 10))
    plt.scatter(symbol_samples.real, symbol_samples.imag, alpha=0.5)
    plt.title('Constellation Diagram')
    plt.xlabel('In-phase')
    plt.ylabel('Quadrature')
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('./src/splited_data/costas_loops_constellation_analysis.png')
    plt.show()
    
    # 计算幅度和相位的统计信息
    amplitudes = np.abs(symbol_samples)
    phases = np.angle(symbol_samples)
    
    print(f"幅度统计: 均值={np.mean(amplitudes):.4f}, 标准差={np.std(amplitudes):.4f}")
    print(f"相位统计: 均值={np.mean(phases):.4f}, 标准差={np.std(phases):.4f}")
    
    return amplitudes, phases
# 参数设置
fs = 12e6  # 采样频率
path = '/nas/datasets/LYX/PCMA/QPSK_16/1050/'
filename = 'splited_data.dat'
file = path + filename
sps = 8
# 读取数据
data = np.fromfile(file, dtype=np.int16)
complex_data = data[0::2].astype(np.float32) + 1j * data[1::2].astype(np.float32)

complex_data_compensated, phase_history = costas_loop(complex_data, loop_bandwidth=0.00001, sps=8)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
center_offset = int(sps / 2)
symbol_samples = complex_data_compensated[center_offset::sps]

plt.scatter(symbol_samples.real, symbol_samples.imag, alpha=0.5)
plt.title('Compensated Constellation Diagram')
plt.xlabel('In-phase')
plt.ylabel('Quadrature')
plt.grid(True)
plt.axis('equal')

plt.subplot(2, 1, 2)
center_offset = int(sps / 2)
symbol_samples = complex_data[center_offset::sps]
plt.scatter(symbol_samples.real, symbol_samples.imag, alpha=0.5)
plt.title('Original Constellation Diagram')
plt.xlabel('In-phase')
plt.ylabel('Quadrature')
plt.grid(True)
plt.axis('equal')

plt.tight_layout()
plt.savefig('./src/splited_data/costas_loop_phase_estimate.png')
