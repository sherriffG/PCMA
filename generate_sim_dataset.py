import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
import torch
import os
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

# ============= 参数设置 =============
mod_order = 4            # QPSK
beta = 0.33              # 滚降系数
sps = 8                  # 每符号采样数
fs = 12e6               # 采样频率
num_taps = 64            # 滤波器阶数
snr_db = 23.8            # 信噪比（dB）
freq_offset = 50     # 频偏（Hz）
init_phase = -0.0027     # 初始相位（弧度）
freq_overlap_percentage = 100  # 频率重叠百分比
amplititude_ratio = 0.5  # 信号2的幅度比例
random_phase_diff = 0  # 信号2的随机相位差
random_delay = 0 # 信号2的随机延时（符号数）
input_len = 3072 # 网络输入为3072个采样点

# 读取采集数据
file_path = '/nas/datasets/LYX/PCMA/QPSK_16/1050/1050000000_10000000__12000000_20250623170252_902610_0000.dat'
file_size = os.path.getsize(file_path)
print(f"File size: {file_size} bytes")
num_samples = file_size // (2 * 2)  # 每个样本由两个16位整数（I和Q）组成
print(f"Number of samples: {num_samples}")

data = np.memmap(file_path, dtype=np.int16, mode='r', shape=(num_samples,))
data = data.reshape(-1, 2)  # 将数据重塑为每行两个元素（I和Q）
# 转为复数
complex_data = data[:, 0].astype(np.float32) + 1j * data[:, 1].astype(np.float32)

# 切半
half = len(complex_data) // 2
signal1 = complex_data[:half]
signal2 = complex_data[half:half*2]
def split_blocks(signal, block_len):
    num_blocks = len(signal) // block_len
    return [signal[i*block_len:(i+1)*block_len] for i in range(num_blocks)]

signal1_blocks = split_blocks(signal1, input_len)
signal2_blocks = split_blocks(signal2, input_len)
def qpsk_mod(bits):
    # 每2个比特映射成一个QPSK符号
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
    return np.array(symbols) / np.sqrt(2)  # 归一化功率

# ============= 升余弦滤波器设计 =============
def rrc_filter(beta, sps, num_taps):
    t = np.arange(-num_taps//2, num_taps//2) / sps
    with np.errstate(divide='ignore', invalid='ignore'):
        h = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t) ** 2)
        h[np.isnan(h)] = 1.0 - beta + (4 * beta / np.pi)
    h = h / np.sqrt(np.sum(h**2))  # 归一化
    return h

rrc = rrc_filter(beta, sps, num_taps)

# ============= 添加 AWGN =============
def awgn(signal, snr_db):
    signal_power = np.mean(np.abs(signal)**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    return signal + noise

# ============= 数据集参数 =============

def classify_symbols(symbols):
    # 定义理想的QPSK星座点
    ideal_qpsk_constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j])
    # 对每个接收到的符号进行判决
    symbols_decided = np.zeros_like(symbols)
    for i, sym in enumerate(symbols):
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
    bits_decided = []
    for sym in symbols_decided:
        if sym == ideal_qpsk_constellation[0]:
            bits_decided.extend([0, 0])
        elif sym == ideal_qpsk_constellation[1]:
            bits_decided.extend([1, 0])
        elif sym == ideal_qpsk_constellation[2]:
            bits_decided.extend([0, 1])
        elif sym == ideal_qpsk_constellation[3]:
            bits_decided.extend([1, 1])
    return np.array(bits_decided)
def simple_normalize_dataset(dataset):
    """简单的两遍归一化"""
    # 收集所有信号数据
    all_signals = []
    for entry in dataset:
        all_signals.append(entry['mixsignal'])
        all_signals.append(entry['rfsignal1'])
        all_signals.append(entry['rfsignal2'])
    
    # 计算全局统计量
    all_i = np.concatenate([np.real(sig) for sig in all_signals])
    all_q = np.concatenate([np.imag(sig) for sig in all_signals])
    
    i_mean, i_std = np.mean(all_i), np.std(all_i)
    q_mean, q_std = np.mean(all_q), np.std(all_q)
    
    print(f"归一化参数 - I: mean={i_mean:.6f}, std={i_std:.6f}")
    print(f"归一化参数 - Q: mean={q_mean:.6f}, std={q_std:.6f}")
    
    # 应用归一化
    for entry in dataset:
        entry['mixsignal'] = (np.real(entry['mixsignal']) - i_mean) / i_std + 1j * (np.imag(entry['mixsignal']) - q_mean) / q_std
        entry['rfsignal1'] = (np.real(entry['rfsignal1']) - i_mean) / i_std + 1j * (np.imag(entry['rfsignal1']) - q_mean) / q_std
        entry['rfsignal2'] = (np.real(entry['rfsignal2']) - i_mean) / i_std + 1j * (np.imag(entry['rfsignal2']) - q_mean) / q_std
    
    return dataset
bit_len = int(3072 / 4)      # bit串长度
num_blocks_total = 50000      # 块数

dataset = []
entry_count = 0
idx = 0

sim_ratios = [0.25,0.5,0.75,1]
for sim_ratio in sim_ratios:
    num_blocks_sim = int(num_blocks_total * sim_ratio)
    num_blocks_real = num_blocks_total - num_blocks_sim
    dataset = []
    entry_count = 0
    idx = 0

    # 生成仿真数据
    for _ in range(num_blocks_sim):
        # 生成两组独立信号
        bits1 = np.random.randint(0, 2, bit_len)
        bits2 = np.random.randint(0, 2, bit_len)

        symbols1 = qpsk_mod(bits1)
        symbols2 = qpsk_mod(bits2) * np.exp(1j * random_phase_diff)

        symbols_up1 = np.zeros(len(symbols1) * sps, dtype=complex)
        symbols_up2 = np.zeros(len(symbols2) * sps, dtype=complex)
        symbols_up1[::sps] = symbols1
        symbols_up2[::sps] = symbols2 * amplititude_ratio

        tx1 = convolve(symbols_up1, rrc, mode='same')
        tx2 = convolve(symbols_up2, rrc, mode='same')


        start_idx = idx * len(tx1)
        end_idx = (idx+1) * len(tx1) 
        t = np.arange(start_idx,end_idx) / fs
        phase_offset1 = 2 * np.pi * freq_offset * t
        phase_offset2 = 2 * np.pi * freq_offset * t + init_phase

        tx1 = tx1 * np.exp(1j * phase_offset1)
        tx2 = tx2 * np.exp(1j * phase_offset2)

        rx1 = awgn(tx1, snr_db)
        rx2 = awgn(tx2, snr_db)
        rx = rx1 + rx2
        # 保存为2维实数（I/Q分量）
        rx_iq = np.stack([np.real(rx), np.imag(rx)], axis=1)

        # 训练不需要精确比特流 
        new_entry = {
            'mixsignal':rx,
            'rfsignal1':tx1,
            'rfsignal2':tx2,
            'params':(snr_db, freq_overlap_percentage, amplititude_ratio, sps, random_phase_diff, random_delay, 'QPSK'),
            'bits1':-1,
            'bits2':-1,
            'origin_len':-1
        }
        entry_count += 1
        if entry_count % 100 == 0:
            print(f"已生成 {entry_count} 块信号，当前sim_ratio={sim_ratio}")
        dataset.append(new_entry)
        idx += 1

    # 采集数据
    for index in range(num_blocks_real):
        
        
        mixsignal = signal1_blocks[index] + signal2_blocks[index]*amplititude_ratio

        # 训练不需要精确比特
        new_entry = {
            'mixsignal':mixsignal,
            'rfsignal1':signal1_blocks[index],
            'rfsignal2':signal2_blocks[index],
            'params':(snr_db, freq_overlap_percentage, amplititude_ratio, sps, random_phase_diff, random_delay, 'QPSK'),
            'bits1':-1,
            'bits2':-1,
            'origin_len':-1
        }
        dataset.append(new_entry)
        entry_count += 1
        if entry_count % 100 == 0:
            print(f"已生成 {entry_count} 块信号，当前sim_ratio={sim_ratio}")
    dataset_normed = simple_normalize_dataset(dataset)
    print('归一化完成，开始保存数据集...')
    torch.save(dataset_normed, f'/nas/datasets/yixin/PCMA/sim_data/qpsk_sim_data_simr{int(sim_ratio*100)}.pth')
    print(f"已生成并保存信号，每块长度 {bit_len*sps}，保存路径: /nas/datasets/yixin/PCMA/sim_data/qpsk_sim_data_simr{sim_ratio}.pth")