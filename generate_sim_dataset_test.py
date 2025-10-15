import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from compensation import costas_loop

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
best_offset = 3

file_path = '/nas/datasets/yixin/PCMA/sim_data/splited_data.pth'
file_size = os.path.getsize(file_path)
print(f"File size: {file_size} bytes")
num_samples = file_size // (2 * 2)  # 每个样本由两个16位整数（I和Q）组成
print(f"Number of samples: {num_samples}")

data = np.memmap(file_path, dtype=np.int16, mode='r', shape=(num_samples,))
data = data.reshape(-1, 2)  # 将数据重塑为每行两个元素（I和Q）
# 转为复数
complex_data = data[:, 0].astype(np.float32) + 1j * data[:, 1].astype(np.float32)


complex_data_compensated, phase_history = costas_loop(complex_data,loop_bandwidth=0.00001, sps=8)
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

# 切半
half = len(complex_data) // 2
signal1 = complex_data[:half]
signal2 = complex_data[half:half*2]
symbols1 = symbols_decided[:len(symbols_decided)//2]
symbols2 = symbols_decided[len(symbols_decided)//2:len(symbols_decided)]

def split_blocks(signal, block_len):
    num_blocks = len(signal) // block_len
    return [signal[i*block_len:(i+1)*block_len] for i in range(num_blocks)]

signal1_blocks = split_blocks(signal1, input_len)
signal2_blocks = split_blocks(signal2, input_len)
symbols1_blocks = split_blocks(symbols1, input_len//sps)
symbols2_blocks = split_blocks(symbols2, input_len//sps)

def energy_normalize_dataset(dataset):
    """分别对仿真信号和采集信号分组归一化，使能量一致"""
    # 统计仿真信号和采集信号的能量
    sim_energies = []
    real_energies = []
    for entry in dataset:
        if entry['origin_len'] == 1:  # 仿真信号
            sim_energies.append(np.mean(np.abs(entry['mixsignal'])**2))
        elif entry['origin_len'] == -1:  # 采集信号
            real_energies.append(np.mean(np.abs(entry['mixsignal'])**2))
    # 计算均值能量
    sim_energy_mean = np.mean(sim_energies) if sim_energies else 1.0
    real_energy_mean = np.mean(real_energies) if real_energies else 1.0

    print(f"仿真信号平均能量: {sim_energy_mean:.6f}")
    print(f"采集信号平均能量: {real_energy_mean:.6f}")

    # 分别归一化
    for entry in dataset:
        if entry['origin_len'] == 1:  # 仿真信号
            scale = np.sqrt(sim_energy_mean)
        elif entry['origin_len'] == -1:  # 采集信号
            scale = np.sqrt(real_energy_mean)
        else:
            scale = 1.0
        entry['mixsignal'] = entry['mixsignal'] / scale
        entry['rfsignal1'] = entry['rfsignal1'] / scale
        entry['rfsignal2'] = entry['rfsignal2'] / scale
    return dataset
def qpsk_demod(symbols):
    # QPSK判决，返回比特流
    bits = []
    symbols = symbols * np.sqrt(2)  # 还原归一化
    for s in symbols:
        if s.real >= 0 and s.imag >= 0:
            b1, b2 = 0, 0
        elif s.real < 0 and s.imag >= 0:
            b1, b2 = 0, 1
        elif s.real >= 0 and s.imag < 0:
            b1, b2 = 1, 0
        else:
            b1, b2 = 1, 1
        bits.extend([b1, b2])
    return np.array(bits)
dataset = []
entry_count = 0
num_blocks_real = min(len(signal1_blocks), len(signal2_blocks))
print('可用块数:', num_blocks_real)

for index in range(num_blocks_real):
    
    
    mixsignal = signal1_blocks[index] + signal2_blocks[index]*amplititude_ratio
    bits1 = qpsk_demod(symbols1_blocks[index])
    bits2 = qpsk_demod(symbols2_blocks[index])

    new_entry = {
        'mixsignal':mixsignal,
        'rfsignal1':signal1_blocks[index],
        'rfsignal2':signal2_blocks[index],
        'params':(snr_db, freq_overlap_percentage, amplititude_ratio, sps, random_phase_diff, random_delay, 'QPSK'),
        'bits1':bits1,
        'bits2':bits2,
        'origin_len':len(symbols1_blocks[index])
    }
    dataset.append(new_entry)
    entry_count += 1
    if entry_count % 100 == 0:
        print(f"已生成 {entry_count} 块信号")
dataset_normed = energy_normalize_dataset(dataset)
print('归一化完成，开始保存数据集...')
torch.save(dataset_normed, f'/nas/datasets/yixin/PCMA/sim_data/qpsk_sim_data_test.pth')
print(f"已生成并保存信号，每块长度 {input_len}，保存路径: /nas/datasets/yixin/PCMA/sim_data/qpsk_sim_data_test.pth")