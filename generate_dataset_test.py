import numpy as np
import random
import torch
import os
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import send_email
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
random.seed(512)
from sklearn.manifold import TSNE

file_path = '/nas/datasets/LYX/PCMA/QPSK_16/1050/1050000000_10000000__12000000_20250623170252_902610_0000.dat'
file_size = os.path.getsize(file_path)
print(f"File size: {file_size} bytes")
num_samples = file_size // (2 * 2)
print(f"Number of samples: {num_samples}")

data = np.memmap(file_path, dtype=np.int16, mode='r', shape=(num_samples,))
data = data.reshape(-1, 2)
complex_data = data[:, 0].astype(np.float32) + 1j * data[:, 1].astype(np.float32)

half = len(complex_data) // 2
signal1 = complex_data[:half]
signal2 = complex_data[half:half*2]

symbol_len = 16
max_len = int(128 / 2 * 2 * 24*10)
offset = 3
n_cluster = 4

scalar1I = StandardScaler()
saclar1Q = StandardScaler()
signal1I = scalar1I.fit_transform(np.real(signal1).reshape(-1,1))
signal1Q = saclar1Q.fit_transform(np.imag(signal1).reshape(-1,1))
signal1_rms = (signal1I + 1j*signal1Q).flatten()

signal2I = StandardScaler()
saclar2Q = StandardScaler()
signal2I = signal2I.fit_transform(np.real(signal2).reshape(-1,1))
signal2Q = saclar2Q.fit_transform(np.imag(signal2).reshape(-1,1)) 
signal2_rms = (signal2I + 1j*signal2Q).flatten()

def split_blocks(signal, block_len):
    num_blocks = len(signal) // block_len
    return [signal[i*block_len:(i+1)*block_len] for i in range(num_blocks)]

signal1_blocks = split_blocks(signal1_rms, max_len)
signal2_blocks = split_blocks(signal2_rms, max_len)
num_block = min(len(signal1_blocks), len(signal2_blocks))
num_block = 100
snr_range = list(range(0,11,1))

def get_centers_from_labels(X, labels):
    """根据DBSCAN聚类结果计算每个类的中心"""
    centers = []
    label_ids = []
    for label in np.unique(labels):
        if label == -1:
            continue  # 跳过噪声点
        mask = (labels == label)
        centers.append(X[mask].mean(axis=0))
        label_ids.append(label)
    return np.array(centers), np.array(label_ids)

def get_centers_from_gmm(gmm):
    """GMM聚类中心"""
    return gmm.means_

def match_centers(prev_centers, curr_centers):
    dist = np.linalg.norm(prev_centers[:, None, :] - curr_centers[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(dist)
    return col_ind

def plot_eye_diagram(signal, samples_per_symbol, offset=0, num_eyes=10):
    """
    绘制信号的眼图
    
    参数:
    signal: 输入信号数组
    samples_per_symbol: 每个符号的采样点数
    offset: 起始偏移量
    num_eyes: 显示的眼图数量
    """
    # 计算需要提取的样本数
    span = num_eyes * samples_per_symbol
    signal = signal[offset:offset + span]
    
    # 创建眼图网格
    eye = np.reshape(signal, (num_eyes, samples_per_symbol))
    
    # 创建时间轴
    t = np.arange(0, samples_per_symbol) / samples_per_symbol
    
    # 绘制眼图
    plt.figure(figsize=(10, 6))
    for i in range(num_eyes):
        plt.plot(t, eye[i, :], 'b-', alpha=0.5, linewidth=0.5)
    
    plt.title('Eye Diagram')
    plt.xlabel('Time (Symbol Periods)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig('eye_diagram_mid.png')
from matplotlib.colors import LinearSegmentedColormap
def plot_eye_diagram_red_gradient(signal, samples_per_symbol, offset=0, num_eyes=10):
    """
    绘制信号的眼图，使用浅红到深红的颜色渐变
    
    参数:
    signal: 输入信号数组
    samples_per_symbol: 每个符号的采样点数
    offset: 起始偏移量
    num_eyes: 显示的眼图数量
    """
    # 确保有足够的数据
    if len(signal) < offset + num_eyes * samples_per_symbol:
        raise ValueError("信号长度不足以绘制指定数量的眼图")
    
    # 提取需要的信号段
    signal_segment = signal[offset:offset + num_eyes * samples_per_symbol]
    
    # 重塑为眼图矩阵 (num_eyes, samples_per_symbol)
    eye_matrix = signal_segment.reshape((num_eyes, samples_per_symbol))
    
    # 创建时间轴 (归一化到符号周期)
    t = np.linspace(0, 1, samples_per_symbol)
    
    # 创建红色渐变颜色映射 - 从浅红到深红
    red_colors = [(1, 0.8, 0.8), (0.8, 0, 0)]  # 从浅红到深红
    red_cmap = LinearSegmentedColormap.from_list("red_gradient", red_colors, N=num_eyes)
    colors = red_cmap(np.linspace(0, 1, num_eyes))
    
    # 绘制眼图
    plt.figure(figsize=(10, 6))
    for i in range(num_eyes):
        plt.plot(t, eye_matrix[i, :], color=colors[i], alpha=0.8, linewidth=1.0)
    
    plt.title('Eye Diagram with Red Gradient')
    plt.xlabel('Time (Symbol Periods)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=red_cmap, norm=plt.Normalize(0, num_eyes))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Eye Pattern Index')
    plt.savefig('eye_diagram_red_gradient_mid.png')
def enhanced_eye_diagram_with_color(signal, samples_per_symbol, offset=0, num_eyes=20, 
                                   cmap_name='viridis', alpha=0.7):
    """
    增强版眼图绘制，具有颜色渐变效果
    
    参数:
    signal: 输入信号
    samples_per_symbol: 每个符号的采样点数
    offset: 起始偏移量
    num_eyes: 显示的眼图数量
    cmap_name: 颜色映射名称
    alpha: 线条透明度
    """
    # 确保有足够的数据
    required_length = offset + num_eyes * samples_per_symbol
    if len(signal) < required_length:
        raise ValueError(f"信号长度不足。需要至少{required_length}个样本，但只有{len(signal)}个")
    
    # 提取信号段
    signal_segment = signal[offset:offset + num_eyes * samples_per_symbol]
    
    # 重塑为眼图矩阵
    eye_matrix = signal_segment.reshape((num_eyes, samples_per_symbol))
    
    # 创建时间轴 (归一化到两个符号周期)
    t = np.linspace(0, 2, samples_per_symbol)
    
    # 获取颜色映射
    cmap = plt.get_cmap(cmap_name)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 绘制眼图轨迹，每条线使用不同的颜色
    for i in range(num_eyes):
        color = cmap(i / num_eyes)  # 根据索引计算颜色
        ax.plot(t, eye_matrix[i, :], color=color, alpha=alpha, linewidth=1.0)
    
    # 计算并绘制平均眼图
    mean_eye = np.mean(eye_matrix, axis=0)
    ax.plot(t, mean_eye, 'k-', linewidth=2.5, label='Mean')
    
    ax.set_title('Eye Diagram with Color Gradient')
    ax.set_xlabel('Time (Symbol Periods)')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, num_eyes))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Eye Pattern Index')
    
    plt.tight_layout()
    plt.savefig('enhanced_eye_diagram_mid.png')

data_list = []
prev_centers1 = None
prev_labels1 = None
prev_centers2 = None
prev_labels2 = None
entry_count = 0
idx = 0
snr_range = [1e5]
for i in range(num_block):
    for snr in snr_range:
        oversampling_ratio = 8
        freq_overlap_percentage = 100
        amplititude_ratio = 0.8
        random_phase_diff = 0
        random_delay = 0
        modulation_type = 'QPSK'
        signal1_i = signal1_blocks[idx]
        signal2_i = signal2_blocks[len(signal2_blocks) - 1 - idx]
        idx += 1

        mixsignal = signal1_i + signal2_i*amplititude_ratio
        mixsignal_noisy = 0
        if snr >= 1e5:
            mixsignal_noisy = mixsignal
        else:
            signal_power = np.mean(np.abs(mixsignal)**2)
            snr_linear = 10**(snr/10.0)
            noise_power = signal_power / snr_linear
            noise_stddev = np.sqrt(noise_power)
            noise = (noise_stddev/np.sqrt(2)) * (np.random.randn(*mixsignal.shape)+ 1j*np.random.randn(*mixsignal.shape))
            mixsignal_noisy = mixsignal + noise
        params = (snr, freq_overlap_percentage, amplititude_ratio, oversampling_ratio, random_phase_diff, random_delay, modulation_type)

        # ----------- 对信号1聚类并对齐 -----------
        I1 = np.real(signal1_i).astype(np.float32)
        Q1 = np.imag(signal1_i).astype(np.float32)
        num_symbols1 = len(I1) // symbol_len
        I1 = I1[:num_symbols1 * symbol_len]
        Q1 = Q1[:num_symbols1 * symbol_len]
        # 新聚类方式：每16个点作为一个符号，I和Q拼成32维向量
        I1_reshape = I1.reshape(num_symbols1, symbol_len)[:,0:10]
        Q1_reshape = Q1.reshape(num_symbols1, symbol_len)[:,0:10]
        IQ1_symbols = np.concatenate([I1_reshape, Q1_reshape], axis=1)  

        # tsne1 = TSNE(n_components=2)
        # IQ1_2d = tsne1.fit_transform(IQ1_symbols)

        # kmeas1 = KMeans(n_clusters=n_cluster, random_state=0).fit(IQ1_2d)
        # labels1 = kmeas1.labels_
        # centers1 = kmeas1.cluster_centers_

        # plt.scatter(IQ1_2d[:,0], IQ1_2d[:,1], c=labels1, cmap='tab10', alpha=0.6)
        # plt.scatter(centers1[:,0], centers1[:,1], c='red', marker='x', s=100)
        # plt.title('KMeans Clustering Result Example')
        # plt.xlabel('tSNE1')
        # plt.ylabel('tSNE2')
        # plt.savefig('kmeans_clustering_example_mid.png')

        # if entry_count == 0:
        #     aligned_label1 = labels1
        #     aligned_centers1 = centers1
        # else:
        #     perm1 = match_centers(prev_centers1, centers1)
        #     label_map1 = {old: new for new, old in enumerate(perm1)}
        #     aligned_label1 = np.array([label_map1[l] for l in labels1])
        #     aligned_centers1 = centers1[perm1]

        # ----------- 对信号2聚类并对齐 -----------
        I2 = np.real(signal2_i).astype(np.float32)
        Q2 = np.imag(signal2_i).astype(np.float32)
        num_symbols2 = len(I2) // symbol_len
        I2 = I2[:num_symbols2 * symbol_len]
        Q2 = Q2[:num_symbols2 * symbol_len]
        # 新聚类方式：每16个点作为一个符号，I和Q拼成32维向量
        I2_reshape = I2.reshape(num_symbols2, symbol_len)[:,6:16]
        Q2_reshape = Q2.reshape(num_symbols2, symbol_len)[:,6:16]
        IQ2_symbols = np.concatenate([I2_reshape, Q2_reshape], axis=1)  # (num_symbols2, 32)
        # tsne2 = TSNE(n_components=2)
        # IQ2_2d = tsne2.fit_transform(IQ2_symbols)
        # kmeans2 = KMeans(n_clusters=n_cluster, random_state=0).fit(IQ2_2d)
        # labels2 = kmeans2.labels_
        # centers2 = kmeans2.cluster_centers_

        # gmm2 = GaussianMixture(n_components=n_cluster, random_state=0).fit(IQ2_symbols)
        # label2 = gmm2.predict(IQ2_symbols)
        # centers2 = get_centers_from_gmm(gmm2)

        # if entry_count == 0:
        #     aligned_label2 = labels2
        #     aligned_centers2 = centers2
        # else:
        #     perm2 = match_centers(prev_centers2, centers2)
        #     label_map2 = {old: new for new, old in enumerate(perm2)}
        #     aligned_label2 = np.array([label_map2[l] for l in labels2])
        #     aligned_centers2 = centers2[perm2]

        # ----------- 画眼图 -----------
        if entry_count == 0:
            plot_eye_diagram(signal1_i, symbol_len, offset=offset, num_eyes=100)
        # -------------------------------------

        origin_len = -1
        bits1 = -1
        bits2 = -1
        new_entry = {
            'mixsignal': mixsignal_noisy,
            'rfsignal1': signal1_i,
            'rfsignal2': signal2_i,
            'params': params,
            'bits1': bits1,
            'bits2': bits2,
            'origin_len': origin_len,
            # 'label1': aligned_label1,
            # 'centers1': aligned_centers1,
            # 'label2': aligned_label2,
            # 'centers2': aligned_centers2,
            'IQ1_symbols': IQ1_symbols,
        }
        data_list.append(new_entry)
        entry_count += 1

        # prev_centers1 = aligned_centers1
        # prev_labels1 = aligned_label1
        # prev_centers2 = aligned_centers2
        # prev_labels2 = aligned_label2

        if entry_count % 100 == 0:
            print(f"已生成 {entry_count} 个 entry")

# torch.save(data_list, '/nas/datasets/yixin/PCMA/NEWFILES/sim_data_mode_QPSK_16_1050_test_with_label_mid.pth')
torch.save(data_list,'/nas/datasets/yixin/PCMA/NEWFILES/sim_data_mode_QPSK_16_1050_test_with_label_mid_temp.pth')
send_email.send_email()  # 发送邮件通知