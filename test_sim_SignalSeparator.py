from model_complex import SignalSeparator
import torch
import time
import numpy as np
from torch.utils.data import DataLoader
from load_dataset import SignalDataset
import matplotlib.pyplot as plt
import pandas as pd
from compensation import costas_loop

plt.rcParams['font.size'] = 14

test_mode = 'qpsk_sim_data'
selected_cuda = 6
device = torch.device(f'cuda:{selected_cuda}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(int(selected_cuda))
model = SignalSeparator().to(device)

model.load_state_dict(torch.load('./src/check_points/signal_separator_'+test_mode+'_simr75_epoch74.pth',weights_only=True))  # 加载模型权重
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
loaded_data = torch.load('/nas/datasets/yixin/PCMA/sim_data/'+test_mode+'_test.pth')

dataset = SignalDataset(loaded_data)
batch_size = 64
test_loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

running_test_loss = 0.0
fs = 12e6
freq_offset = 50     # 频偏（Hz）
init_phase = -0.0027     # 初始相位（弧度）

columns = ['snr', 'freq_overlap_percentage', 'amplititude_ratio', 'oversampling_ratio','random_phase_diff','random_delay','modulation_type']
result_df = pd.DataFrame(columns=columns)

batch_idx = 0

def rrc_filter(beta, sps, num_taps):
    t = np.arange(-num_taps//2, num_taps//2) / sps
    with np.errstate(divide='ignore', invalid='ignore'):
        h = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t) ** 2)
        h[np.isnan(h)] = 1.0 - beta + (4 * beta / np.pi)
    h = h / np.sqrt(np.sum(h**2))  # 归一化
    return h

# 升余弦参数（与生成数据时一致）
beta = 0.33
sps = 8
num_taps = 64
rrc = rrc_filter(beta, sps, num_taps)

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

def extract_symbols(rx, sps):
    # 抽取符号点
    return rx[::sps]

idx = 0
true_bits1_all = []
true_bits2_all = []

pred_rfsignal1_all = []
pred_rfsignal2_all = []

rfsignal1_all = []
rfsignal2_all = []

with torch.no_grad():
    for batch in test_loader:
        batch_idx += 1
        start = time.perf_counter()
        mixsignal_real = batch['mixsignal_real'].to(device).unsqueeze(1)
        mixsignal_imag = batch['mixsignal_imag'].to(device).unsqueeze(1)
        rfsignal1_real = batch['rfsignal1_real'].to(device).unsqueeze(1)
        rfsignal1_imag = batch['rfsignal1_imag'].to(device).unsqueeze(1)
        rfsignal2_real = batch['rfsignal2_real'].to(device).unsqueeze(1)
        rfsignal2_imag = batch['rfsignal2_imag'].to(device).unsqueeze(1)
        bits1 = batch['bits1']
        bits2 = batch['bits2']
        (snr,freq_overlap_percetage,amplititude_ratio,oversampling_ratio,random_phase_diff,random_delay,modulation_type) = batch['params']
        origin_len = batch['origin_len']

        mixsignal = torch.cat([mixsignal_real,mixsignal_imag],dim=1)
        rfsignal1 = torch.cat([rfsignal1_real,rfsignal1_imag],dim=1)
        rfsignal2 = torch.cat([rfsignal2_real,rfsignal2_imag],dim=1)
        
        test_output = model(mixsignal)
        pred_rfsignal1 = torch.cat([test_output[0],test_output[1]],dim=1)
        pred_rfsignal2 = torch.cat([test_output[2],test_output[3]],dim=1)

        loss1 = torch.nn.functional.mse_loss(pred_rfsignal1, rfsignal1, reduction='none')
        loss1 = (loss1.mean(dim=[1,2]) / torch.norm(rfsignal1, dim=[1,2])).mean()
        loss2 = torch.nn.functional.mse_loss(pred_rfsignal2, rfsignal2, reduction='none')
        loss2 = (loss2.mean(dim=[1,2]) / torch.norm(rfsignal2, dim=[1,2])).mean()
        test_loss = loss1 + loss2
    
        running_test_loss += test_loss.item() 

        for i in range(rfsignal1_real.size()[0]):
            mixsignal_real_i = mixsignal_real[i].cpu().numpy()[0]
            mixsignal_imag_i = mixsignal_imag[i].cpu().numpy()[0]
            rfsignal1_real_i = rfsignal1_real[i].cpu().numpy()[0]
            rfsignal1_imag_i = rfsignal1_imag[i].cpu().numpy()[0]
            rfsignal1_i = rfsignal1_real_i + 1j * rfsignal1_imag_i
            rfsignal2_real_i = rfsignal2_real[i].cpu().numpy()[0]
            rfsignal2_imag_i = rfsignal2_imag[i].cpu().numpy()[0]
            rfsignal2_i = rfsignal2_real_i + 1j * rfsignal2_imag_i

            pred_rfsignal1_real_i = test_output[0][i].cpu().numpy()[0]
            pred_rfsignal1_imag_i = test_output[1][i].cpu().numpy()[0]
            pred_rfsignal1_i = pred_rfsignal1_real_i + 1j * pred_rfsignal1_imag_i
            pred_rfsignal2_real_i = test_output[2][i].cpu().numpy()[0]
            pred_rfsignal2_imag_i = test_output[3][i].cpu().numpy()[0]
            pred_rfsignal2_i = pred_rfsignal2_real_i + 1j * pred_rfsignal2_imag_i

            bits1_i = bits1[i].cpu().numpy()
            bits2_i = bits2[i].cpu().numpy()

            snr_i = snr[i].item()
            freq_overlap_percetage_i = freq_overlap_percetage[i].item()
            amplititude_ratio_i = float(str(amplititude_ratio[i].item())[:5])

            oversampling_ratio_i = oversampling_ratio[i].item()
            random_phase_diff_i = float(str(random_phase_diff[i].item())[:5])
            random_delay_i = float(str(random_delay[i].item())[:5])
            modulation_type_i = modulation_type[i]

            # 保存第一个batch和第十个batch信号1实部对比图
            if (batch_idx == 1 or batch_idx == 9) and i == 0:
                plt.figure(figsize=(10, 6))
                plt.subplot(2,1,1)
                plt.plot(rfsignal1_real_i[:16*20], label='true sig1 real')
                plt.plot(pred_rfsignal1_real_i[:16*20], label='pred sig1 real')
                plt.legend()
                plt.grid(True)
                plt.title(f'Batch {batch_idx} Signal 1 Real Part vs Predicted')
                
                plt.subplot(2,1,2)
                plt.plot(rfsignal2_real_i[:16*20], label='true sig2 real')
                plt.plot(pred_rfsignal2_real_i[:16*20], label='pred sig2 real')
                plt.legend()
                plt.grid(True)
                plt.title('True Signal 2 Real Part vs Predicted Signal 2 Real Part')
                plt.savefig(f'./src/pics/{test_mode}_real_batch{batch_idx}_sample{i}.png')
                plt.close()

            # 预测信号解调
            pred_rfsignal1_complex = pred_rfsignal1_real_i + 1j * pred_rfsignal1_imag_i
            pred_rfsignal2_complex = pred_rfsignal2_real_i + 1j * pred_rfsignal2_imag_i

            pred_rfsignal1_all.append(pred_rfsignal1_complex)
            pred_rfsignal2_all.append(pred_rfsignal2_complex)
            true_bits1_all.append(bits1_i)
            true_bits2_all.append(bits2_i)
            rfsignal1_all.append(rfsignal1_i)
            rfsignal2_all.append(rfsignal2_i)


            start_idx = len(pred_rfsignal1_complex) * idx
            end_idx = len(pred_rfsignal1_complex) * (idx + 1)

            t = np.arange(start_idx, end_idx) / fs  # 注意这里不是从0开始
            init_phase = -0.0027
            # 补偿频偏和相偏
            phase_comp1 = np.exp(-1j * (2 * np.pi * freq_offset * t))
            phase_comp2 = np.exp(-1j * (2 * np.pi * freq_offset * t + init_phase))
            pred_rfsignal1_i_comp = pred_rfsignal1_i * phase_comp1
            pred_rfsignal2_i_comp = pred_rfsignal2_i * phase_comp2
            rfsignal1_i_comp = rfsignal1_i * phase_comp1
            rfsignal2_i_comp = rfsignal2_i * phase_comp2

            rfsignal2_i_compensated, phase_history = costas_loop(rfsignal2_i, loop_bandwidth=0.001, sps=8)


            # 匹配滤波
            pred_rfsignal1_i_filt = np.convolve(pred_rfsignal1_i_comp, rrc, mode='same')
            pred_rfsignal2_i_filt = np.convolve(pred_rfsignal2_i_comp, rrc, mode='same')
            rfsignal1_i_filt = np.convolve(rfsignal1_i_comp, rrc, mode='same')
            rfsignal2_i_filt = np.convolve(rfsignal2_i_comp, rrc, mode='same')
            rfsignal1_i_filt2 = np.convolve(rfsignal2_i_compensated, rrc, mode='same')

            # 抽取符号点
            pred_symbols1 = extract_symbols(pred_rfsignal1_i_filt, sps=8)  # sps根据你的数据集设置
            pred_symbols2 = extract_symbols(pred_rfsignal2_i_filt, sps=8) * np.exp(-1j*random_phase_diff_i)
            symbols1 = extract_symbols(rfsignal1_i_filt, sps=8)  # sps根据你的数据集设置
            symbols2 = extract_symbols(rfsignal2_i_filt, sps=8) * np.exp(-1j*random_phase_diff_i)  # 考虑相位差

            # plt.figure(figsize=(10, 6))
            # plt.scatter(np.real(symbols2_2), np.imag(symbols2_2), color='green', s=10, label='Corrected Symbols 2')
            # plt.title(f'Symbol Constellation for Signal 2 after Costas Loop (Batch {batch_idx} Sample {i})')
            # plt.xlabel('In-Phase')
            # plt.ylabel('Quadrature')
            # plt.grid(True)
            # plt.savefig(f'./src/pics/{test_mode}_costas_loop_batch{batch_idx}_sample{i}.png')
            # plt.close()

            # QPSK解调
            pred_bits1 = qpsk_demod(pred_symbols1)
            pred_bits2 = qpsk_demod(pred_symbols2)
            demod_bits1 = qpsk_demod(symbols1)
            demod_bits2 = qpsk_demod(symbols2)


            # 与原始比特对比，计算BER
            true_bits1 = bits1[i].cpu().numpy()
            true_bits2 = bits2[i].cpu().numpy()
            ber1 = np.mean(pred_bits1 != demod_bits1[:len(pred_bits1)])
            ber2 = np.mean(pred_bits2 != demod_bits2[:len(pred_bits2)])
            ber1_ideal = np.mean(demod_bits1 != true_bits1[:len(demod_bits1)])
            ber2_ideal = np.mean(demod_bits2 != true_bits2[:len(demod_bits2)])
            # if ber1_ideal != 0 or ber2_ideal != 0:
            #     print("errer in ideal demodulation!")
            #     os._exit(0)
            # 可视化
            if idx == 0:
                plt.figure(figsize=(10, 6))
                plt.subplot(2,1,1)
                plt.scatter(np.real(pred_symbols1), np.imag(pred_symbols1), color='blue', s=10, label='Predicted Symbols 1')
                plt.scatter(np.real(symbols1), np.imag(symbols1), color='red', s=10, label='True Symbols 1')
                plt.title(f'Symbol Constellation for Signal 1 (Batch {batch_idx} Sample {i})')
                plt.xlabel('In-Phase')
                plt.ylabel('Quadrature')
                plt.grid(True)
                
                plt.subplot(2,1,2)
                plt.plot(pred_rfsignal1_real_i[:16*20], label='Predicted Signal 1 Real Part')
                plt.plot(rfsignal1_real_i[:16*20], label='True Signal 1 Real Part')
                plt.title(f'Signal 1 Real Part Comparison (Batch {batch_idx} Sample {i})')
                plt.xlabel('Sample Index')
                plt.ylabel('Amplitude')
                plt.grid(True)
                plt.legend()
                plt.savefig(f'./src/pics/{test_mode}_real1_comp_batch{batch_idx}_sample{i}.png')
                plt.close()
            
            # ------------------------------------------------------------------
            new_row = {
                'snr':snr_i,
                'freq_overlap_percentage':freq_overlap_percetage_i,
                'amplititude_ratio':amplititude_ratio_i,
                'oversampling_ratio':oversampling_ratio_i,
                'random_phase_diff':random_phase_diff_i,
                'random_delay':random_delay_i,
                'BER1':ber1,
                'BER2':ber2,
                'BER':(ber1+ber2)/2,
                'modulation_type':modulation_type_i
            }
            new_row_df = pd.DataFrame([new_row])
            idx += 1
            # 如果result_df是空的，需要先初始化
            if result_df.empty:
                result_df = new_row_df
            else:
                # 使用concat进行拼接
                result_df = pd.concat([result_df, new_row_df], ignore_index=True)
        end = time.perf_counter()
        print(f"Time taken for batch {i+1}: {end - start} seconds")
print(f"Test Loss: {running_test_loss / len(test_loader)}")

pred_rfsignal1_all = np.array(pred_rfsignal1_all).flatten()
pred_rfsignal2_all = np.array(pred_rfsignal2_all).flatten()
rfsignal1_all = np.array(rfsignal1_all).flatten()
rfsignal2_all = np.array(rfsignal2_all).flatten()
true_bits1_all = np.array(true_bits1_all).flatten()
true_bits2_all = np.array(true_bits2_all).flatten()

if test_mode == 'qpsk_sim_data':

    BER_mean = result_df['BER'].mean()
    print("BER均值为:", BER_mean)
    BER1_mean = result_df['BER1'].mean()
    print("BER1均值为:", BER1_mean)
    BER2_mean = result_df['BER2'].mean()
    print("BER2均值为:", BER2_mean)

    n = len(result_df['BER'])

    mid_point = n // 2
    first_half_mean = result_df['BER'].iloc[0:mid_point].mean()

    second_half_mean = result_df['BER'].iloc[mid_point:].mean()
    print(f"前半部分均值: {first_half_mean}")
    print(f"后半部分均值: {second_half_mean}")
