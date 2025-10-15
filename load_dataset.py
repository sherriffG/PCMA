import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import hdf5storage
import numpy as np

class mat_SignalDataset(Dataset):
    def __init__(self, mat_filepath, type=0):
        # 加载MAT文件
        if type == 0:
            data = loadmat(mat_filepath)
        elif type == 1:
            data = hdf5storage.loadmat(mat_filepath)
        else:
            raise ValueError("Invalid type.")
        # 提取结构体数组并处理维度
        struct_array = data['dataset'].squeeze()  # 形状变为 (N,)
        
        # 转换为可迭代的numpy记录数组
        self.samples = struct_array.view(np.recarray)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 处理复杂信号数据
        def process_signal(signal):
            # MATLAB single类型转numpy float32
            real_part = signal['real'].astype(np.float32)
            imag_part = signal['imag'].astype(np.float32)
            return torch.complex(
                torch.tensor(real_part), 
                torch.tensor(imag_part)
            )
        
        # 提取信号
        rfsignal_real = torch.tensor(sample['RFSignal'].real.astype(np.float32))
        rfsignal_imag = torch.tensor(sample['RFSignal'].imag.astype(np.float32))
        signal1_real = torch.tensor(sample['Signal1'].real.astype(np.float32))
        signal1_imag = torch.tensor(sample['Signal1'].imag.astype(np.float32))
        signal2_real = torch.tensor(sample['Signal2'].real.astype(np.float32))
        signal2_imag = torch.tensor(sample['Signal2'].imag.astype(np.float32))
        
        # 提取参数
        params = (
            sample['rate'].item(),
            sample['a_rate'].item(),
            sample['snr'].item(),
            sample['overlap_percentage'].item()
        )
        
        # 提取原始符号
        in1 = torch.tensor(sample['in1'], dtype=torch.int)
        in2 = torch.tensor(sample['in2'], dtype=torch.int)
        
        return (rfsignal_real,rfsignal_imag), (signal1_real,signal1_imag, signal2_real,signal2_imag), params, (in1, in2)
class SignalDataset_with_label(Dataset):
    def __init__(self,data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        sample = self.data[index]

        mixsignal_real = torch.from_numpy(sample['mixsignal'].real.copy()).float()
        mixsignal_imag = torch.from_numpy(sample['mixsignal'].imag.copy()).float()
        # mixsignal_clean_real = torch.from_numpy(sample['mixsignal_clean'].real.copy()).float()
        # mixsignal_clean_imag = torch.from_numpy(sample['mixsignal_clean'].imag.copy()).float()
        rfsignal1_real = torch.from_numpy(sample['rfsignal1'].real.copy()).float()
        rfsignal1_imag = torch.from_numpy(sample['rfsignal1'].imag.copy()).float()
        rfsignal2_real = torch.from_numpy(sample['rfsignal2'].real.copy()).float()
        rfsignal2_imag = torch.from_numpy(sample['rfsignal2'].imag.copy()).float()

        params = sample['params']

        bits1 = torch.tensor(sample['bits1'],dtype=torch.long)
        bits2 = torch.tensor(sample['bits2'],dtype=torch.long)

        origin_len = sample['origin_len']
        label1 = sample['label1']  # 获取标签
        centers1 = sample['centers1']  # 获取聚类中心
        label2 = sample['label2']  # 获取标签
        centers2 = sample['centers2']  # 获取聚类

        return{
            'mixsignal_real':mixsignal_real,
            'mixsignal_imag':mixsignal_imag,
            # 'mixsignal_clean_real':mixsignal_clean_real,
            # 'mixsignal_clean_imag':mixsignal_clean_imag,
            'rfsignal1_real':rfsignal1_real,
            'rfsignal1_imag':rfsignal1_imag,
            'rfsignal2_real':rfsignal2_real,
            'rfsignal2_imag':rfsignal2_imag,
            'params':params,
            'bits1':bits1,
            'bits2':bits2,
            'origin_len':origin_len,
            'label1': label1,
            'centers1': centers1,
            'label2': label2,
            'centers2': centers2
        }
class SignalDataset(Dataset):
    def __init__(self,data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        sample = self.data[index]

        mixsignal_real = torch.from_numpy(sample['mixsignal'].real.copy()).float()
        mixsignal_imag = torch.from_numpy(sample['mixsignal'].imag.copy()).float()
        # mixsignal_clean_real = torch.from_numpy(sample['mixsignal_clean'].real.copy()).float()
        # mixsignal_clean_imag = torch.from_numpy(sample['mixsignal_clean'].imag.copy()).float()
        rfsignal1_real = torch.from_numpy(sample['rfsignal1'].real.copy()).float()
        rfsignal1_imag = torch.from_numpy(sample['rfsignal1'].imag.copy()).float()
        rfsignal2_real = torch.from_numpy(sample['rfsignal2'].real.copy()).float()
        rfsignal2_imag = torch.from_numpy(sample['rfsignal2'].imag.copy()).float()

        params = sample['params']

        bits1 = torch.tensor(sample['bits1'],dtype=torch.long)
        bits2 = torch.tensor(sample['bits2'],dtype=torch.long)

        origin_len = sample['origin_len']

        return{
            'mixsignal_real':mixsignal_real,
            'mixsignal_imag':mixsignal_imag,
            # 'mixsignal_clean_real':mixsignal_clean_real,
            # 'mixsignal_clean_imag':mixsignal_clean_imag,
            'rfsignal1_real':rfsignal1_real,
            'rfsignal1_imag':rfsignal1_imag,
            'rfsignal2_real':rfsignal2_real,
            'rfsignal2_imag':rfsignal2_imag,
            'params':params,
            'bits1':bits1,
            'bits2':bits2,
            'origin_len':origin_len
        }