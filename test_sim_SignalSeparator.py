# test_sim_SignalSeparator.py
# -*- coding: utf-8 -*-
import os, re, math, argparse, numpy as np, pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from scipy.signal import convolve

from compensation import costas_loop  # 目前没用到，但保留
from model_complex import SignalSeparator

torch.backends.cudnn.benchmark = True

# ==================== 分布式工具 ====================
def dist_is_initialized():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if dist_is_initialized() else 0

def get_world_size():
    return dist.get_world_size() if dist_is_initialized() else 1

def setup_ddp(backend="nccl"):
    if dist_is_initialized():
        return
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend=backend, init_method="env://")

def cleanup_ddp():
    if dist_is_initialized():
        dist.barrier()
        dist.destroy_process_group()

# ==================== DSP 基本件 ====================
beta, sps, num_taps = 0.33, 8, 64
fs = 12e6

def rc_filter(beta, sps, num_taps):
    t = np.arange(-num_taps//2, num_taps//2) / sps
    with np.errstate(divide='ignore', invalid='ignore'):
        h = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t) ** 2)
        h[np.isnan(h)] = 1.0 - beta + (4 * beta / np.pi)
    h = h / np.sqrt(np.sum(h**2))
    return h

rc = rc_filter(beta, sps, num_taps)

# ==================== 调制相关：bit/符号数 ====================
BITS_PER_SYMBOL = {
    "QPSK": 2,
    "8PSK": 3,
    "16QAM": 4,
}

# ==================== 解调函数 ====================
def qpsk_demod(symbols: np.ndarray) -> np.ndarray:
    """
    与 qpsk_mod 对应：
      00 -> (+,+)
      01 -> (-,+)
      10 -> (+,-)
      11 -> (-,-)
    """
    bits = []
    sym = symbols * np.sqrt(2)
    for s in sym:
        if s.real >= 0 and s.imag >= 0: b1, b2 = 0, 0
        elif s.real < 0 and s.imag >= 0: b1, b2 = 0, 1
        elif s.real >= 0 and s.imag < 0: b1, b2 = 1, 0
        else: b1, b2 = 1, 1
        bits.extend([b1, b2])
    return np.array(bits, dtype=np.int8)

def psk8_demod(symbols: np.ndarray) -> np.ndarray:
    """
    对应 8PSK：
      bits -> k = b0*4 + b1*2 + b2 (见你之前的调制实现)
      s = exp(j * 2πk/8)
    这里反过来：从角度恢复 k，再还原 bits (b0,b1,b2)。
    """
    angles = np.angle(symbols)
    angles = np.mod(angles, 2*np.pi)
    step = 2*np.pi / 8.0
    k = np.round(angles / step).astype(int) % 8  # 0..7
    bits = []
    for val in k:
        b0 = (val >> 2) & 1  # 权重 4
        b1 = (val >> 1) & 1  # 权重 2
        b2 = val & 1         # 权重 1
        bits.extend([b0, b1, b2])
    return np.array(bits, dtype=np.int8)

def qam16_demod(symbols: np.ndarray) -> np.ndarray:
    """
    对应 16QAM：
      I,Q ∈ {-3,-1,1,3}/sqrt(10)，Gray 编码：
        -3 -> 00, -1 -> 01, 1 -> 11, 3 -> 10
    这里按最近邻找 I/Q 所在的 level，再映射回 bits。
    """
    levels = np.array([-3., -1., 1., 3.]) / np.sqrt(10.0)
    bits = []
    for s in symbols:
        I = s.real
        Q = s.imag
        idx_I = np.argmin((I - levels)**2)
        idx_Q = np.argmin((Q - levels)**2)
        level_I = levels[idx_I]
        level_Q = levels[idx_Q]

        # level -> Gray bits
        if level_I < (-2/np.sqrt(10)):   bi0, bi1 = 0, 0   # -3
        elif level_I < 0:                bi0, bi1 = 0, 1   # -1
        elif level_I > (2/np.sqrt(10)):  bi0, bi1 = 1, 0   # 3
        else:                            bi0, bi1 = 1, 1   # 1

        if level_Q < (-2/np.sqrt(10)):   bq0, bq1 = 0, 0
        elif level_Q < 0:                bq0, bq1 = 0, 1
        elif level_Q > (2/np.sqrt(10)):  bq0, bq1 = 1, 0
        else:                            bq0, bq1 = 1, 1

        bits.extend([bi0, bi1, bq0, bq1])
    return np.array(bits, dtype=np.int8)

def demod_by_mod(symbols: np.ndarray, modulation: str) -> np.ndarray:
    modulation = (modulation or "QPSK").upper()
    if modulation == "QPSK":
        return qpsk_demod(symbols)
    elif modulation == "8PSK":
        return psk8_demod(symbols)
    elif modulation == "16QAM":
        return qam16_demod(symbols)
    else:
        # 默认当 QPSK 处理，避免崩溃
        return qpsk_demod(symbols)


# ==================== 调制函数（用于 SER 计算）====================
def qpsk_mod(bits: np.ndarray) -> np.ndarray:
    """QPSK Gray 映射，与原实现兼容。"""
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


def psk8_mod(bits: np.ndarray) -> np.ndarray:
    """
    8PSK 映射：
    - 每 3bit -> 一个符号，采用自然编码：k = b2*4 + b1*2 + b0
    - 符号 = exp(j*(2πk/8))，整体为 8 点均匀分布在单位圆上。
    """
    assert len(bits) % 3 == 0
    bits = bits.reshape(-1, 3)
    idx = bits[:, 0] * 4 + bits[:, 1] * 2 + bits[:, 2]
    phase = 2 * np.pi * idx / 8.0
    symbols = np.exp(1j * phase)
    return symbols.astype(complex)


def qam16_mod(bits: np.ndarray) -> np.ndarray:
    """
    16QAM 映射：
    - 每 4bit -> 2bit(I) + 2bit(Q)
    - 采用 Gray 编码 + 方阵 [-3,-1,1,3]，然后整体归一化到平均能量=1。
    """
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


def modulate(bits: np.ndarray, modulation: str) -> np.ndarray:
    """统一入口：根据 modulation 调用对应的调制函数。"""
    modulation = modulation.upper()
    if modulation == "QPSK":
        return qpsk_mod(bits)
    elif modulation == "8PSK":
        return psk8_mod(bits)
    elif modulation == "16QAM":
        return qam16_mod(bits)
    else:
        return qpsk_mod(bits)

# ==================== 其它工具 ====================
def align_phase(ref, est):
    c = np.mean(ref * np.conj(est) + 1e-12)
    a = np.angle(c)
    return est * np.exp(-1j * a)

def wrap_2pi(x): return np.mod(x, 2*np.pi)

def evm_rms(ref_syms, est_syms):
    num = np.mean(np.abs(est_syms - ref_syms)**2)
    den = np.mean(np.abs(ref_syms)**2) + 1e-12
    return float(np.sqrt(num / den))

def find_best_offset(y_mf, sps):
    best_off = 0
    best_eng = -1.0
    for off in range(sps):
        sym = y_mf[off::sps]
        eng = np.mean(np.abs(sym)**2)
        if eng > best_eng:
            best_eng = eng
            best_off = off
    return best_off

def mf_and_sample(wave, sps, rc, num_taps, guard_sym=None):
    if guard_sym is None:
        guard_sym = num_taps // sps  # 64/8=8 符号

    if wave is None or len(wave) == 0:
        return np.zeros(0, dtype=np.complex64)

    y_mf = convolve(wave, rc, mode='same')
    off = find_best_offset(y_mf, sps)
    syms = y_mf[off::sps]
    if len(syms) <= 2 * guard_sym:
        return np.zeros(0, dtype=np.complex64)
    syms = syms[guard_sym:-guard_sym]

    m = np.mean(np.abs(syms))
    if m > 0:
        syms = syms / m
    return syms.astype(np.complex64)

def slice_bits_to_match_syms(bits_full: np.ndarray, n_syms_used: int, bits_per_sym: int):
    """
    通用版：支持 QPSK/8PSK/16QAM
    """
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

# ==================== 解析 params ====================
def parse_params_all(pstr: str):
    """
    通用解析：兼容旧的 QPSK test_all 和新的 test_snr_amp/test_cfo_phase/test_delay
    """
    if not isinstance(pstr, str) or not pstr:
        return {}
    parts = [x.strip() for x in pstr.split(',')]
    floats = []
    for x in parts:
        try: floats.append(float(x))
        except: pass
    snr = floats[0] if len(floats) >= 1 else None
    amp = floats[1] if len(floats) >= 2 else None

    m_f1   = re.search(r"f_off1=([\-0-9.]+)\s*Hz", pstr)
    m_f2   = re.search(r"f_off2=([\-0-9.]+)\s*Hz", pstr)
    m_p1   = re.search(r"phi1=([\-0-9.]+)\s*rad", pstr)
    m_p2   = re.search(r"phi2=([\-0-9.]+)\s*rad", pstr)
    m_rep  = re.search(r"rep=([0-9]+)", pstr)
    m_d1   = re.search(r"delay1_samp=([\-0-9]+)", pstr)
    m_d2   = re.search(r"delay2_samp=([\-0-9]+)", pstr)
    m_dd   = re.search(r"delay_diff_samp=([\-0-9]+)", pstr)
    m_mod1 = re.search(r"mod1=([A-Za-z0-9]+)", pstr)
    m_mod2 = re.search(r"mod2=([A-Za-z0-9]+)", pstr)

    mod1 = m_mod1.group(1) if m_mod1 else None
    mod2 = m_mod2.group(1) if m_mod2 else None
    # 兼容旧的 QPSK test_all：params 末尾有 "QPSK"
    if (mod1 is None or mod2 is None) and ("QPSK" in pstr):
        mod1 = mod1 or "QPSK"
        mod2 = mod2 or "QPSK"

    return {
        'snr': snr,
        'amp': amp,
        'f_off1': float(m_f1.group(1)) if m_f1 else None,
        'f_off2': float(m_f2.group(1)) if m_f2 else None,
        'phi1': float(m_p1.group(1)) if m_p1 else None,
        'phi2': float(m_p2.group(1)) if m_p2 else None,
        'rep': int(m_rep.group(1)) if m_rep else None,
        'delay1': int(m_d1.group(1)) if m_d1 else None,
        'delay2': int(m_d2.group(1)) if m_d2 else None,
        'delay_diff': int(m_dd.group(1)) if m_dd else None,
        'mod1': mod1,
        'mod2': mod2,
    }

# ==================== Dataset ====================
class GenericTestDataset(Dataset):
    def __init__(self, entries):
        self.entries = entries
    def __len__(self):
        return len(self.entries)
    def __getitem__(self, idx):
        e = self.entries[idx]
        def c2ri(x):
            x = np.asarray(x)
            return np.stack([x.real.astype(np.float32), x.imag.astype(np.float32)], axis=0)

        raw = e.get('params', "")
        params_str = ", ".join(map(str, raw)) if isinstance(raw, (list, tuple)) else str(raw)
        b1 = np.asarray(e.get('bits1', np.array([-1], dtype=np.int8)), dtype=np.int8)
        b2 = np.asarray(e.get('bits2', np.array([-1], dtype=np.int8)), dtype=np.int8)
        return {
            'mixsignal_ri': c2ri(e['mixsignal']),
            'rfsignal1_ri': c2ri(e['rfsignal1']),
            'rfsignal2_ri': c2ri(e['rfsignal2']),
            'bits1': b1,
            'bits2': b2,
            'params': params_str,
        }

# ==================== Collate Function ====================
def collate_fn_pad(batch):
    """
    自定义 collate 函数，处理长度不一致的信号。
    将所有信号 padding 到 batch 内的最大长度。
    """
    # 找到 batch 内所有信号的最大长度
    max_len = 0
    for sample in batch:
        max_len = max(max_len, sample['mixsignal_ri'].shape[1])
        max_len = max(max_len, sample['rfsignal1_ri'].shape[1])
        max_len = max(max_len, sample['rfsignal2_ri'].shape[1])
    
    # 收集所有需要 padding 的 tensor
    mixsignal_list = []
    rfsignal1_list = []
    rfsignal2_list = []
    bits1_list = []
    bits2_list = []
    params_list = []
    original_lengths = []
    
    for sample in batch:
        mix_ri = sample['mixsignal_ri']  # (2, T)
        rf1_ri = sample['rfsignal1_ri']  # (2, T)
        rf2_ri = sample['rfsignal2_ri']  # (2, T)
        
        # 记录原始长度
        orig_len = mix_ri.shape[1]
        original_lengths.append(orig_len)
        
        # Padding 到最大长度（在最后一个维度）
        if mix_ri.shape[1] < max_len:
            pad_width = max_len - mix_ri.shape[1]
            mix_ri = np.pad(mix_ri, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        if rf1_ri.shape[1] < max_len:
            pad_width = max_len - rf1_ri.shape[1]
            rf1_ri = np.pad(rf1_ri, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        if rf2_ri.shape[1] < max_len:
            pad_width = max_len - rf2_ri.shape[1]
            rf2_ri = np.pad(rf2_ri, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        
        mixsignal_list.append(torch.from_numpy(mix_ri))
        rfsignal1_list.append(torch.from_numpy(rf1_ri))
        rfsignal2_list.append(torch.from_numpy(rf2_ri))
        bits1_list.append(torch.from_numpy(sample['bits1']))
        bits2_list.append(torch.from_numpy(sample['bits2']))
        params_list.append(sample['params'])
    
    # Stack 所有 tensor
    return {
        'mixsignal_ri': torch.stack(mixsignal_list, dim=0),  # (B, 2, max_len)
        'rfsignal1_ri': torch.stack(rfsignal1_list, dim=0),  # (B, 2, max_len)
        'rfsignal2_ri': torch.stack(rfsignal2_list, dim=0),  # (B, 2, max_len)
        'bits1': bits1_list,  # List of tensors (不同长度)
        'bits2': bits2_list,  # List of tensors (不同长度)
        'params': params_list,  # List of strings
        'original_lengths': torch.tensor(original_lengths, dtype=torch.long),  # (B,)
    }

# ==================== 主流程 ====================
def main():
    parser = argparse.ArgumentParser("DDP Inference for SignalSeparator (multi-mod, CSV only)")
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--test_data_path', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='./results/infer')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--amp', action='store_true', default=True,
                        help='use torch.cuda.amp for inference')
    args = parser.parse_args()

    setup_ddp(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    rank, world = get_rank(), get_world_size()
    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"[DDP] world_size={world}, device={device}")
        print(f"[IO] ckpt={args.ckpt_path}")
        print(f"[IO] test={args.test_data_path}")
        print(f"[IO] out={args.out_dir}")

    # 加载模型
    model = SignalSeparator().to(device)
    state = torch.load(args.ckpt_path, map_location='cpu')
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        model.load_state_dict(state, strict=False)
    model.eval()
    if rank == 0:
        print("Params:", sum(p.numel() for p in model.parameters()))

    # 读取数据 & DDP 切分
    loaded_data = torch.load(args.test_data_path)  # list of dict
    dataset = GenericTestDataset(loaded_data)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank,
                                 shuffle=False, drop_last=False) if dist_is_initialized() else None
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False if sampler else False,
                        num_workers=args.num_workers, pin_memory=True, sampler=sampler,
                        collate_fn=collate_fn_pad)

    TMP_DIR = os.path.join(args.out_dir, "_tmp_csv")
    if rank == 0:
        os.makedirs(TMP_DIR, exist_ok=True)

    results = []
    autocast_enabled = (device.type == 'cuda') and args.amp

    with torch.no_grad():
        for bi, batch in enumerate(loader, 1):
            x = batch['mixsignal_ri'].to(device)  # (B,2,T)

            if autocast_enabled:
                with torch.cuda.amp.autocast():
                    y = model(x)
            else:
                y = model(x)

            if isinstance(y, (tuple, list)):
                # 模型返回 list[4], 每个 [B,1,T] -> [B,4,T]
                y = torch.cat(y, dim=1)

            y_np = y.detach().cpu().numpy()
            p1 = y_np[:, 0:2, :]
            p2 = y_np[:, 2:4, :]

            g1 = batch['rfsignal1_ri'].cpu().numpy()
            g2 = batch['rfsignal2_ri'].cpu().numpy()
            original_lengths = batch['original_lengths'].cpu().numpy()  # (B,)

            # 归一化 MSE（只计算原始长度部分）
            B = p1.shape[0]
            loss1_list = []
            loss2_list = []
            for i in range(B):
                orig_len = int(original_lengths[i])
                p1_i = p1[i, :, :orig_len]
                p2_i = p2[i, :, :orig_len]
                g1_i = g1[i, :, :orig_len]
                g2_i = g2[i, :, :orig_len]
                loss1_i = ((p1_i - g1_i) ** 2).mean() / (np.linalg.norm(g1_i) + 1e-12)
                loss2_i = ((p2_i - g2_i) ** 2).mean() / (np.linalg.norm(g2_i) + 1e-12)
                loss1_list.append(loss1_i)
                loss2_list.append(loss2_i)
            loss1 = np.array(loss1_list)
            loss2 = np.array(loss2_list)

            for i in range(B):
                orig_len = int(original_lengths[i])
                # 截断到原始长度
                pr1 = (p1[i,0,:orig_len] + 1j*p1[i,1,:orig_len]).astype(np.complex64)
                pr2 = (p2[i,0,:orig_len] + 1j*p2[i,1,:orig_len]).astype(np.complex64)
                gt1 = (g1[i,0,:orig_len] + 1j*g1[i,1,:orig_len]).astype(np.complex64)
                gt2 = (g2[i,0,:orig_len] + 1j*g2[i,1,:orig_len]).astype(np.complex64)

                params_batch = batch['params']
                if isinstance(params_batch, (list, tuple)):
                    params_i = params_batch[i]
                else:
                    params_i = str(params_batch)
                meta = parse_params_all(str(params_i))

                f1 = meta.get('f_off1', 0.0) or 0.0
                f2 = meta.get('f_off2', 0.0) or 0.0
                phi1 = meta.get('phi1', 0.0) or 0.0
                phi2 = meta.get('phi2', 0.0) or 0.0
                mod1 = meta.get('mod1') or "QPSK"
                mod2 = meta.get('mod2') or "QPSK"

                n = np.arange(len(pr1))
                t = n / fs

                # 理想补偿 CFO + 相位
                pr1_c = pr1 * np.exp(-1j * (2 * np.pi * float(f1) * t + float(phi1)))
                pr2_c = pr2 * np.exp(-1j * (2 * np.pi * float(f2) * t + float(phi2)))
                gt1_c = gt1 * np.exp(-1j * (2 * np.pi * float(f1) * t + float(phi1)))
                gt2_c = gt2 * np.exp(-1j * (2 * np.pi * float(f2) * t + float(phi2)))

                # MF + 抽样 + guard + 幅度归一化
                ps1 = mf_and_sample(pr1_c, sps, rc, num_taps)
                ps2 = mf_and_sample(pr2_c, sps, rc, num_taps)
                gs1 = mf_and_sample(gt1_c, sps, rc, num_taps)
                gs2 = mf_and_sample(gt2_c, sps, rc, num_taps)

                L1 = min(len(ps1), len(gs1))
                L2 = min(len(ps2), len(gs2))
                if L1 <= 0 or L2 <= 0:
                    continue
                ps1 = ps1[:L1]; gs1 = gs1[:L1]
                ps2 = ps2[:L2]; gs2 = gs2[:L2]

                # 相位对齐
                ps1_aligned = align_phase(gs1, ps1)
                ps2_aligned = align_phase(gs2, ps2)

                evm1 = evm_rms(gs1, ps1_aligned)
                evm2 = evm_rms(gs2, ps2_aligned)

                # bits 参考（bits1/bits2 是 list of tensors）
                b1_full = batch['bits1'][i].cpu().numpy().astype(np.int8)
                b2_full = batch['bits2'][i].cpu().numpy().astype(np.int8)

                bps1 = BITS_PER_SYMBOL.get(mod1.upper(), 2)
                bps2 = BITS_PER_SYMBOL.get(mod2.upper(), 2)

                b1_ref = slice_bits_to_match_syms(b1_full, len(gs1), bps1)
                b2_ref = slice_bits_to_match_syms(b2_full, len(gs2), bps2)

                # 解调
                b1_hat = demod_by_mod(ps1_aligned, mod1)
                b2_hat = demod_by_mod(ps2_aligned, mod2)

                # 计算 SER：按符号分组比较比特组合
                # 确保比特序列长度是 bps 的倍数
                Lb1 = min(len(b1_hat), len(b1_ref))
                Lb2 = min(len(b2_hat), len(b2_ref))
                Lb1 = (Lb1 // bps1) * bps1  # 向下取整到符号边界
                Lb2 = (Lb2 // bps2) * bps2
                
                if Lb1 > 0 and Lb2 > 0:
                    # 将比特序列按符号分组（每 bps 个比特为一个符号）
                    # 比较每个符号对应的比特组合是否相同
                    n_syms1 = Lb1 // bps1
                    n_syms2 = Lb2 // bps2
                    
                    # 将比特序列重塑为 (n_syms, bps) 形状
                    b1_ref_syms = b1_ref[:Lb1].reshape(n_syms1, bps1)
                    b1_hat_syms = b1_hat[:Lb1].reshape(n_syms1, bps1)
                    b2_ref_syms = b2_ref[:Lb2].reshape(n_syms2, bps2)
                    b2_hat_syms = b2_hat[:Lb2].reshape(n_syms2, bps2)
                    
                    # 比较每个符号的比特组合是否相同
                    # 使用 np.any(axis=1) 来检查每个符号的比特是否不同
                    sym_errors1 = np.any(b1_ref_syms != b1_hat_syms, axis=1)
                    sym_errors2 = np.any(b2_ref_syms != b2_hat_syms, axis=1)
                    
                    ser1 = float(np.mean(sym_errors1)) if n_syms1 > 0 else 1.0
                    ser2 = float(np.mean(sym_errors2)) if n_syms2 > 0 else 1.0
                else:
                    ser1 = 1.0
                    ser2 = 1.0
                
                ser = 0.5 * (ser1 + ser2)

                phi_diff = float(wrap_2pi(phi2 - phi1))

                results.append({
                    'loss1': float(loss1[i]),
                    'loss2': float(loss2[i]),
                    'SER1': ser1, 'SER2': ser2, 'SER': ser,
                    'snr': meta.get('snr'),
                    'amp': meta.get('amp'),
                    'f1': meta.get('f_off1'),
                    'f2': meta.get('f_off2'),
                    'phi1': phi1,
                    'phi2': phi2,
                    'phi_diff': phi_diff,
                    'rep': meta.get('rep'),
                    'delay1': meta.get('delay1'),
                    'delay2': meta.get('delay2'),
                    'delay_diff': meta.get('delay_diff'),
                    'mod1': mod1,
                    'mod2': mod2,
                    'evm1': evm1,
                    'evm2': evm2,
                })

            if rank == 0:
                print(f"[Rank0][Batch {bi}] done.")

    # 每个 rank 写各自的 csv
    tmp_csv = os.path.join(TMP_DIR, f"metrics_rank{rank}.csv")
    os.makedirs(TMP_DIR, exist_ok=True)
    pd.DataFrame(results).to_csv(tmp_csv, index=False)

    # 同步并合并
    if dist_is_initialized():
        dist.barrier()

    if rank == 0:
        dfs = []
        for r in range(world):
            path = os.path.join(TMP_DIR, f"metrics_rank{r}.csv")
            if os.path.exists(path):
                dfs.append(pd.read_csv(path))
        df = pd.concat(dfs, ignore_index=True) if len(dfs) else pd.DataFrame()
        final_csv = os.path.join(args.out_dir, "metrics_all.csv")
        df.to_csv(final_csv, index=False)
        print(f"[Rank0] merged metrics saved to: {final_csv}")
        if "SER" in df.columns and len(df):
            print("Overall mean SER:", df["SER"].mean())
        else:
            print("Overall mean SER: N/A")

    cleanup_ddp()

if __name__ == '__main__':
    main()
