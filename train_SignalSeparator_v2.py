# -*- coding: utf-8 -*-
import os, re, math, random, argparse
from collections import defaultdict
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict, Any, Iterator, Optional, Tuple
from torch.utils.data import IterableDataset, DataLoader
from itertools import islice
from datetime import timedelta
from send_email import send_email

from model_complex import SignalSeparator  # 你的模型`


# ===== 可选调试环境变量（也可在外部shell设置） =====
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "INFO")
# TORCH_DISTRIBUTED_DEBUG: OFF/INFO/WARN/ERROR/DETAIL (DETAIL会产生大量输出)
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "INFO")  # 从 DETAIL 改为 INFO 减少输出
# NCCL_DEBUG: WARN/INFO/TRACE (WARN只显示警告，INFO显示更多信息)
os.environ.setdefault("NCCL_DEBUG", "INFO")  # 从 WARN 改为 INFO，或设为空字符串关闭
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
# 改善 P2P 被禁用时的通信（通过 PCIe 或网络）
# 注意：P2P 被禁用时，NCCL 会自动使用 PCIe 或网络进行通信
# 这些设置可以帮助改善通信性能，但通常不需要手动设置
# os.environ.setdefault("NCCL_IB_DISABLE", "0")  # 如果 InfiniBand 可用，启用它
# os.environ.setdefault("NCCL_SHM_DISABLE", "0")  # 启用共享内存通信

# =========================
# DDP: 工具 & 初始化
# =========================
def is_dist_avail_and_initialized():
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def get_rank():
    return torch.distributed.get_rank() if is_dist_avail_and_initialized() else 0

def get_world_size():
    return torch.distributed.get_world_size() if is_dist_avail_and_initialized() else 1

def setup_distributed(backend: str = "nccl", timeout_seconds: int = 7200):
    if is_dist_avail_and_initialized():
        return
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch.distributed.init_process_group(
            backend=backend, init_method="env://",
            timeout=timedelta(seconds=timeout_seconds)
        )

def cleanup_distributed():
    if is_dist_avail_and_initialized():
        try:
            torch.distributed.destroy_process_group()
        except Exception:
            pass

# ----------------------------
# 数据分片检索（避免 glob，兼容带 [] 的文件名）
# ----------------------------
def _find_shard_files(dataset_dir: str, mode_prefix: str) -> List[str]:
    mode_prefix = mode_prefix.strip()
    shard_files, single_files = [], []
    for entry in os.scandir(dataset_dir):
        if not entry.is_file():
            continue
        name = entry.name
        if not name.endswith(".pth"):
            continue
        if not name.startswith(mode_prefix):
            continue
        if "shard" in name:
            shard_files.append(entry.path)
        else:
            single_files.append(entry.path)

    if shard_files:
        def shard_key(path: str) -> int:
            name = os.path.basename(path)
            m1 = re.search(r"_shard(\d+)\.pth$", name)
            if m1: return int(m1.group(1))
            m2 = re.search(r"_shard(\d+)-of-\d+\.pth$", name)
            if m2: return int(m2.group(1))
            return 10**9
        return sorted(shard_files, key=shard_key)
    return sorted(single_files)

def _parse_snr_from_params(params) -> float:
    """
    从params中解析SNR值（dB）
    params可能是tuple或str，格式如: (snr_db, amp, sps, 'f_off1=...', ...)
    """
    if isinstance(params, (tuple, list)):
        # 第一个元素通常是SNR
        if len(params) > 0:
            try:
                return float(params[0])
            except (ValueError, TypeError):
                pass
    elif isinstance(params, str):
        # 尝试从字符串中提取SNR
        m = re.search(r'snr[=:]?([0-9.]+)', params, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except (ValueError, TypeError):
                pass
    # 默认值（如果无法解析）
    return 15.0  # 默认15dB

def _parse_params(params) -> Dict[str, Any]:
    """
    从params中解析所有参数（SNR、调制方式、频偏、相偏等）
    """
    result = {
        'snr_db': 15.0,
        'mod1': 'QPSK',
        'mod2': 'QPSK',
        'f_off1': 0.0,
        'f_off2': 0.0,
        'phi1': 0.0,
        'phi2': 0.0,
    }
    
    param_str = str(params) if not isinstance(params, str) else params
    
    # 解析SNR
    m = re.search(r'snr[=:]?([0-9.]+)', param_str, re.IGNORECASE)
    if m:
        try:
            result['snr_db'] = float(m.group(1))
        except (ValueError, TypeError):
            pass
    
    # 解析调制方式
    m = re.search(r'mod1=([A-Za-z0-9]+)', param_str, re.IGNORECASE)
    if m:
        result['mod1'] = m.group(1).upper()
    m = re.search(r'mod2=([A-Za-z0-9]+)', param_str, re.IGNORECASE)
    if m:
        result['mod2'] = m.group(1).upper()
    
    # 解析频偏
    m = re.search(r'f_off1=([\-0-9.]+)\s*Hz', param_str, re.IGNORECASE)
    if m:
        try:
            result['f_off1'] = float(m.group(1))
        except (ValueError, TypeError):
            pass
    m = re.search(r'f_off2=([\-0-9.]+)\s*Hz', param_str, re.IGNORECASE)
    if m:
        try:
            result['f_off2'] = float(m.group(1))
        except (ValueError, TypeError):
            pass
    
    # 解析相偏
    m = re.search(r'phi1=([\-0-9.]+)\s*rad', param_str, re.IGNORECASE)
    if m:
        try:
            result['phi1'] = float(m.group(1))
        except (ValueError, TypeError):
            pass
    m = re.search(r'phi2=([\-0-9.]+)\s*rad', param_str, re.IGNORECASE)
    if m:
        try:
            result['phi2'] = float(m.group(1))
        except (ValueError, TypeError):
            pass
    
    return result

def _entry_to_sample(e: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    def to_float(x: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        if np.iscomplexobj(x):
            real = torch.from_numpy(np.asarray(x.real, dtype=np.float32))
            imag = torch.from_numpy(np.asarray(x.imag, dtype=np.float32))
        else:
            x = np.asarray(x, dtype=np.float32)
            real = torch.from_numpy(x[..., 0])
            imag = torch.from_numpy(x[..., 1])
        return real, imag

    mix_r, mix_i = to_float(e['mixsignal'])
    r1_r, r1_i   = to_float(e['rfsignal1'])
    r2_r, r2_i   = to_float(e['rfsignal2'])
    
    # 解析所有参数（用于噪声一致性损失和EVM损失）
    params_dict = _parse_params(e.get('params', None))
    snr_db = params_dict['snr_db']
    mod1 = params_dict['mod1']
    mod2 = params_dict['mod2']
    f_off1 = params_dict['f_off1']
    f_off2 = params_dict['f_off2']
    phi1 = params_dict['phi1']
    phi2 = params_dict['phi2']

    return {
        'mixsignal_real': mix_r, 'mixsignal_imag': mix_i,
        'rfsignal1_real': r1_r,  'rfsignal1_imag': r1_i,
        'rfsignal2_real': r2_r,  'rfsignal2_imag': r2_i,
        'snr_db': torch.tensor(snr_db, dtype=torch.float32),
        'mod1': mod1,
        'mod2': mod2,
        'f_off1': torch.tensor(f_off1, dtype=torch.float32),
        'f_off2': torch.tensor(f_off2, dtype=torch.float32),
        'phi1': torch.tensor(phi1, dtype=torch.float32),
        'phi2': torch.tensor(phi2, dtype=torch.float32),
    }

def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, List[Any]] = {}
    for sample in batch:
        for k, v in sample.items():
            out.setdefault(k, []).append(v)
    
    result = {}
    for k, vlist in out.items():
        if k in ['mod1', 'mod2']:
            # 字符串列表，直接保留
            result[k] = vlist
        else:
            # 张量，stack
            result[k] = torch.stack(vlist, dim=0)
    return result

def _get_shard_sizes(shard_files: List[str]) -> List[int]:
    """
    获取每个分片文件的样本数。
    注意：需要加载每个分片文件，对于大数据集可能较慢。
    """
    sizes = []
    total_shards = len(shard_files)
    rank = get_rank()
    
    if rank == 0:
        print(f"[Data] 正在加载 {total_shards} 个分片文件以获取样本数...")
    
    for idx, p in enumerate(shard_files):
        if rank == 0 and (idx + 1) % max(1, total_shards // 10) == 0:
            print(f"[Data] 已处理 {idx + 1}/{total_shards} 个分片...")
        entries = torch.load(p, map_location='cpu')  # 使用CPU加载，避免占用GPU内存
        sizes.append(len(entries))
        del entries
    
    if rank == 0:
        print(f"[Data] 所有分片加载完成，总样本数: {sum(sizes)}")
    
    return sizes

def _build_sample_level_plan(
    shard_files: List[str],
    train_ratio: float,
    seed: int = 2025
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int]:
    assert 0.0 < train_ratio < 1.0
    sizes = _get_shard_sizes(shard_files)
    total = sum(sizes)
    target_train = int(round(total * train_ratio))

    alloc_floor = [int(math.floor(s * train_ratio)) for s in sizes]
    fracs = [s * train_ratio - f for s, f in zip(sizes, alloc_floor)]
    deficit = target_train - sum(alloc_floor)

    order = sorted(range(len(sizes)), key=lambda i: fracs[i], reverse=True)
    alloc = alloc_floor[:]
    for i in range(max(0, deficit)):
        alloc[order[i]] += 1

    train_plan, val_plan = [], []
    for i, (path, sz, n_train) in enumerate(zip(shard_files, sizes, alloc)):
        local_rng = random.Random(seed ^ (i + 1))
        idx = list(range(sz))
        local_rng.shuffle(idx)
        train_idx = idx[:n_train]
        val_idx   = idx[n_train:]
        if train_idx:
            train_plan.append({'path': path, 'indices': train_idx})
        if val_idx:
            val_plan.append({'path': path, 'indices': val_idx})

    train_samples = sum(len(x['indices']) for x in train_plan)
    val_samples   = sum(len(x['indices']) for x in val_plan)
    return train_plan, val_plan, train_samples, val_samples

# ----------------------------
# 流式 IterableDataset（按 rank 切 plan）
# ----------------------------
class StreamSignalDatasetPlan(IterableDataset):
    def __init__(
        self,
        plan: List[Dict[str, Any]],
        shuffle_shards_per_epoch: bool = True,
        shuffle_within_shard: bool = True,
        seed: int = 2025,
        dist_rank: int = 0,
        dist_world_size: int = 1,
    ):
        super().__init__()
        self.plan_all = list(plan)
        self.shuffle_shards_per_epoch = shuffle_shards_per_epoch
        self.shuffle_within_shard = shuffle_within_shard
        self.base_seed = seed
        self.dist_rank = dist_rank
        self.dist_world_size = dist_world_size
        
        # ★★★ 修复：按样本分配，而不是按分片文件分配
        # 将所有样本索引收集起来，然后按 rank 分配
        all_samples = []
        for item in self.plan_all:
            for idx in item['indices']:
                all_samples.append((item['path'], idx))
        
        # 按 rank 分配样本（每个 rank 处理分配给它的样本）
        self.samples_for_rank = all_samples[self.dist_rank::self.dist_world_size]
        
        # 为了兼容原有代码，仍然保留 plan 结构，但只包含当前 rank 的样本
        # 按分片文件分组
        rank_plan_dict = defaultdict(list)
        for path, idx in self.samples_for_rank:
            rank_plan_dict[path].append(idx)
        
        self.plan = [{'path': path, 'indices': indices} for path, indices in rank_plan_dict.items()]

    def _iter_one_epoch(self, worker_info: Optional[torch.utils.data.get_worker_info]) -> Iterator[Dict[str, torch.Tensor]]:
        rng = random.Random(self.base_seed + (0 if worker_info is None else worker_info.id))
        plan = list(self.plan)
        if worker_info is not None:
            num_workers = worker_info.num_workers
            wid = worker_info.id
            plan = plan[wid::num_workers]

        if self.shuffle_shards_per_epoch:
            rng.shuffle(plan)

        for j, item in enumerate(plan):
            shard_path = item['path']
            indices = list(item['indices'])
            if self.shuffle_within_shard:
                local_rng = random.Random(self.base_seed ^ (j + 1))
                local_rng.shuffle(indices)
            entries = torch.load(shard_path)
            for idx in indices:
                yield _entry_to_sample(entries[idx])
            del entries

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        return self._iter_one_epoch(worker_info)

def prepare_dataloaders(
    dataset_path: str,
    mode: str,
    batch_size: int = 64,
    train_ratio: float = 0.9,
    num_workers: int = 0,           # 先设 0 跑通；稳定后再提到 2/4 并配合 pin_memory=True
    seed: int = 2025,
    dist_rank: int = 0,
    dist_world_size: int = 1,
):
    rank = get_rank()
    if rank == 0:
        print(f"[Data] 开始准备数据加载器...")
        print(f"[Data] 数据集路径: {dataset_path}")
        print(f"[Data] 模式前缀: {mode}")
    
    shard_files = _find_shard_files(dataset_path, mode)
    if rank == 0:
        print(f"[Data] 找到 {len(shard_files)} 个分片文件")
    
    if len(shard_files) == 0:
        error_msg = f"未找到分片文件，检查目录与前缀是否正确：{dataset_path} / {mode}"
        if rank == 0:
            send_email(text=f"训练失败: {error_msg}")
        raise AssertionError(error_msg)

    if rank == 0:
        print(f"[Data] 开始构建样本级计划（需要加载所有分片以获取大小）...")
    train_plan, val_plan, train_samples, val_samples = _build_sample_level_plan(
        shard_files, train_ratio=train_ratio, seed=seed
    )
    if rank == 0:
        print(f"[Data] 样本级计划构建完成")

    train_dataset = StreamSignalDatasetPlan(
        plan=train_plan, shuffle_shards_per_epoch=True, shuffle_within_shard=True,
        seed=seed, dist_rank=dist_rank, dist_world_size=dist_world_size
    )
    val_dataset = StreamSignalDatasetPlan(
        plan=val_plan, shuffle_shards_per_epoch=False, shuffle_within_shard=False,
        seed=seed, dist_rank=dist_rank, dist_world_size=dist_world_size
    )

    # 本 rank 样本与批次数（注意 drop_last=True）
    train_samples_rank = sum(len(item['indices']) for item in train_dataset.plan)
    val_samples_rank   = sum(len(item['indices']) for item in val_dataset.plan)
    train_batches_rank = max(1, train_samples_rank // batch_size)                  # drop_last=True
    val_batches_rank   = max(1, (val_samples_rank + batch_size - 1) // batch_size) # drop_last=False

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        drop_last=True,
        collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        drop_last=False,
        collate_fn=_collate_fn,
    )

    # 在所有 rank 上打印，帮助调试数据分配问题
    print(f"[Rank {rank}] [Data] shards={len(shard_files)} | train_total≈{train_samples} | val_total≈{val_samples} | ratio={train_ratio:.2f}")
    print(f"[Rank {rank}] [Data] samples_rank(train)={train_samples_rank}, batches_rank(train)={train_batches_rank}")
    print(f"[Rank {rank}] [Data] samples_rank(val)  ={val_samples_rank}, batches_rank(val)  ={val_batches_rank}")
    print(f"[Rank {rank}] [Data] plan_items(train)={len(train_dataset.plan)}, plan_items(val)={len(val_dataset.plan)}")
    
    # 检查数据分配是否合理
    if train_samples_rank == 0:
        error_msg = f"[Rank {rank}] 错误：训练集样本数为 0！请检查数据分配逻辑。"
        if rank == 0:
            send_email(text=f"训练失败: {error_msg}")
        raise ValueError(error_msg)
    if train_batches_rank < 10 and rank == 0:
        print(f"[WARN] Rank {rank} 的批次数很少 ({train_batches_rank})，可能导致训练不稳定。")

    return train_loader, val_loader, train_batches_rank, val_batches_rank

# =========================
# 学习率调度：Warmup + Cosine
# =========================

class WarmupCosineSchedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.0, last_epoch: int = -1):
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps = max(self.warmup_steps + 1, int(total_steps))
        self.min_lr_ratio = float(min_lr_ratio)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                lr = base_lr * float(step) / float(self.warmup_steps)
            else:
                t = (step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
                cosv = 0.5 * (1 + math.cos(math.pi * t))
                lr = base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosv)
            lrs.append(lr)
        return lrs

# =========================
# EMA
# =========================
class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=(1.0 - self.decay))

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name].data)

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name].data)
        self.backup = {}

# =========================
# 损失
# =========================
def normalized_time_mse(pred_2ch: torch.Tensor, tgt_2ch: torch.Tensor, eps: float = 1e-12, max_ratio: float = 1000.0) -> torch.Tensor:
    # 采用均方能量做归一化：确保分子/分母量纲一致，避免因使用 L2 范数（sqrt）导致尺度异常
    # 使用对数损失来压缩大值，提高数值稳定性
    diff2 = (pred_2ch - tgt_2ch) ** 2
    num = diff2.mean(dim=[1, 2])
    # 使用目标信号的均方能量作为分母（而不是 L2 范数），这样当目标能量接近 0 时更稳定
    den = torch.mean(tgt_2ch ** 2, dim=[1, 2]) + eps
    ratio = num / den
    # 裁剪ratio到合理范围，防止异常大值
    ratio = torch.clamp(ratio, min=0.0, max=max_ratio)
    # 使用 log(1 + ratio) 来压缩大值，提高数值稳定性
    return torch.log1p(ratio)  # log(1 + ratio)，数值稳定

# =========================
# RRC滤波器（用于EVM损失）
# =========================
def create_rrc_filter_torch(beta: float = 0.33, sps: int = 8, num_taps: int = 64, device='cpu'):
    """
    创建RRC滤波器的torch版本
    """
    t = torch.arange(-num_taps//2, num_taps//2, dtype=torch.float32, device=device) / sps
    # sinc(x) = sin(πx) / (πx)
    with torch.no_grad():
        pi_t = math.pi * t
        sinc_val = torch.sin(pi_t) / (pi_t + 1e-12)
        sinc_val[t.abs() < 1e-8] = 1.0  # t=0处
    h = sinc_val * torch.cos(math.pi * beta * t) / (1 - (2 * beta * t) ** 2 + 1e-12)
    # 处理NaN（分母为0处）
    h[torch.isnan(h)] = 1.0 - beta + (4 * beta / math.pi)
    # 归一化
    h = h / torch.sqrt(torch.sum(h**2) + 1e-12)
    return h

def evm_loss(pred1_2ch: torch.Tensor, pred2_2ch: torch.Tensor,
             tgt1_2ch: torch.Tensor, tgt2_2ch: torch.Tensor,
             mod1_list: List[str], mod2_list: List[str],
             f_off1: torch.Tensor, f_off2: torch.Tensor,
             phi1: torch.Tensor, phi2: torch.Tensor,
             fs: float = 12e6, sps: int = 8, beta: float = 0.33, num_taps: int = 64,
             eps: float = 1e-12) -> torch.Tensor:
    """
    EVM损失: LEVM = 1/2 * ( (Σk ||ẑk^(1) - zk^(1)||² / Σk ||zk^(1)||²) + (Σk ||ẑk^(2) - zk^(2)||² / Σk ||zk^(2)||²) )
    
    对预测信号和真实信号进行解调链路处理：
    1. 频偏/相偏补偿
    2. RRC卷积
    3. 下采样（每sps个样本取一个）
    4. 计算符号级EVM
    
    Args:
        pred1_2ch, pred2_2ch: (B, 2, T) 预测信号 [real, imag]
        tgt1_2ch, tgt2_2ch: (B, 2, T) 真实信号 [real, imag]
        mod1_list, mod2_list: List[str] 调制方式列表
        f_off1, f_off2: (B,) 频偏（Hz）
        phi1, phi2: (B,) 相偏（rad）
        fs: 采样频率
        sps: 每符号采样数
        beta: RRC滚降因子
        num_taps: RRC滤波器抽头数
    
    Returns:
        loss: (B,) 每个样本的EVM损失
    """
    device = pred1_2ch.device
    B, _, T = pred1_2ch.shape
    
    # 创建RRC滤波器
    rrc_filter = create_rrc_filter_torch(beta, sps, num_taps, device=device)  # (num_taps,)
    rrc_filter = rrc_filter.view(1, 1, -1)  # (1, 1, num_taps)
    
    # 时间轴
    t = torch.arange(T, dtype=torch.float32, device=device) / fs  # (T,)
    
    # 初始化损失
    evm_losses = []
    
    for b in range(B):
        # 处理信号1
        pred1_cplx = pred1_2ch[b, 0, :] + 1j * pred1_2ch[b, 1, :]  # (T,)
        tgt1_cplx = tgt1_2ch[b, 0, :] + 1j * tgt1_2ch[b, 1, :]  # (T,)
        
        # 频偏/相偏补偿（理想补偿）
        pred1_comp = pred1_cplx * torch.exp(-1j * (2 * math.pi * f_off1[b] * t + phi1[b]))
        tgt1_comp = tgt1_cplx * torch.exp(-1j * (2 * math.pi * f_off1[b] * t + phi1[b]))
        
        # RRC卷积（对复数信号，分别对real和imag做卷积）
        pred1_real = torch.nn.functional.conv1d(
            pred1_comp.real.unsqueeze(0).unsqueeze(0), rrc_filter, padding=num_taps//2
        ).squeeze()[:T]
        pred1_imag = torch.nn.functional.conv1d(
            pred1_comp.imag.unsqueeze(0).unsqueeze(0), rrc_filter, padding=num_taps//2
        ).squeeze()[:T]
        pred1_mf = pred1_real + 1j * pred1_imag
        
        tgt1_real = torch.nn.functional.conv1d(
            tgt1_comp.real.unsqueeze(0).unsqueeze(0), rrc_filter, padding=num_taps//2
        ).squeeze()[:T]
        tgt1_imag = torch.nn.functional.conv1d(
            tgt1_comp.imag.unsqueeze(0).unsqueeze(0), rrc_filter, padding=num_taps//2
        ).squeeze()[:T]
        tgt1_mf = tgt1_real + 1j * tgt1_imag
        
        # 下采样（每sps个样本取一个，找最佳偏移）
        guard_sym = num_taps // sps
        best_offset = 0
        best_eng = -1.0
        for off in range(sps):
            sym = tgt1_mf[off::sps]
            if len(sym) > 0:
                eng = torch.mean(torch.abs(sym)**2)
                if eng > best_eng:
                    best_eng = eng
                    best_offset = off
        
        pred1_syms = pred1_mf[best_offset::sps]
        tgt1_syms = tgt1_mf[best_offset::sps]
        
        # 去除guard符号
        if len(pred1_syms) > 2 * guard_sym:
            pred1_syms = pred1_syms[guard_sym:-guard_sym]
            tgt1_syms = tgt1_syms[guard_sym:-guard_sym]
        
        # 归一化
        if len(pred1_syms) > 0 and torch.mean(torch.abs(pred1_syms)) > 0:
            pred1_syms = pred1_syms / torch.mean(torch.abs(pred1_syms))
        if len(tgt1_syms) > 0 and torch.mean(torch.abs(tgt1_syms)) > 0:
            tgt1_syms = tgt1_syms / torch.mean(torch.abs(tgt1_syms))
        
        # 计算EVM1
        if len(pred1_syms) > 0 and len(tgt1_syms) > 0:
            L = min(len(pred1_syms), len(tgt1_syms))
            pred1_syms = pred1_syms[:L]
            tgt1_syms = tgt1_syms[:L]
            evm1_num = torch.mean(torch.abs(pred1_syms - tgt1_syms)**2)
            evm1_den = torch.mean(torch.abs(tgt1_syms)**2) + eps
            evm1 = evm1_num / evm1_den
        else:
            evm1 = torch.tensor(1e6, device=device, dtype=torch.float32)
        
        # 处理信号2（类似）
        pred2_cplx = pred2_2ch[b, 0, :] + 1j * pred2_2ch[b, 1, :]
        tgt2_cplx = tgt2_2ch[b, 0, :] + 1j * tgt2_2ch[b, 1, :]
        
        pred2_comp = pred2_cplx * torch.exp(-1j * (2 * math.pi * f_off2[b] * t + phi2[b]))
        tgt2_comp = tgt2_cplx * torch.exp(-1j * (2 * math.pi * f_off2[b] * t + phi2[b]))
        
        pred2_real = torch.nn.functional.conv1d(
            pred2_comp.real.unsqueeze(0).unsqueeze(0), rrc_filter, padding=num_taps//2
        ).squeeze()[:T]
        pred2_imag = torch.nn.functional.conv1d(
            pred2_comp.imag.unsqueeze(0).unsqueeze(0), rrc_filter, padding=num_taps//2
        ).squeeze()[:T]
        pred2_mf = pred2_real + 1j * pred2_imag
        
        tgt2_real = torch.nn.functional.conv1d(
            tgt2_comp.real.unsqueeze(0).unsqueeze(0), rrc_filter, padding=num_taps//2
        ).squeeze()[:T]
        tgt2_imag = torch.nn.functional.conv1d(
            tgt2_comp.imag.unsqueeze(0).unsqueeze(0), rrc_filter, padding=num_taps//2
        ).squeeze()[:T]
        tgt2_mf = tgt2_real + 1j * tgt2_imag
        
        best_offset2 = 0
        best_eng2 = -1.0
        for off in range(sps):
            sym = tgt2_mf[off::sps]
            if len(sym) > 0:
                eng = torch.mean(torch.abs(sym)**2)
                if eng > best_eng2:
                    best_eng2 = eng
                    best_offset2 = off
        
        pred2_syms = pred2_mf[best_offset2::sps]
        tgt2_syms = tgt2_mf[best_offset2::sps]
        
        if len(pred2_syms) > 2 * guard_sym:
            pred2_syms = pred2_syms[guard_sym:-guard_sym]
            tgt2_syms = tgt2_syms[guard_sym:-guard_sym]
        
        if len(pred2_syms) > 0 and torch.mean(torch.abs(pred2_syms)) > 0:
            pred2_syms = pred2_syms / torch.mean(torch.abs(pred2_syms))
        if len(tgt2_syms) > 0 and torch.mean(torch.abs(tgt2_syms)) > 0:
            tgt2_syms = tgt2_syms / torch.mean(torch.abs(tgt2_syms))
        
        if len(pred2_syms) > 0 and len(tgt2_syms) > 0:
            L = min(len(pred2_syms), len(tgt2_syms))
            pred2_syms = pred2_syms[:L]
            tgt2_syms = tgt2_syms[:L]
            evm2_num = torch.mean(torch.abs(pred2_syms - tgt2_syms)**2)
            evm2_den = torch.mean(torch.abs(tgt2_syms)**2) + eps
            evm2 = evm2_num / evm2_den
        else:
            evm2 = torch.tensor(1e6, device=device, dtype=torch.float32)
        
        # LEVM = 1/2 * (EVM1 + EVM2)
        evm_losses.append(0.5 * (evm1 + evm2))
    
    return torch.stack(evm_losses)  # (B,)

def noise_consistency_loss(pred1_2ch: torch.Tensor, pred2_2ch: torch.Tensor, 
                          mixsignal_2ch: torch.Tensor, snr_db: torch.Tensor, 
                          eps: float = 1e-12) -> torch.Tensor:
    """
    噪声一致性损失: Lnoise = ((Pr - Ptarget_n) / Ptarget_n)²
    
    Args:
        pred1_2ch: (B, 2, T) 预测信号1 [real, imag]
        pred2_2ch: (B, 2, T) 预测信号2 [real, imag]
        mixsignal_2ch: (B, 2, T) 混合信号 [real, imag]
        snr_db: (B,) SNR值（dB）
        eps: 数值稳定性常数
    
    Returns:
        loss: (B,) 每个样本的噪声一致性损失
    """
    # 计算残差噪声: r = y - (x̂1 + x̂2)
    pred_sum = pred1_2ch + pred2_2ch  # (B, 2, T)
    residual = mixsignal_2ch - pred_sum  # (B, 2, T)
    
    # 计算残差噪声功率 Pr = mean(|r|²)
    residual_power = torch.mean(residual ** 2, dim=[1, 2])  # (B,)
    
    # 计算目标噪声功率 Ptarget_n
    # 从SNR计算: SNR = 10*log10(Psignal / Pnoise) => Pnoise = Psignal / 10^(SNR/10)
    # 使用混合信号功率作为信号功率的估计
    signal_power = torch.mean(mixsignal_2ch ** 2, dim=[1, 2])  # (B,)
    snr_linear = torch.pow(10.0, snr_db / 10.0)  # (B,)
    target_noise_power = signal_power / (snr_linear + eps)  # (B,)
    
    # 计算相对误差: (Pr - Ptarget_n) / Ptarget_n
    relative_error = (residual_power - target_noise_power) / (target_noise_power + eps)  # (B,)
    
    # Lnoise = ((Pr - Ptarget_n) / Ptarget_n)²
    loss = relative_error ** 2  # (B,)
    
    return loss

def calculate_loss(output, batch_data, alpha_time=1.0, lambda_noise=0.0, lambda_evm=0.0):
    """
    计算总损失: Ltotal = Lwave + λnoise * Lnoise + λEVM * LEVM
    
    Args:
        output: 模型输出，4个 (B,1,T) 张量的列表
        batch_data: 包含mixsignal, rfsignal1, rfsignal2, snr_db, mod1, mod2, f_off1, f_off2, phi1, phi2
        alpha_time: 波形损失权重
        lambda_noise: 噪声一致性损失权重
        lambda_evm: EVM损失权重
    """
    pred1 = torch.cat([output[0], output[1]], dim=1)  # (B,2,T)
    pred2 = torch.cat([output[2], output[3]], dim=1)  # (B,2,T)
    tgt1  = batch_data['rfsignal1']                   # (B,2,T)
    tgt2  = batch_data['rfsignal2']                   # (B,2,T)
    mixsignal = batch_data['mixsignal']               # (B,2,T)
    snr_db = batch_data['snr_db']                     # (B,)

    # 波形损失 Lwave
    nmse1 = normalized_time_mse(pred1, tgt1)
    nmse2 = normalized_time_mse(pred2, tgt2)
    nmse  = 0.5 * (nmse1 + nmse2)
    nmse = torch.nan_to_num(nmse, nan=1e6, posinf=1e6, neginf=1e6)
    l_wave = alpha_time * nmse.mean()

    # 噪声一致性损失 Lnoise
    l_noise = torch.tensor(0.0, device=l_wave.device, dtype=l_wave.dtype)
    if lambda_noise > 0:
        noise_loss_per_sample = noise_consistency_loss(pred1, pred2, mixsignal, snr_db)
        noise_loss_per_sample = torch.nan_to_num(noise_loss_per_sample, nan=1e6, posinf=1e6, neginf=1e6)
        l_noise = lambda_noise * noise_loss_per_sample.mean()

    # EVM损失 LEVM
    l_evm = torch.tensor(0.0, device=l_wave.device, dtype=l_wave.dtype)
    if lambda_evm > 0:
        evm_loss_per_sample = evm_loss(
            pred1, pred2, tgt1, tgt2,
            batch_data['mod1'], batch_data['mod2'],
            batch_data['f_off1'], batch_data['f_off2'],
            batch_data['phi1'], batch_data['phi2']
        )
        evm_loss_per_sample = torch.nan_to_num(evm_loss_per_sample, nan=1e6, posinf=1e6, neginf=1e6)
        l_evm = lambda_evm * evm_loss_per_sample.mean()

    loss = l_wave + l_noise + l_evm

    # 最后再次保护：若 loss 非法，则返回一个大数（训练端会跳过该 batch 并记录）
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(1e6, device=loss.device, dtype=loss.dtype)
    return loss

def safe_save_model(state_dict, filepath, max_retries=3):
    """
    安全保存模型，使用临时文件然后原子性移动，带重试机制
    """
    import tempfile
    dirname = os.path.dirname(filepath)
    basename = os.path.basename(filepath)
    temp_file = os.path.join(dirname, f'.tmp_{basename}_{os.getpid()}')
    
    for attempt in range(max_retries):
        try:
            # 先保存到临时文件
            torch.save(state_dict, temp_file)
            # 原子性移动到目标文件
            if os.path.exists(filepath):
                os.remove(filepath)
            os.rename(temp_file, filepath)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(0.5 * (attempt + 1))  # 指数退避
                print(f"[WARN] 保存失败 (尝试 {attempt+1}/{max_retries}): {e}, 重试...")
            else:
                # 清理临时文件
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                error_msg = f"保存模型失败 (已重试 {max_retries} 次): {filepath}, 错误: {e}"
                send_email(text=f"训练失败: {error_msg}")
                raise RuntimeError(error_msg)
    return False

def get_loss_weights(epoch_idx: int, args) -> Tuple[float, float, float]:
    """
    分阶段返回损失权重: (alpha_time, lambda_noise, lambda_evm)
    
    阶段划分：
    - 0 <= epoch < mse_epochs: 纯 MSE (alpha_time=1.0, lambda_noise=0.0, lambda_evm=0.0)
    - mse_epochs <= epoch < mse_epochs + noise_epochs: 加入噪声损失 (alpha_time=1.0, lambda_noise=args.lambda_noise, lambda_evm=0.0)
    - epoch >= mse_epochs + noise_epochs: 加入所有损失 (alpha_time=1.0, lambda_noise=args.lambda_noise, lambda_evm=args.lambda_evm)
    """
    # 阶段1：纯MSE
    if epoch_idx < args.mse_epochs:
        return 1.0, 0.0, 0.0
    
    # 阶段2：MSE + 噪声损失
    if epoch_idx < args.mse_epochs + args.noise_epochs:
        return 1.0, args.lambda_noise, 0.0
    
    # 阶段3：所有损失
    return 1.0, args.lambda_noise, args.lambda_evm



# =========================
# 模型/优化/训练流程
# =========================
def initialize_model(device, lr=2e-4):
    model = SignalSeparator().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam
    return model, optimizer

def process_batch(batch, device):
    mixsignal_real = batch['mixsignal_real'].to(device).unsqueeze(1)
    mixsignal_imag = batch['mixsignal_imag'].to(device).unsqueeze(1)
    rfsignal1_real = batch['rfsignal1_real'].to(device).unsqueeze(1)
    rfsignal1_imag = batch['rfsignal1_imag'].to(device).unsqueeze(1)
    rfsignal2_real = batch['rfsignal2_real'].to(device).unsqueeze(1)
    rfsignal2_imag = batch['rfsignal2_imag'].to(device).unsqueeze(1)
    snr_db = batch['snr_db'].to(device)  # (B,)
    f_off1 = batch['f_off1'].to(device)  # (B,)
    f_off2 = batch['f_off2'].to(device)  # (B,)
    phi1 = batch['phi1'].to(device)  # (B,)
    phi2 = batch['phi2'].to(device)  # (B,)
    return {
        'mixsignal': torch.cat([mixsignal_real, mixsignal_imag], dim=1),
        'rfsignal1': torch.cat([rfsignal1_real, rfsignal1_imag], dim=1),
        'rfsignal2': torch.cat([rfsignal2_real, rfsignal2_imag], dim=1),
        'snr_db': snr_db,
        'mod1': batch['mod1'],  # List[str]
        'mod2': batch['mod2'],  # List[str]
        'f_off1': f_off1,
        'f_off2': f_off2,
        'phi1': phi1,
        'phi2': phi2,
    }

def train_epoch(model, train_loader, optimizer, scheduler, scaler, ema, device,
                total_batches: int, rank: int, accum_steps: int,
                alpha_time: float, lambda_noise: float, lambda_evm: float, max_grad_norm: float,
                max_nan_inf_count: int = 100):
    model.train()
    total_loss = 0.0
    step_in_epoch = 0
    nan_inf_count = 0  # 记录NaN/Inf的次数
    
    # 在所有进程上打印，确保能看到所有进程的状态
    print(f"[Rank {rank}] 开始训练 epoch，total_batches={total_batches}")
    
    pbar = tqdm(total=total_batches, disable=(rank != 0), desc=f"Training[α={alpha_time:.3f},λn={lambda_noise:.3f},λe={lambda_evm:.3f}]")

    optimizer.zero_grad(set_to_none=True)
    
    # 确保所有进程都进入训练循环
    if is_dist_avail_and_initialized():
        torch.distributed.barrier()
    print(f"[Rank {rank}] 进入训练循环，准备迭代 {total_batches} 个 batch")
    
    batch_count = 0
    for batch in islice(train_loader, total_batches):
        if batch_count == 0:
            print(f"[Rank {rank}] 开始处理第一个 batch...")
        
        batch_data = process_batch(batch, device)
        if batch_count == 0:
            print(f"[Rank {rank}] 数据已移动到设备，开始前向传播...")
        
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            output = model(batch_data['mixsignal'])
            if batch_count == 0:
                print(f"[Rank {rank}] 前向传播完成，计算损失...")
            loss = calculate_loss(output, batch_data, alpha_time=alpha_time, lambda_noise=lambda_noise, lambda_evm=lambda_evm)
            loss = loss / max(1, accum_steps)

        # 检查loss是否为NaN/Inf
        loss_is_bad = torch.isnan(loss) or torch.isinf(loss)
        if loss_is_bad:
            nan_inf_count += 1
            if rank == 0:
                print(f"[Train][WARN] NaN/Inf loss encountered at step {step_in_epoch}, count={nan_inf_count}/{max_nan_inf_count}")
            # 如果loss是NaN/Inf，将其替换为一个较小的固定值，避免反向传播时出现问题
            # 使用max_grad_norm的平方作为替代值，确保梯度裁剪能正常工作
            loss = torch.tensor(max_grad_norm ** 2, device=loss.device, dtype=loss.dtype) / max(1, accum_steps)
            # 即使loss是NaN/Inf，也继续反向传播（会被梯度裁剪处理）
            # 但跳过累加到total_loss

        if batch_count == 0:
            print(f"[Rank {rank}] 开始反向传播（DDP 梯度同步）...")
        scaler.scale(loss).backward()
        if batch_count == 0:
            print(f"[Rank {rank}] 反向传播完成")
        step_in_epoch += 1
        batch_count += 1

        if step_in_epoch % max(1, accum_steps) == 0:
            if step_in_epoch == accum_steps:
                print(f"[Rank {rank}] 开始 optimizer step（累积 {accum_steps} 步）...")
            
            if step_in_epoch == accum_steps:
                print(f"[Rank {rank}] 执行 scaler.unscale_(optimizer)...")
            scaler.unscale_(optimizer)
            if step_in_epoch == accum_steps:
                print(f"[Rank {rank}] scaler.unscale_ 完成，检查梯度...")
            
            # 检查并清理梯度中的 NaN/Inf（总是执行，无论是否有NaN/Inf）
            found_bad_grad = False
            for p in model.parameters():
                if p.grad is None: continue
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    found_bad_grad = True
                    nan_inf_count += 1
                    # 清理NaN/Inf梯度，替换为0
                    p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

            if found_bad_grad:
                if rank == 0:
                    print(f"[Train][WARN] Found NaN/Inf in gradients, cleaned. Count={nan_inf_count}/{max_nan_inf_count}")

            # ★★★ 自适应梯度裁剪：如果检测到NaN/Inf，使用更激进的裁剪阈值
            # 如果当前batch有NaN/Inf问题，使用更小的裁剪阈值
            adaptive_grad_norm = max_grad_norm
            if found_bad_grad or loss_is_bad:
                # 当检测到NaN/Inf时，使用更激进的梯度裁剪（原阈值的50%）
                adaptive_grad_norm = max_grad_norm * 0.5
                if rank == 0 and step_in_epoch % 10 == 0:  # 每10步打印一次，避免刷屏
                    print(f"[Train][WARN] Using aggressive grad clipping: {adaptive_grad_norm:.4f} (normal: {max_grad_norm:.4f})")
            
            # ★★★ 总是执行梯度裁剪，无论是否有NaN/Inf（已清理）
            if step_in_epoch == accum_steps:
                print(f"[Rank {rank}] 执行 clip_grad_norm_...")
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), adaptive_grad_norm)
                if step_in_epoch == accum_steps:
                    print(f"[Rank {rank}] clip_grad_norm_ 完成，grad_norm={grad_norm.item():.6f}")
                
                # 如果梯度范数过大，记录警告
                if grad_norm > max_grad_norm * 2.0 and rank == 0 and step_in_epoch % 10 == 0:
                    print(f"[Train][WARN] Large gradient norm detected: {grad_norm.item():.4f} (threshold: {max_grad_norm:.4f})")
                    
            except Exception as e:
                if rank == 0:
                    print(f"[Train][ERROR] clip_grad_norm_ failed: {e}, zeroing grads")
                # 如果裁剪失败，清零梯度
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                nan_inf_count += 1
                continue
            
            # 在所有进程上打印，确保能看到所有进程的状态
            if step_in_epoch == accum_steps:
                print(f"[Rank {rank}] 准备执行 optimizer.step()...")
            
            # 确保所有进程都到达这里（在 clip_grad_norm_ 之后）
            if is_dist_avail_and_initialized():
                if step_in_epoch == accum_steps:
                    print(f"[Rank {rank}] 等待所有进程到达 barrier...")
                torch.distributed.barrier()
                if step_in_epoch == accum_steps:
                    print(f"[Rank {rank}] barrier 完成，所有进程已同步")
            
            if step_in_epoch == accum_steps:
                print(f"[Rank {rank}] 开始执行 scaler.step(optimizer)...")
            
            try:
                scaler.step(optimizer)
                if step_in_epoch == accum_steps:
                    print(f"[Rank {rank}] scaler.step(optimizer) 完成")
            except Exception as e:
                print(f"[Rank {rank}] ERROR in scaler.step(optimizer): {e}")
                if rank == 0:
                    send_email(text=f"训练失败: scaler.step(optimizer) 错误: {e}")
                raise
            
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if ema is not None:
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    ema.update(model.module)
                else:
                    ema.update(model)
            if scheduler is not None:
                scheduler.step()

            if rank == 0:
                pbar.update(1)
            
            # 检查NaN/Inf计数是否超过阈值（仅发送邮件，不停止训练）
            if nan_inf_count >= max_nan_inf_count:
                if rank == 0:
                    print(f"[Train][WARN] NaN/Inf count ({nan_inf_count}) exceeds threshold ({max_nan_inf_count}), continuing with aggressive gradient clipping")
                    # 发送邮件通知，但不停止训练
                    send_email(text=f"Epoch训练中NaN/Inf计数超过阈值: {nan_inf_count}/{max_nan_inf_count}，已启用更激进的梯度裁剪，训练继续")

        # 只累加有效的loss（非NaN/Inf）
        if not loss_is_bad:
            total_loss += loss.item()

    pbar.close()
    
    # 同步NaN/Inf计数到所有进程
    nan_inf_tensor = torch.tensor([nan_inf_count], device=device, dtype=torch.int32)
    if is_dist_avail_and_initialized():
        torch.distributed.all_reduce(nan_inf_tensor, op=torch.distributed.ReduceOp.SUM)
        nan_inf_count = int(nan_inf_tensor.item())
    
    if rank == 0:
        print(f"[Train] Epoch completed. NaN/Inf count: {nan_inf_count}/{max_nan_inf_count}")

    # 计算平均loss（只使用有效batch）
    valid_batches = step_in_epoch - nan_inf_count if nan_inf_count < step_in_epoch else 1
    avg = torch.tensor([total_loss / max(1, valid_batches)], device=device)
    if is_dist_avail_and_initialized():
        torch.distributed.all_reduce(avg, op=torch.distributed.ReduceOp.SUM)
        avg = avg / get_world_size()
    
    return avg.item(), nan_inf_count

@torch.no_grad()
def validate_epoch(model, val_loader, device, total_batches: int, rank: int,
                   alpha_time: float, lambda_noise: float, lambda_evm: float, use_ema: bool, ema: Optional[EMA] = None):
    need_ema = (use_ema and ema is not None)
    if need_ema:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            ema.apply_shadow(model.module)
        else:
            ema.apply_shadow(model)

    model.eval()
    total_loss = 0.0
    pbar = tqdm(total=total_batches, disable=(rank != 0), desc="Validating(EMA)")
    for batch in islice(val_loader, total_batches):
        batch_data = process_batch(batch, device)
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            output = model(batch_data['mixsignal'])
            loss = calculate_loss(output, batch_data, alpha_time=alpha_time, lambda_noise=lambda_noise, lambda_evm=lambda_evm)
        # 防护：若单个 batch 的 loss 非法（理论上 calculate_loss 已处理），记录并替换为大数
        if torch.isnan(loss) or torch.isinf(loss):
            if rank == 0:
                print(f"[Val][WARN] NaN/Inf loss encountered in validation batch; replacing with 1e6")
            loss = torch.tensor(1e6, device=device, dtype=loss.dtype)
        total_loss += loss.item()
        if rank == 0:
            pbar.update(1)
    pbar.close()

    # 防护：若 total_loss 非法（例如累加过程中出现 NaN/Inf），则直接设为大数，避免 all_reduce 传播 NaN
    if math.isnan(total_loss) or math.isinf(total_loss):
        if rank == 0:
            print(f"[Val][WARN] total_loss is NaN/Inf ({total_loss}); setting avg to 1e6")
        avg = torch.tensor([1e6], device=device)
    else:
        avg = torch.tensor([total_loss / max(1, total_batches)], device=device)
    if is_dist_avail_and_initialized():
        torch.distributed.all_reduce(avg, op=torch.distributed.ReduceOp.SUM)
        avg = avg / get_world_size()

    if need_ema:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            ema.restore(model.module)
        else:
            ema.restore(model)

    return avg.item()

def plot_and_save_loss(train_losses, val_losses, save_path):
    import numpy as _np
    plt.figure(figsize=(10, 5))
    # 转换为 numpy 并清洗非法值
    tr = _np.array(train_losses, dtype=float)
    vl = _np.array(val_losses, dtype=float)
    tr = _np.nan_to_num(tr, nan=1e6, posinf=1e6, neginf=1e6)
    vl = _np.nan_to_num(vl, nan=1e6, posinf=1e6, neginf=1e6)
    # log scale 下不能有 0 或负值，裁剪到最小正值
    eps = 1e-12
    tr = _np.clip(tr, eps, None)
    vl = _np.clip(vl, eps, None)

    epochs_tr = list(range(1, len(tr) + 1))
    epochs_vl = list(range(1, len(vl) + 1))
    plt.plot(epochs_tr, tr, marker='o', linewidth=1.2, label='Training Loss')
    plt.plot(epochs_vl, vl, marker='o', linewidth=1.2, label='Validation Loss (EMA)')
    plt.xlabel('Epoch'); plt.ylabel('Loss (log scale)'); plt.yscale('log')
    plt.grid(True, which="both", ls="--", linewidth=0.5); plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path); plt.close()

# =========================
# 主程序
# =========================
def main():
    # 避免 fork+CUDA 的潜在死锁（可选）
    try:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Train Signal Separator (A+C)")
    parser.add_argument('--gpus', type=str, default='')
    parser.add_argument('--dataset_path', type=str, default='/nas/datasets/yixin/PCMA/sim_data')
    parser.add_argument('--N', type=int, default=100000)
    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=64)   # 每卡
    parser.add_argument('--num_workers', type=int, default=0)   # 先 0 跑通
    parser.add_argument('--epochs', type=int, default=80)
    # 优化/调度
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--min_lr_ratio', type=float, default=0.1)
    parser.add_argument('--accum_steps', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--use_ema', action='store_true', default=False)
    # ===== 损失权重 =====
    parser.add_argument('--alpha_time', type=float, default=1.0,
                        help='波形损失（Lwave）的权重')
    parser.add_argument('--lambda_noise', type=float, default=0.1,
                        help='噪声一致性损失（Lnoise）的权重（在阶段2和3中使用）')
    parser.add_argument('--lambda_evm', type=float, default=0.5,
                        help='EVM损失（LEVM）的权重（在阶段3中使用）')
    # ===== 分阶段训练配置 =====
    parser.add_argument('--mse_epochs', type=int, default=100,
                        help='阶段1：纯MSE训练的epoch数（0 <= epoch < mse_epochs）')
    parser.add_argument('--noise_epochs', type=int, default=20,
                        help='阶段2：加入噪声损失的epoch数（mse_epochs <= epoch < mse_epochs + noise_epochs）')
    # 阶段3的epoch数 = total_epochs - mse_epochs - noise_epochs
    parser.add_argument('--mode_name', type=str, default=None,
                        help='数据集前缀名称（mode_name），如果未指定则使用默认值。用于匹配数据集文件。')
    parser.add_argument('--init_ckpt', type=str, default=None,
                        help='可选：从已训练的 checkpoint 加载权重（例如纯 MSE 阶段的模型）继续训练')
    parser.add_argument('--max_nan_inf_count', type=int, default=100,
                        help='每个epoch允许的最大NaN/Inf次数，超过此阈值将发送邮件通知并启用更激进的梯度裁剪（不停止训练）')

    args = parser.parse_args()
    # DDP 初始化 & 设备
    setup_distributed(backend="nccl", timeout_seconds=7200)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # 如果未指定mode_name，使用默认值
    if args.mode_name is None:
        mode_name = f'mixedmods_train_robust_rand_freqU[0,200]_phi1U[0.0000,6.2832]_phi2U[0.0000,6.2832]_ampU[0.20,0.90]_snrU[8,22]_N{args.N}_varsnr_ampr_phi1phi2_delay0T_c64'
    else:
        mode_name = args.mode_name
    # mode_name = f'mixedmods_train_aligned_rand_freqU[30,130]_phi1U[0.0000,6.2832]_phi2U[0.0000,6.2832]_ampU[0.40,0.90]_snrU[12,18]_N100000_varsnr_ampr_phi1phi2_delay0T_c64'
    
    # 所有输出保存到 /nas/datasets/yixin/PCMA/src
    base_output_dir = '/nas/datasets/yixin/PCMA/src'
    save_pic = os.path.join(base_output_dir, 'loss','improved', f'loss_SigSep_{mode_name}.png')
    save_dir = os.path.join(base_output_dir, 'check_points', 'all','improved')

    rank = get_rank(); world = get_world_size()
    
    # 确保保存目录存在，并检查权限和空间
    # 只在 rank 0 创建目录，其他 rank 等待
    if rank == 0:
        try:
            # 递归创建所有需要的目录
            os.makedirs(save_dir, exist_ok=True, mode=0o755)
            os.makedirs(os.path.dirname(save_pic), exist_ok=True, mode=0o755)  # 确保 loss 目录存在
            
            # 验证目录确实存在
            if not os.path.exists(save_dir):
                error_msg = f"目录创建失败: {save_dir}"
                send_email(text=f"训练失败: {error_msg}")
                raise RuntimeError(error_msg)
            if not os.path.isdir(save_dir):
                error_msg = f"路径不是目录: {save_dir}"
                send_email(text=f"训练失败: {error_msg}")
                raise RuntimeError(error_msg)
            
            # 检查目录是否可写
            test_file = os.path.join(save_dir, '.write_test')
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
            except Exception as e:
                error_msg = f"保存目录不可写: {save_dir}, 错误: {e}"
                send_email(text=f"训练失败: {error_msg}")
                raise RuntimeError(error_msg)
            
            # 检查磁盘空间（至少需要 1GB 可用空间）
            import shutil
            stat = shutil.disk_usage(save_dir)
            free_gb = stat.free / (1024**3)
            if free_gb < 1.0:
                error_msg = f"磁盘空间不足: {save_dir} 仅剩 {free_gb:.2f} GB，至少需要 1 GB"
                send_email(text=f"训练失败: {error_msg}")
                raise RuntimeError(error_msg)
            
            print(f"[Disk] 可用空间: {free_gb:.2f} GB")
            print(f"[Path] 保存目录: {save_dir}")
            print(f"[Path] 损失图片: {save_pic}")
        except Exception as e:
            print(f"[ERROR] 保存目录检查失败: {e}")
            send_email(text=f"训练失败: 保存目录检查失败: {e}")
            raise
    
    # 所有 rank 等待 rank 0 完成目录创建
    if is_dist_avail_and_initialized():
        torch.distributed.barrier()
    
    # 所有 rank 验证目录存在（防止竞争条件）
    if not os.path.exists(save_dir):
        error_msg = f"目录不存在: {save_dir} (可能由其他进程创建失败)"
        if rank == 0:
            send_email(text=f"训练失败: {error_msg}")
        raise RuntimeError(error_msg)
    
    final_ckpt = os.path.join(save_dir, f'signal_separator_{mode_name}.pth')
    best_ckpt  = os.path.join(save_dir, f'signal_separator_{mode_name}_best.pth')
    if rank == 0:
        print(f"[Train] world_size={world}, device={device}")

    # DataLoaders（返回各 rank 本地批次数）
    train_loader, val_loader, train_batches_local, val_batches_local = prepare_dataloaders(
        args.dataset_path, mode=mode_name, batch_size=args.batch_size,
        train_ratio=args.train_ratio, num_workers=args.num_workers,
        seed=2025, dist_rank=rank, dist_world_size=world
    )

    # ★★★ 关键修复：把批次数统一为“全局最小”
    tb = torch.tensor([train_batches_local], device=device, dtype=torch.int32)
    vb = torch.tensor([val_batches_local],   device=device, dtype=torch.int32)
    if is_dist_avail_and_initialized():
        torch.distributed.all_reduce(tb, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(vb, op=torch.distributed.ReduceOp.MIN)
    train_batches = int(tb.item())
    val_batches   = int(vb.item())
    if rank == 0:
        print(f"[Batches] local(train)={train_batches_local}, global_min(train)={train_batches}")
        print(f"[Batches] local(val)  ={val_batches_local}, global_min(val)  ={val_batches}")

    # 模型/优化器
    model, optimizer = initialize_model(device, lr=args.lr)

    # 如果提供了 init_ckpt，则加载预训练权重（如纯 MSE 阶段的模型）
    if args.init_ckpt is not None and os.path.exists(args.init_ckpt):
        try:
            state = torch.load(args.init_ckpt, map_location=device)
            # 兼容 DDP/非DDP，两种常见存储方式
            missing, unexpected = model.load_state_dict(state, strict=False)
            if rank == 0:
                print(f"[Init] Loaded init_ckpt: {args.init_ckpt}")
                if missing:
                    print(f"[Init][Warn] Missing keys: {missing}")
                if unexpected:
                    print(f"[Init][Warn] Unexpected keys: {unexpected}")
        except Exception as e:
            if rank == 0:
                print(f"[Init][ERROR] Failed to load init_ckpt {args.init_ckpt}: {e}")
                send_email(text=f"训练初始化失败: 加载 init_ckpt 出错: {e}")
            raise
    elif args.init_ckpt is not None:
        if rank == 0:
            print(f"[Init][WARN] init_ckpt 不存在: {args.init_ckpt}")

    if is_dist_avail_and_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == 'cuda' else None,
            output_device=local_rank if device.type == 'cuda' else None,
            find_unused_parameters=False,
            broadcast_buffers=False,   # ★ 推荐关闭，减少额外广播差异
        )

    # 计算 step 级总步数 & warmup
    steps_per_epoch = math.ceil(train_batches / max(1, args.accum_steps))  # 每卡
    total_steps = args.epochs * max(1, steps_per_epoch)
    warmup_steps_eff = args.warmup_steps
    if args.warmup_epochs > 0:
        warmup_steps_eff = args.warmup_epochs * max(1, steps_per_epoch)
        if rank == 0:
            print(f"[LR] use warmup_epochs={args.warmup_epochs} => warmup_steps={warmup_steps_eff} (per-rank)")

    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps_eff,
                                     total_steps=total_steps, min_lr_ratio=args.min_lr_ratio)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    ema = EMA(model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model,
              decay=args.ema_decay) if args.use_ema else None

    train_losses, val_losses = [], []
    best_val = float('inf')
    # track saved file paths so we keep only two files on disk
    latest_saved_path = None
    best_saved_path = None

    for epoch in range(args.epochs):
        alpha_t, lambda_n, lambda_e = get_loss_weights(epoch, args)
        
        # 确定当前阶段
        if epoch < args.mse_epochs:
            stage = "阶段1: MSE"
        elif epoch < args.mse_epochs + args.noise_epochs:
            stage = "阶段2: MSE+Noise"
        else:
            stage = "阶段3: MSE+Noise+EVM"
        
        if rank == 0:
            print(f"\n=== Epoch {epoch+1}/{args.epochs} [{stage}] | α_time={alpha_t:.3f} | λ_noise={lambda_n:.3f} | λ_evm={lambda_e:.3f} ===")

        tr, nan_inf_count = train_epoch(model, train_loader, optimizer, scheduler, scaler, ema, device,
                        total_batches=train_batches, rank=rank, accum_steps=args.accum_steps,
                        alpha_time=alpha_t, lambda_noise=lambda_n, lambda_evm=lambda_e, max_grad_norm=args.max_grad_norm,
                        max_nan_inf_count=args.max_nan_inf_count)
        
        # 检查NaN/Inf计数（仅发送邮件通知，不停止训练）
        if nan_inf_count >= args.max_nan_inf_count:
            if rank == 0:
                print(f"--------------------------------")
                print(f"[WARN] Epoch {epoch+1} NaN/Inf count: {nan_inf_count}/{args.max_nan_inf_count}")
                print(f"[WARN] 训练将继续，已启用更激进的梯度裁剪策略")
                print(f"--------------------------------")
                send_email(text=f"Epoch {epoch+1} NaN/Inf计数超过阈值: {nan_inf_count}/{args.max_nan_inf_count}，训练继续")
            # 不break，继续训练

        vl = validate_epoch(model, val_loader, device, total_batches=val_batches, rank=rank,
                    alpha_time=alpha_t, lambda_noise=lambda_n, lambda_evm=lambda_e,
                    use_ema=args.use_ema, ema=ema)


        if rank == 0:
            train_losses.append(tr); val_losses.append(vl)
            print(f"Train Loss: {tr:.6f} | Val Loss(EMA): {vl:.6f} | NaN/Inf Count: {nan_inf_count}/{args.max_nan_inf_count}")
            plot_and_save_loss(train_losses, val_losses, save_pic)

            # 保存 latest：只保留一个 latest 文件（覆盖之前的），文件名包含 epoch
            model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            latest_fname = os.path.join(save_dir, f'signal_separator_{mode_name}_latest_epoch{epoch+1}.pth')
            # 删除之前的 latest 文件（如果存在），确保目录中只有一个 latest
            if latest_saved_path is not None and os.path.exists(latest_saved_path):
                try:
                    os.remove(latest_saved_path)
                except Exception:
                    pass
            torch.save(model_to_save.state_dict(), latest_fname)
            latest_saved_path = latest_fname

            # 若验证更好，则保存 best：只保留一个 best 文件，文件名包含 epoch
            if vl < best_val - 1e-8:
                best_val = vl
                # 若启用 EMA，使用 EMA 权重保存（以保持原先行为）；否则直接保存当前模型权重
                if ema is not None:
                    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                        ema.apply_shadow(model.module)
                        state = model.module.state_dict()
                        ema.restore(model.module)
                    else:
                        ema.apply_shadow(model)
                        state = model.state_dict()
                        ema.restore(model)
                else:
                    state = model_to_save.state_dict()

                best_fname = os.path.join(save_dir, f'signal_separator_{mode_name}_best_epoch{epoch+1}.pth')
                # 删除之前的 best 文件（如果存在），确保目录中只有一个 best
                if best_saved_path is not None and os.path.exists(best_saved_path):
                    try:
                        os.remove(best_saved_path)
                    except Exception:
                        pass
                torch.save(state, best_fname)
                best_saved_path = best_fname
                print(f"[Best] updated: {best_saved_path} (val={best_val:.6f})")

    # 保存最终（EMA）
    if rank == 0:
        if ema is not None:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                ema.apply_shadow(model.module); torch.save(model.module.state_dict(), final_ckpt); ema.restore(model.module)
            else:
                ema.apply_shadow(model); torch.save(model.state_dict(), final_ckpt); ema.restore(model)
        else:
            torch.save((model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model).state_dict(), final_ckpt)
        print(f"Training completed. Final: {final_ckpt}\nBest: {best_ckpt} (val={best_val:.6f})")

    cleanup_distributed()

if __name__ == '__main__':
    try:
        main()
        send_email(text=f"Training completed")
    except Exception as e:
        send_email(text=f"训练异常退出: {str(e)}")
        raise