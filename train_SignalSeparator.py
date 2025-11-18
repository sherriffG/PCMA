# -*- coding: utf-8 -*-
import os, re, math, random, argparse
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict, Any, Iterator, Optional, Tuple
from torch.utils.data import IterableDataset, DataLoader
from itertools import islice
from datetime import timedelta

from model_complex import SignalSeparator  # 你的模型

# ===== 可选调试环境变量（也可在外部shell设置） =====
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "INFO")
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")

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

    return {
        'mixsignal_real': mix_r, 'mixsignal_imag': mix_i,
        'rfsignal1_real': r1_r,  'rfsignal1_imag': r1_i,
        'rfsignal2_real': r2_r,  'rfsignal2_imag': r2_i,
    }

def _collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, List[torch.Tensor]] = {}
    for sample in batch:
        for k, v in sample.items():
            out.setdefault(k, []).append(v)
    return {k: torch.stack(vlist, dim=0) for k, vlist in out.items()}

def _get_shard_sizes(shard_files: List[str]) -> List[int]:
    sizes = []
    for p in shard_files:
        entries = torch.load(p)
        sizes.append(len(entries))
        del entries
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
        self.plan = self.plan_all[self.dist_rank::self.dist_world_size]

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
    shard_files = _find_shard_files(dataset_path, mode)
    assert len(shard_files) > 0, f"未找到分片文件，检查目录与前缀是否正确：{dataset_path} / {mode}"

    train_plan, val_plan, train_samples, val_samples = _build_sample_level_plan(
        shard_files, train_ratio=train_ratio, seed=seed
    )

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

    if get_rank() == 0:
        print(f"[Data] shards={len(shard_files)} | train≈{train_samples} | val≈{val_samples} | ratio={train_ratio:.2f}")
        print(f"[Data] world_size={dist_world_size}")
        print(f"[Data/rank0] samples_rank(train)={train_samples_rank}, batches_rank(train)={train_batches_rank}")
        print(f"[Data/rank0] samples_rank(val)  ={val_samples_rank}, batches_rank(val)  ={val_batches_rank}")

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
def complex_si_mse(pred_2ch, tgt_2ch, eps: float = 1e-12, a_mag_clip: float = 10.0):
    y = pred_2ch[:, 0, :] + 1j * pred_2ch[:, 1, :]
    x = tgt_2ch[:, 0, :] + 1j * tgt_2ch[:, 1, :]

    num = torch.sum(torch.conj(y) * x, dim=1)
    den = torch.sum(torch.conj(y) * y, dim=1) + eps
    a = num / den
    if a_mag_clip is not None and a_mag_clip > 0:
        mag = torch.abs(a)
        a = a * torch.clamp(a_mag_clip / (mag + 1e-12), max=1.0)

    aligned = a.unsqueeze(-1) * y
    err = aligned - x
    mse = torch.mean(torch.view_as_real(err)**2, dim=[1, 2])
    return mse

def normalized_time_mse(pred_2ch: torch.Tensor, tgt_2ch: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    diff2 = (pred_2ch - tgt_2ch) ** 2
    num = diff2.mean(dim=[1, 2])
    den = torch.norm(tgt_2ch, dim=[1, 2]) + eps
    return num / den

def calculate_loss(output, batch_data, alpha_time=1.0, beta_csi=0.5):
    pred1 = torch.cat([output[0], output[1]], dim=1)  # (B,2,T)
    pred2 = torch.cat([output[2], output[3]], dim=1)  # (B,2,T)
    tgt1  = batch_data['rfsignal1']                   # (B,2,T)
    tgt2  = batch_data['rfsignal2']                   # (B,2,T)

    nmse1 = normalized_time_mse(pred1, tgt1)
    nmse2 = normalized_time_mse(pred2, tgt2)
    nmse  = 0.5 * (nmse1 + nmse2)

    csi1 = complex_si_mse(pred1, tgt1, a_mag_clip=10.0)
    csi2 = complex_si_mse(pred2, tgt2, a_mag_clip=10.0)
    csi  = 0.5 * (csi1 + csi2)

    loss = alpha_time * nmse.mean() + beta_csi * csi.mean()
    return loss

def get_loss_weights(epoch_idx: int, args) -> Tuple[float, float]:
    """
    返回 (alpha_time, beta_csi)
    - 0 <= epoch < mse_epochs: 纯 MSE => (1.0, 0.0)
    - epoch >= mse_epochs:
        * 若 csi_warmup_epochs == 0: 立即 (alpha_after, csi_beta)
        * 若 csi_warmup_epochs  > 0:
            t = clip((epoch - mse_epochs) / csi_warmup_epochs, 0, 1)
            beta = t * args.csi_beta
            alpha:
               - 若 args.fade_mse: alpha = (1 - t) * 1.0 + t * args.alpha_after
               - 否则:            alpha = (1.0 if t < 1.0 else args.alpha_after)
    """
    # 阶段 1：纯 MSE
    if epoch_idx < args.mse_epochs:
        return 1.0, 0.0

    # 阶段 2：CSI
    if args.csi_warmup_epochs <= 0:
        return float(args.alpha_after), float(args.csi_beta)

    # 线性 warmup from 0 -> csi_beta
    k = epoch_idx - args.mse_epochs
    t = min(1.0, max(0.0, k / max(1, args.csi_warmup_epochs)))

    beta = float(t * args.csi_beta)
    if args.fade_mse:
        alpha = (1.0 - t) * 1.0 + t * float(args.alpha_after)
    else:
        alpha = 1.0 if t < 1.0 else float(args.alpha_after)

    return float(alpha), float(beta)



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
    return {
        'mixsignal': torch.cat([mixsignal_real, mixsignal_imag], dim=1),
        'rfsignal1': torch.cat([rfsignal1_real, rfsignal1_imag], dim=1),
        'rfsignal2': torch.cat([rfsignal2_real, rfsignal2_imag], dim=1),
    }

def train_epoch(model, train_loader, optimizer, scheduler, scaler, ema, device,
                total_batches: int, rank: int, accum_steps: int,
                alpha_time: float, beta_csi: float, max_grad_norm: float):
    model.train()
    total_loss = 0.0
    step_in_epoch = 0
    pbar = tqdm(total=total_batches, disable=(rank != 0), desc=f"Training[βcsi={beta_csi:.3f}]")

    optimizer.zero_grad(set_to_none=True)
    for batch in islice(train_loader, total_batches):
        batch_data = process_batch(batch, device)
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            output = model(batch_data['mixsignal'])
            loss = calculate_loss(output, batch_data, alpha_time=alpha_time, beta_csi=beta_csi)
            loss = loss / max(1, accum_steps)

        scaler.scale(loss).backward()
        step_in_epoch += 1

        if step_in_epoch % max(1, accum_steps) == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
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

        total_loss += loss.item()

    pbar.close()

    avg = torch.tensor([total_loss / max(1, step_in_epoch)], device=device)
    if is_dist_avail_and_initialized():
        torch.distributed.all_reduce(avg, op=torch.distributed.ReduceOp.SUM)
        avg = avg / get_world_size()
    return avg.item()

@torch.no_grad()
def validate_epoch(model, val_loader, device, total_batches: int, rank: int,
                   alpha_time: float, beta_csi: float, use_ema: bool, ema: Optional[EMA] = None):
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
            loss = calculate_loss(output, batch_data, alpha_time=alpha_time, beta_csi=beta_csi)
        total_loss += loss.item()
        if rank == 0:
            pbar.update(1)
    pbar.close()

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
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss (EMA)')
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
    parser.add_argument('--use_ema', action='store_true', default=True)
    # ===== 损失阶段式调度 =====
    parser.add_argument('--mse_epochs', type=int, default=40,
                        help='前 N 个 epoch 仅优化 MSE（alpha=1, beta=0）')
    parser.add_argument('--csi_warmup_epochs', type=int, default=5,
                        help='从第 mse_epochs 起，CSI 权重从 0 线性升到 csi_beta，MSE 同步线性衰减到 alpha_after；=0 表示硬切换')
    parser.add_argument('--csi_beta', type=float, default=1.0,
                        help='切换到 CSI 阶段后的 CSI 系数目标值')
    parser.add_argument('--alpha_after', type=float, default=0.0,
                        help='进入 CSI 阶段后，MSE 的残留权重（通常设为 0.0 即纯 CSI），如需少量正则可设 0.1 等')
    parser.add_argument('--fade_mse', action='store_true', default=True,
                        help='启用 MSE->CSI 的线性衰减（与 csi_warmup_epochs 配合）。如果关闭则在 warmup 期间保持 alpha=1.0，之后骤降到 alpha_after')



    args = parser.parse_args()

    # DDP 初始化 & 设备
    setup_distributed(backend="nccl", timeout_seconds=7200)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    mode_name = f'qpsk_train_rand_freqU[0,200]_phi1U[0.0000,6.2832]_phi2U[0.0000,6.2832]_ampU[0.30,0.90]_snrU[12,30]_N{args.N}_varsnr_ampr_phi1phi2_delay0T_c64'
    save_pic = f'./src/pics/all/loss_SigSep_{mode_name}.png'
    save_dir = './src/check_points/all'; os.makedirs(save_dir, exist_ok=True)
    final_ckpt = os.path.join(save_dir, f'signal_separator_{mode_name}.pth')
    best_ckpt  = os.path.join(save_dir, f'signal_separator_{mode_name}_best.pth')

    rank = get_rank(); world = get_world_size()
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

    for epoch in range(args.epochs):
        alpha_t, beta_c = get_loss_weights(epoch, args)
        if rank == 0:
            print(f"\n=== Epoch {epoch+1}/{args.epochs} | α_time={alpha_t:.3f} | β_CSI={beta_c:.3f} ===")

        tr = train_epoch(model, train_loader, optimizer, scheduler, scaler, ema, device,
                        total_batches=train_batches, rank=rank, accum_steps=args.accum_steps,
                        alpha_time=alpha_t, beta_csi=beta_c, max_grad_norm=args.max_grad_norm)

        vl = validate_epoch(model, val_loader, device, total_batches=val_batches, rank=rank,
                            alpha_time=alpha_t, beta_csi=beta_c,
                            use_ema=True, ema=ema)


        if rank == 0:
            train_losses.append(tr); val_losses.append(vl)
            print(f"Train Loss: {tr:.6f} | Val Loss(EMA): {vl:.6f}")
            plot_and_save_loss(train_losses, val_losses, save_pic)

            if vl < best_val - 1e-8:
                best_val = vl
                if ema is not None:
                    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                        ema.apply_shadow(model.module); torch.save(model.module.state_dict(), best_ckpt); ema.restore(model.module)
                    else:
                        ema.apply_shadow(model); torch.save(model.state_dict(), best_ckpt); ema.restore(model)
                else:
                    torch.save((model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model).state_dict(), best_ckpt)
                print(f"[Best] updated: {best_ckpt} (val={best_val:.6f})")

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
    main()
