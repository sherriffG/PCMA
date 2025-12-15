# -*- coding: utf-8 -*-
# ============================================================
# ResNet1D SignalSeparator - Training Entry
#
# Recommended launch (DDP, 8 GPUs) from repo root:
#
#   torchrun --nproc_per_node=8 -m resnet.train_SignalSeparator \
#     --dataset_path /nas/datasets/yixin/PCMA/sim_data \
#     --mode_name "<YOUR_MODE_NAME>" \
#     --N 100000 \
#     --epochs 80 \
#     --batch_size 64 \
#     --lr 2e-4 \
#     --warmup_epochs 0 \
#     --min_lr_ratio 0.1 \
#     --accum_steps 1 \
#     --max_grad_norm 1.0
#
# Notes:
# - This script writes checkpoints with prefix `signal_separator_resnet_...`
#   to avoid overwriting the original (non-resnet) models.
# ============================================================
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

from .model_resnet import SignalSeparator


# ===== 可选调试环境变量（也可在外部shell设置） =====
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "INFO")
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "INFO")
os.environ.setdefault("NCCL_DEBUG", "INFO")
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
            if m1:
                return int(m1.group(1))
            m2 = re.search(r"_shard(\d+)-of-\d+\.pth$", name)
            if m2:
                return int(m2.group(1))
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
    r1_r, r1_i = to_float(e['rfsignal1'])
    r2_r, r2_i = to_float(e['rfsignal2'])

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
    total_shards = len(shard_files)
    rank = get_rank()

    if rank == 0:
        print(f"[Data] 正在加载 {total_shards} 个分片文件以获取样本数...")

    for idx, p in enumerate(shard_files):
        if rank == 0 and (idx + 1) % max(1, total_shards // 10) == 0:
            print(f"[Data] 已处理 {idx + 1}/{total_shards} 个分片...")
        entries = torch.load(p, map_location='cpu')
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
        val_idx = idx[n_train:]
        if train_idx:
            train_plan.append({'path': path, 'indices': train_idx})
        if val_idx:
            val_plan.append({'path': path, 'indices': val_idx})

    train_samples = sum(len(x['indices']) for x in train_plan)
    val_samples = sum(len(x['indices']) for x in val_plan)
    return train_plan, val_plan, train_samples, val_samples


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

        # 按样本分配
        all_samples = []
        for item in self.plan_all:
            for idx in item['indices']:
                all_samples.append((item['path'], idx))
        self.samples_for_rank = all_samples[self.dist_rank::self.dist_world_size]

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
    num_workers: int = 0,
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
    assert len(shard_files) > 0, f"未找到分片文件，检查目录与前缀是否正确：{dataset_path} / {mode}"

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

    train_samples_rank = sum(len(item['indices']) for item in train_dataset.plan)
    val_samples_rank = sum(len(item['indices']) for item in val_dataset.plan)
    train_batches_rank = max(1, train_samples_rank // batch_size)                   # drop_last=True
    val_batches_rank = max(1, (val_samples_rank + batch_size - 1) // batch_size)    # drop_last=False

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

    print(f"[Rank {rank}] [Data] shards={len(shard_files)} | train_total≈{train_samples} | val_total≈{val_samples} | ratio={train_ratio:.2f}")
    print(f"[Rank {rank}] [Data] samples_rank(train)={train_samples_rank}, batches_rank(train)={train_batches_rank}")
    print(f"[Rank {rank}] [Data] samples_rank(val)  ={val_samples_rank}, batches_rank(val)  ={val_batches_rank}")
    print(f"[Rank {rank}] [Data] plan_items(train)={len(train_dataset.plan)}, plan_items(val)={len(val_dataset.plan)}")

    if train_samples_rank == 0:
        raise ValueError(f"[Rank {rank}] 错误：训练集样本数为 0！请检查数据分配逻辑。")
    if train_batches_rank < 10 and rank == 0:
        print(f"[WARN] Rank {rank} 的批次数很少 ({train_batches_rank})，可能导致训练不稳定。")

    return train_loader, val_loader, train_batches_rank, val_batches_rank


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
# 损失（与原版保持一致）
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
    mse = torch.mean(torch.view_as_real(err) ** 2, dim=[1, 2])
    return mse


def normalized_time_mse(pred_2ch: torch.Tensor, tgt_2ch: torch.Tensor, eps: float = 1e-12, max_ratio: float = 1000.0) -> torch.Tensor:
    diff2 = (pred_2ch - tgt_2ch) ** 2
    num = diff2.mean(dim=[1, 2])
    den = torch.mean(tgt_2ch ** 2, dim=[1, 2]) + eps
    ratio = num / den
    ratio = torch.clamp(ratio, min=0.0, max=max_ratio)
    return torch.log1p(ratio)


def calculate_loss(output, batch_data, alpha_time=1.0, beta_csi=0.5):
    pred1 = torch.cat([output[0], output[1]], dim=1)
    pred2 = torch.cat([output[2], output[3]], dim=1)
    tgt1 = batch_data['rfsignal1']
    tgt2 = batch_data['rfsignal2']

    nmse1 = normalized_time_mse(pred1, tgt1)
    nmse2 = normalized_time_mse(pred2, tgt2)
    nmse = 0.5 * (nmse1 + nmse2)

    csi1 = complex_si_mse(pred1, tgt1, a_mag_clip=10.0)
    csi2 = complex_si_mse(pred2, tgt2, a_mag_clip=10.0)
    csi = 0.5 * (csi1 + csi2)

    nmse = torch.nan_to_num(nmse, nan=1e6, posinf=1e6, neginf=1e6)
    csi = torch.nan_to_num(csi, nan=1e6, posinf=1e6, neginf=1e6)

    loss = alpha_time * nmse.mean() + beta_csi * csi.mean()
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(1e6, device=loss.device, dtype=loss.dtype)
    return loss


def safe_save_model(state_dict, filepath, max_retries=3):
    import tempfile
    dirname = os.path.dirname(filepath)
    basename = os.path.basename(filepath)

    for attempt in range(max_retries):
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, dir=dirname, prefix=basename + ".tmp.") as f:
                temp_file = f.name
            torch.save(state_dict, temp_file)
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception:
                    pass
            os.rename(temp_file, filepath)
            return True
        except Exception as e:
            if temp_file is not None and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass
            if attempt == max_retries - 1:
                raise RuntimeError(f"保存模型失败 (已重试 {max_retries} 次): {filepath}, 错误: {e}")
    return False


def get_loss_weights(epoch_idx: int, args) -> Tuple[float, float]:
    if epoch_idx < args.mse_epochs:
        return 1.0, 0.0
    if args.csi_warmup_epochs <= 0:
        return float(args.alpha_after), float(args.csi_beta)

    k = epoch_idx - args.mse_epochs
    t = min(1.0, max(0.0, k / max(1, args.csi_warmup_epochs)))

    beta = float(t * args.csi_beta)
    if args.fade_mse:
        alpha = (1.0 - t) * 1.0 + t * float(args.alpha_after)
    else:
        alpha = 1.0 if t < 1.0 else float(args.alpha_after)
    return float(alpha), float(beta)


def initialize_model(device, lr=2e-4):
    model = SignalSeparator().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
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
                alpha_time: float, beta_csi: float, max_grad_norm: float,
                max_nan_inf_count: int = 100):
    model.train()
    total_loss = 0.0
    step_in_epoch = 0
    nan_inf_count = 0

    print(f"[Rank {rank}] 开始训练 epoch，total_batches={total_batches}")
    pbar = tqdm(total=total_batches, disable=(rank != 0), desc=f"Training[βcsi={beta_csi:.3f}]")

    optimizer.zero_grad(set_to_none=True)
    if is_dist_avail_and_initialized():
        torch.distributed.barrier()

    batch_count = 0
    for batch in islice(train_loader, total_batches):
        batch_data = process_batch(batch, device)
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            output = model(batch_data['mixsignal'])
            loss = calculate_loss(output, batch_data, alpha_time=alpha_time, beta_csi=beta_csi)
            loss = loss / max(1, accum_steps)

        loss_is_bad = torch.isnan(loss) or torch.isinf(loss)
        if loss_is_bad:
            nan_inf_count += 1
            if rank == 0:
                print(f"[Train][WARN] NaN/Inf loss encountered at step {step_in_epoch}, count={nan_inf_count}/{max_nan_inf_count}")

        scaler.scale(loss).backward()
        step_in_epoch += 1
        batch_count += 1

        if step_in_epoch % max(1, accum_steps) == 0:
            scaler.unscale_(optimizer)

            found_bad_grad = False
            for p in model.parameters():
                if p.grad is None:
                    continue
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    found_bad_grad = True
                    nan_inf_count += 1
                    p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
            if found_bad_grad and rank == 0:
                print(f"[Train][WARN] Found NaN/Inf in gradients, cleaned. Count={nan_inf_count}/{max_nan_inf_count}")

            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            except Exception as e:
                if rank == 0:
                    print(f"[Train][ERROR] clip_grad_norm_ failed: {e}, zeroing grads")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                nan_inf_count += 1
                continue

            if is_dist_avail_and_initialized():
                torch.distributed.barrier()

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

            if nan_inf_count >= max_nan_inf_count:
                if rank == 0:
                    print(f"[Train][ERROR] NaN/Inf count ({nan_inf_count}) exceeds threshold ({max_nan_inf_count}), stopping training")
                if is_dist_avail_and_initialized():
                    torch.distributed.barrier()
                raise RuntimeError(f"NaN/Inf count ({nan_inf_count}) exceeds threshold ({max_nan_inf_count})")

        if not loss_is_bad:
            total_loss += loss.item()

    pbar.close()

    nan_inf_tensor = torch.tensor([nan_inf_count], device=device, dtype=torch.int32)
    if is_dist_avail_and_initialized():
        torch.distributed.all_reduce(nan_inf_tensor, op=torch.distributed.ReduceOp.SUM)
        nan_inf_count = int(nan_inf_tensor.item())

    valid_batches = step_in_epoch - nan_inf_count if nan_inf_count < step_in_epoch else 1
    avg = torch.tensor([total_loss / max(1, valid_batches)], device=device)
    if is_dist_avail_and_initialized():
        torch.distributed.all_reduce(avg, op=torch.distributed.ReduceOp.SUM)
        avg = avg / get_world_size()
    return avg.item(), nan_inf_count


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
        if torch.isnan(loss) or torch.isinf(loss):
            if rank == 0:
                print(f"[Val][WARN] NaN/Inf loss encountered in validation batch; replacing with 1e6")
            loss = torch.tensor(1e6, device=device, dtype=loss.dtype)
        total_loss += loss.item()
        if rank == 0:
            pbar.update(1)
    pbar.close()

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
    tr = _np.array(train_losses, dtype=float)
    vl = _np.array(val_losses, dtype=float)
    tr = _np.nan_to_num(tr, nan=1e6, posinf=1e6, neginf=1e6)
    vl = _np.nan_to_num(vl, nan=1e6, posinf=1e6, neginf=1e6)
    eps = 1e-12
    tr = _np.clip(tr, eps, None)
    vl = _np.clip(vl, eps, None)

    epochs_tr = list(range(1, len(tr) + 1))
    epochs_vl = list(range(1, len(vl) + 1))
    plt.plot(epochs_tr, tr, marker='o', linewidth=1.2, label='Training Loss')
    plt.plot(epochs_vl, vl, marker='o', linewidth=1.2, label='Validation Loss (EMA)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def main():
    try:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Train Signal Separator (ResNet1D)")
    parser.add_argument('--gpus', type=str, default='')
    parser.add_argument('--dataset_path', type=str, default='/nas/datasets/yixin/PCMA/sim_data')
    parser.add_argument('--N', type=int, default=100000)
    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--min_lr_ratio', type=float, default=0.1)
    parser.add_argument('--accum_steps', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--use_ema', action='store_true', default=False)
    parser.add_argument('--mse_epochs', type=int, default=40)
    parser.add_argument('--csi_warmup_epochs', type=int, default=5)
    parser.add_argument('--csi_beta', type=float, default=1.0)
    parser.add_argument('--alpha_after', type=float, default=0.0)
    parser.add_argument('--fade_mse', action='store_true', default=True)
    parser.add_argument('--mode_name', type=str, default=None)
    parser.add_argument('--max_nan_inf_count', type=int, default=100)

    args = parser.parse_args()

    setup_distributed(backend="nccl", timeout_seconds=7200)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if args.mode_name is None:
        mode_name = f'mixedmods_train_robust_rand_freqU[0,200]_phi1U[0.0000,6.2832]_phi2U[0.0000,6.2832]_ampU[0.20,0.90]_snrU[8,22]_N{args.N}_varsnr_ampr_phi1phi2_delay0T_c64'
    else:
        mode_name = args.mode_name

    base_output_dir = '/nas/datasets/yixin/PCMA/src'
    save_pic = os.path.join(base_output_dir, 'loss', f'loss_SigSep_resnet_{mode_name}.png')
    save_dir = os.path.join(base_output_dir, 'check_points', 'all')

    rank = get_rank()
    world = get_world_size()

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True, mode=0o755)
        os.makedirs(os.path.dirname(save_pic), exist_ok=True, mode=0o755)
        print(f"[Train] world_size={world}, device={device}")
        print(f"[Path] 保存目录: {save_dir}")
        print(f"[Path] 损失图片: {save_pic}")

    if is_dist_avail_and_initialized():
        torch.distributed.barrier()

    final_ckpt = os.path.join(save_dir, f'signal_separator_resnet_{mode_name}.pth')

    train_loader, val_loader, train_batches_local, val_batches_local = prepare_dataloaders(
        args.dataset_path, mode=mode_name, batch_size=args.batch_size,
        train_ratio=args.train_ratio, num_workers=args.num_workers,
        seed=2025, dist_rank=rank, dist_world_size=world
    )

    tb = torch.tensor([train_batches_local], device=device, dtype=torch.int32)
    vb = torch.tensor([val_batches_local], device=device, dtype=torch.int32)
    if is_dist_avail_and_initialized():
        torch.distributed.all_reduce(tb, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(vb, op=torch.distributed.ReduceOp.MIN)
    train_batches = int(tb.item())
    val_batches = int(vb.item())

    model, optimizer = initialize_model(device, lr=args.lr)
    if is_dist_avail_and_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == 'cuda' else None,
            output_device=local_rank if device.type == 'cuda' else None,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

    steps_per_epoch = math.ceil(train_batches / max(1, args.accum_steps))
    total_steps = args.epochs * max(1, steps_per_epoch)
    warmup_steps_eff = args.warmup_steps
    if args.warmup_epochs > 0:
        warmup_steps_eff = args.warmup_epochs * max(1, steps_per_epoch)

    scheduler = WarmupCosineSchedule(
        optimizer, warmup_steps=warmup_steps_eff,
        total_steps=total_steps, min_lr_ratio=args.min_lr_ratio
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    ema = EMA(model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model,
              decay=args.ema_decay) if args.use_ema else None

    train_losses, val_losses = [], []
    best_val = float('inf')
    latest_saved_path = None
    best_saved_path = None

    for epoch in range(args.epochs):
        alpha_t, beta_c = get_loss_weights(epoch, args)
        if rank == 0:
            print(f"\n=== Epoch {epoch+1}/{args.epochs} | α_time={alpha_t:.3f} | β_CSI={beta_c:.3f} ===")

        tr, nan_inf_count = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, ema, device,
            total_batches=train_batches, rank=rank, accum_steps=args.accum_steps,
            alpha_time=alpha_t, beta_csi=beta_c, max_grad_norm=args.max_grad_norm,
            max_nan_inf_count=args.max_nan_inf_count
        )
        if nan_inf_count >= args.max_nan_inf_count:
            if rank == 0:
                print(f"[Train][ERROR] Training stopped due to excessive NaN/Inf count: {nan_inf_count}/{args.max_nan_inf_count}")
                send_email(text=f"Training stopped due to excessive NaN/Inf count: {nan_inf_count}/{args.max_nan_inf_count}")
            break

        vl = validate_epoch(
            model, val_loader, device, total_batches=val_batches, rank=rank,
            alpha_time=alpha_t, beta_csi=beta_c, use_ema=args.use_ema, ema=ema
        )

        if rank == 0:
            train_losses.append(tr)
            val_losses.append(vl)
            print(f"Train Loss: {tr:.6f} | Val Loss(EMA): {vl:.6f} | NaN/Inf Count: {nan_inf_count}/{args.max_nan_inf_count}")
            plot_and_save_loss(train_losses, val_losses, save_pic)

            model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            latest_fname = os.path.join(save_dir, f'signal_separator_resnet_{mode_name}_latest_epoch{epoch+1}.pth')
            if latest_saved_path is not None and os.path.exists(latest_saved_path):
                try:
                    os.remove(latest_saved_path)
                except Exception:
                    pass
            torch.save(model_to_save.state_dict(), latest_fname)
            latest_saved_path = latest_fname

            if vl < best_val - 1e-8:
                best_val = vl
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

                best_fname = os.path.join(save_dir, f'signal_separator_resnet_{mode_name}_best_epoch{epoch+1}.pth')
                if best_saved_path is not None and os.path.exists(best_saved_path):
                    try:
                        os.remove(best_saved_path)
                    except Exception:
                        pass
                torch.save(state, best_fname)
                best_saved_path = best_fname
                print(f"[Best] updated: {best_saved_path} (val={best_val:.6f})")

    if rank == 0:
        if ema is not None:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                ema.apply_shadow(model.module)
                torch.save(model.module.state_dict(), final_ckpt)
                ema.restore(model.module)
            else:
                ema.apply_shadow(model)
                torch.save(model.state_dict(), final_ckpt)
                ema.restore(model)
        else:
            torch.save((model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model).state_dict(), final_ckpt)
        print(f"Training completed. Final: {final_ckpt} (best_val={best_val:.6f})")

    cleanup_distributed()


if __name__ == '__main__':
    main()
    send_email(text=f"Training completed (resnet)")  # 保持原有行为


