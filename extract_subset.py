#!/usr/bin/env python3
"""
从较大的 .pth 数据集中抽取小子集并保存为新的 .pth 文件。

用法示例：
  python extract_subset.py --input bigset.pth --output smallset_N2000.pth --N 2000
  python extract_subset.py --input bigset.pth --output smallset_rand2000.pth --N 2000 --random --seed 2025

选项：
  --require_bits  若指定，则只从有参考比特（bits1 != -1）的条目中抽取。
"""

import argparse
import random
import torch


def load_dataset(path):
    ds = torch.load(path, map_location="cpu")
    return ds


def save_dataset(entries, out_path):
    torch.save(entries, out_path)


def main():
    p = argparse.ArgumentParser(description="Extract subset from .pth dataset")
    p.add_argument("--input", required=True, help="输入 .pth 文件路径")
    p.add_argument("--output", required=True, help="输出子集 .pth 路径")
    p.add_argument("--N", type=int, default=2000, help="要抽取的样本数 (默认 2000)")
    p.add_argument("--random", action="store_true", help="是否随机抽样（默认按顺序取前 N）")
    p.add_argument("--seed", type=int, default=2025, help="随机抽样种子")
    p.add_argument("--require_bits", action="store_true", help="只从包含参考比特（bits1 != -1）的条目中抽取")
    args = p.parse_args()

    print(f"Loading dataset: {args.input}")
    dataset = load_dataset(args.input)
    total = len(dataset)
    print(f"Total entries in input: {total}")

    indices = list(range(total))

    if args.require_bits:
        filtered = [i for i, e in enumerate(dataset) if not (isinstance(e.get('bits1', None), int) and e.get('bits1') == -1)]
        print(f"Entries with bits (bits1 != -1): {len(filtered)}")
        indices = filtered

    if args.random:
        random.seed(args.seed)
        if len(indices) <= args.N:
            chosen = indices.copy()
        else:
            chosen = random.sample(indices, args.N)
    else:
        chosen = indices[: args.N]

    chosen.sort()
    out_entries = [dataset[i] for i in chosen]
    print(f"Selected {len(out_entries)} entries, saving to {args.output}")
    save_dataset(out_entries, args.output)
    print("Done.")


if __name__ == '__main__':
    main()
