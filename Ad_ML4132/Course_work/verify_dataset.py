"""

verify_dataset.py — 用于验证 dataset.py 是否正确运行

运行：

    python verify_dataset.py

"""

import torch

from dataset import load_processed, LOTRDataset, collate_fn

def verify_single():

    print("=== 单条样本检查 ===")

    data, vocab = load_processed()

    ds = LOTRDataset(data, vocab)

    style, inp, tgt = ds[0]

    print("style_id:", style)

    print("input 前20:", inp[:20])

    print("target前20:", tgt[:20])

def verify_batch():

    print("=== DataLoader batch 检查 ===")

    data, vocab = load_processed()

    ds = LOTRDataset(data, vocab)

    dl = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=collate_fn)

    batch = next(iter(dl))

    style, inp, tgt, mask = batch

    print("style shape:", style.shape)

    print("input shape:", inp.shape)

    print("target shape:", tgt.shape)

    print("mask shape:", mask.shape)

if __name__ == "__main__":

    verify_single()

    verify_batch()

    print("验证完成")


