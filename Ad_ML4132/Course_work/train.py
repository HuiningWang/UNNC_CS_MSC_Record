"""
train.py
训练 Style-Conditioned Retro-Attention LSTM Language Model

依赖：
- preprocess.py 已经生成 ./data/processed/paragraphs.json, vocab.json
- dataset.py: load_processed, LOTRDataset, collate_fn
- model.py: StyleConditionedLSTM

功能：
- 加载数据
- 构建 DataLoader
- 训练模型（交叉熵 + padding mask）
- 保存最优模型 model_best.pt
"""

import os
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import load_processed, LOTRDataset, collate_fn
from model import StyleConditionedLSTM


# ================== 超参数 ==================
BATCH_SIZE = 16
EPOCHS = 1
LR = 1e-3
TOKEN_DIM = 128      # token embedding 维度
STYLE_DIM = 16       # style embedding 维度
HIDDEN_SIZE = 256    # LSTM hidden size
MAX_SEQ_LEN = 256    # 训练时最大序列长度（截断，防止过长）
GRAD_CLIP = 1.0
DEBUG_SMALL_DATA = True      # 改成 False 就用全部数据
DEBUG_LIMIT = 200            # 使用前 200 段


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataloader():
    """加载 processed 数据并构建 DataLoader"""
    data, vocab = load_processed("./data/processed")

    if DEBUG_SMALL_DATA:
        print(f"[DEBUG] 使用前 {DEBUG_LIMIT} 段进行快速训练验证")
        data = data[:DEBUG_LIMIT]

    dataset = LOTRDataset(data, vocab)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    return dataloader, vocab


def build_model(vocab_size: int) -> nn.Module:
    """构建模型并放到 device 上"""
    model = StyleConditionedLSTM(
        vocab_size=vocab_size,
        token_dim=TOKEN_DIM,
        style_dim=STYLE_DIM,
        hidden_size=HIDDEN_SIZE,
    )
    return model.to(DEVICE)


def compute_loss(logits, targets, mask):
    """
    logits: (batch, seq, vocab)
    targets: (batch, seq)
    mask: (batch, seq)  1=有效位置, 0=padding
    """
    batch_size, seq_len, vocab_size = logits.size()

    # 展平
    logits_flat = logits.view(-1, vocab_size)          # (B*T, V)
    targets_flat = targets.view(-1)                    # (B*T,)
    mask_flat = mask.view(-1)                          # (B*T,)

    # 每个 token 的交叉熵
    loss_all = F.cross_entropy(logits_flat, targets_flat, reduction="none")
    # 只对 mask=1 的位置计算
    loss_masked = loss_all * mask_flat
    # 防止除0
    denom = mask_flat.sum().clamp(min=1.0)
    loss = loss_masked.sum() / denom
    return loss


def train():
    print(f"使用设备: {DEVICE}")

    print("1. 构建 DataLoader...")
    dataloader, vocab = build_dataloader()
    vocab_size = len(vocab)
    print(f"   词表大小: {vocab_size}")
    print(f"   训练样本数: {len(dataloader.dataset)}")

    print("2. 构建模型...")
    model = build_model(vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_loss = math.inf

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0.0

        print(f"\n===== Epoch {epoch}/{EPOCHS} =====")
        for step, batch in enumerate(dataloader, start=1):
            style_ids, input_ids, target_ids, attn_mask = batch

            # 截断过长序列（提高训练速度，防止显存爆）
            input_ids = input_ids[:, :MAX_SEQ_LEN]
            target_ids = target_ids[:, :MAX_SEQ_LEN]
            attn_mask = attn_mask[:, :MAX_SEQ_LEN]

            # 移动到设备
            style_ids = style_ids.to(DEVICE)
            input_ids = input_ids.to(DEVICE)
            target_ids = target_ids.to(DEVICE)
            attn_mask = attn_mask.to(DEVICE)

            optimizer.zero_grad()

            # 前向
            logits = model(input_ids, style_ids)

            # 计算 loss
            loss = compute_loss(logits, target_ids, attn_mask)

            # 反向传播
            loss.backward()
            # 梯度裁剪（LSTM 容易梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            # 统计
            num_tokens = attn_mask.sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            if step % 50 == 0:
                avg_loss = total_loss / max(total_tokens, 1.0)
                print(f"  step {step:4d} | loss = {loss.item():.4f} | avg = {avg_loss:.4f}")

        # 一个 epoch 结束
        epoch_loss = total_loss / max(total_tokens, 1.0)
        print(f"Epoch {epoch} finished. Avg loss = {epoch_loss:.4f}")

        # 保存最优模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_path = "./model_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                    "config": {
                        "vocab_size": vocab_size,
                        "token_dim": TOKEN_DIM,
                        "style_dim": STYLE_DIM,
                        "hidden_size": HIDDEN_SIZE,
                    },
                },
                save_path,
            )
            print(f"  => 保存新最佳模型到 {save_path} （loss = {best_loss:.4f}）")

    print("\n训练完成！")


if __name__ == "__main__":
    train()
