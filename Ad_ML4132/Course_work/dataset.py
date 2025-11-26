"""

dataset.py — 用于构建 LOTR 数据集的模块。

"""

import json

import torch

from torch.utils.data import Dataset, DataLoader

import os

def load_processed(process_dir="./data/processed"):

    """加载 paragraphs.json + vocab.json + style_id 自动生成"""

    paragraphs_path = os.path.join(process_dir, "paragraphs.json")

    vocab_path = os.path.join(process_dir, "vocab.json")

    with open(paragraphs_path, "r", encoding="utf-8") as f:

        paragraphs = json.load(f)

    with open(vocab_path, "r", encoding="utf-8") as f:

        vocab = json.load(f)

    # 三本书按顺序拼接，所以 style 需要按段落数量划分

    # 你在 preprocess 是将 3 本书依次 extend 的，因此 paragraphs 顺序为：

    # [书1全部段落] + [书2全部段落] + [书3全部段落]

    # 这里我们用简单方案：平均划分（段落数量几乎一致）

    total = len(paragraphs)

    per_book = total // 3

    style_ids = []

    for i in range(total):

        if i < per_book:

            style_ids.append(0)

        elif i < per_book * 2:

            style_ids.append(1)

        else:

            style_ids.append(2)

    data = list(zip(style_ids, paragraphs))

    return data, vocab

class LOTRDataset(Dataset):

    """返回 (style_id, input_ids, target_ids)"""

    def __init__(self, data, vocab):

        self.data = data

        self.vocab = vocab

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        style_id, token_ids = self.data[idx]

        # 输入为 [w0, w1, ... w(n-1)]

        input_ids = token_ids[:-1]

        # 目标为 [w1, w2, ... wn]

        target_ids = token_ids[1:]

        return style_id, torch.tensor(input_ids), torch.tensor(target_ids)

def collate_fn(batch):

    """padding + attention mask"""

    styles, inputs, targets = zip(*batch)

    lengths = [len(x) for x in inputs]

    max_len = max(lengths)

    padded_inputs = []

    padded_targets = []

    attn_masks = []

    for inp, tgt in zip(inputs, targets):

        pad_len = max_len - len(inp)

        padded_inp = torch.cat([inp, torch.zeros(pad_len, dtype=torch.long)])

        padded_tgt = torch.cat([tgt, torch.zeros(pad_len, dtype=torch.long)])

        mask = torch.cat([torch.ones(len(inp)), torch.zeros(pad_len)])

        padded_inputs.append(padded_inp)

        padded_targets.append(padded_tgt)

        attn_masks.append(mask)

    return (

        torch.tensor(styles),                     # (batch,)

        torch.stack(padded_inputs),               # (batch, max_len)

        torch.stack(padded_targets),              # (batch, max_len)

        torch.stack(attn_masks),                  # (batch, max_len)

    )

def get_dataloader(batch_size=32, shuffle=True):

    data, vocab = load_processed()

    dataset = LOTRDataset(data, vocab)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


