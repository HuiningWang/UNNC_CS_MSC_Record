"""
model_transformer.py

Decoder-only Transformer 语言模型 + Style Embedding
- 兼容现有的 preprocess.py / dataset.py 输出的 token id
- 使用现有的 word-level vocab.json（1A）
- 支持 style_id 控制风格 (0/1/2)
- 适合 LOTR 小说生成（自回归，next-token prediction）

默认配置是「小模型」，方便在笔记本 3070 上验证。
在 Kaggle P100 上训练时，只需要调大 TransformerStyleConfig 里的参数即可。
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class TransformerStyleConfig:
    """
    Transformer 模型配置

    你需要关心的主要参数：
    - vocab_size: 词表大小（必须和 vocab.json 一致）
    - d_model:    Transformer 隐状态维度（embedding 维度）
    - n_heads:    Multi-head Attention 头数
    - n_layers:   Transformer 层数
    - dim_ff:     前馈层隐藏维度
    - max_seq_len: 支持的最大序列长度（>= 训练时截断长度）
    - num_styles: 风格数（现在固定 3，本任务 0/1/2）

    ✅ 目前默认是「小模型」用于本地 3070 验证：
        d_model=256, n_layers=2, n_heads=4, dim_ff=1024

    ⚠️ 如果你在 Kaggle P100 上训练「大一点」的模型，推荐改成：
        d_model=512, n_layers=6, n_heads=8, dim_ff=2048
    """

    vocab_size: int
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    dim_ff: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 512
    num_styles: int = 3  # LOTR 三本书的风格

    # 将来如果你换成 BPE，只需要：
    # 1. 换 tokenizer，重新生成 vocab_size
    # 2. 用新的 vocab_size 初始化这个 config
    # 模型内部不用改。


class TransformerStyleLM(nn.Module):
    """
    Decoder-only Transformer 语言模型（类似一个小号 GPT）+ Style Embedding
    接口和你现在的 LSTM 模型尽量保持一致：

    forward(
        input_ids: (batch, seq),
        style_ids: (batch,),
        attn_mask: (batch, seq) 或 None
    ) -> logits: (batch, seq, vocab_size)

    - input_ids 由 dataset.py 提供（除去最后一个 token）
    - style_ids 来自 LOTRDataset 中的 style_id
    - attn_mask 是 padding mask: 1 表示有效，0 表示 padding
    """

    def __init__(self, config: TransformerStyleConfig):
        super().__init__()
        self.config = config

        d_model = config.d_model
        vocab_size = config.vocab_size
        num_styles = config.num_styles
        max_seq_len = config.max_seq_len

        # token embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # 位置 embedding（learned）
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # 风格 embedding：直接映射到 d_model 维度，方便相加
        self.style_emb = nn.Embedding(num_styles, d_model)

        self.dropout = nn.Dropout(config.dropout)

        # 使用 PyTorch 自带的 TransformerEncoder 做一个「带因果 mask」的 decoder-only LM
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_ff,
            dropout=config.dropout,
            batch_first=True,  # 输入输出都是 (batch, seq, d_model)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            norm=nn.LayerNorm(d_model),
        )

        # 输出层：映射回 vocab
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重初始化
        self._reset_parameters()

    def _reset_parameters(self):
        # 简单的 Xavier 初始化
        nn.init.xavier_uniform_(self.token_emb.weight)
        nn.init.xavier_uniform_(self.pos_emb.weight)
        nn.init.xavier_uniform_(self.style_emb.weight)
        nn.init.xavier_uniform_(self.lm_head.weight)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        生成因果 Mask，防止看到未来信息
        shape: (seq_len, seq_len)，用于 TransformerEncoder 的 mask 参数
        """
        # 上三角为 -inf，表示不能注意未来位置
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        style_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        input_ids: (batch, seq)
        style_ids: (batch,)
        attn_mask: (batch, seq)  1=有效，0=padding（来自你的 collate_fn），可为 None

        return:
            logits: (batch, seq, vocab_size)
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"输入序列长度 {seq_len} 超过了模型配置的 max_seq_len={self.config.max_seq_len}，"
                f"请在 DataLoader / train 脚本中截断，或者增大 max_seq_len。"
            )

        # --------- Embedding 部分 ---------
        # 1) token embedding
        token_emb = self.token_emb(input_ids)  # (B, T, d_model)

        # 2) 位置 embedding
        # 位置索引：0,1,2,...,seq_len-1
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        pos_emb = self.pos_emb(pos_ids)  # (B, T, d_model)

        # 3) style embedding：按 batch 取 style_vec，然后 broadcast 到每个时间步
        style_vec = self.style_emb(style_ids)  # (B, d_model)
        style_emb = style_vec.unsqueeze(1).expand(batch_size, seq_len, -1)  # (B, T, d_model)

        # 4) 三者相加作为最终输入
        x = token_emb + pos_emb + style_emb
        x = self.dropout(x)  # (B, T, d_model)

        # --------- 构造 mask ---------
        # 因果 mask：防止当前时间步看到未来 token
        causal_mask = self._generate_causal_mask(seq_len, device)  # (T, T)

        # padding mask：TransformerEncoder 里 True 表示 padding，需要被忽略
        if attn_mask is not None:
            # attn_mask: 1 表示有效 token，0 表示 padding
            # src_key_padding_mask: True 位置 = 需要屏蔽的位置（padding）
            src_key_padding_mask = (attn_mask == 0)  # (B, T), bool
        else:
            src_key_padding_mask = None

        # --------- 通过 TransformerEncoder（decoder-only 模式） ---------
        # 注意：这里使用的是 encoder，但由于加了 causal_mask，它相当于一个自回归 decoder。
        hidden_states = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
        )  # (B, T, d_model)

        logits = self.lm_head(hidden_states)  # (B, T, vocab_size)
        return logits
