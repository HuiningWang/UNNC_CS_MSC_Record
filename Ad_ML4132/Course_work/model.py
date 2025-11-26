"""
model.py
核心模型：Style-Conditioned Retro-Attention LSTM Language Model

模块：
- token embedding
- style embedding
- LSTM decoder
- Retro Bahdanau attention (对历史 hidden_states)
- 输出层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================
# Retro Bahdanau Attention
# ======================================================
class RetroAttention(nn.Module):
    """
    Bahdanau Additive Attention over previous hidden states

    current_state:  (batch, hidden)
    history_states: (batch, t-1, hidden)
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.W_h = nn.Linear(hidden_size, hidden_size)
        self.W_s = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, current_state, history_states):
        # 如果没有历史（t=0），返回零 context
        if history_states is None or history_states.size(1) == 0:
            batch, hidden = current_state.size()
            context = torch.zeros(batch, hidden, device=current_state.device)
            attn_weights = torch.zeros(batch, 1, device=current_state.device)
            return context, attn_weights

        # current_state: (batch, hidden) → (batch, 1, hidden)
        current_proj = self.W_s(current_state).unsqueeze(1)          # (b,1,h)
        history_proj = self.W_h(history_states)                      # (b,t,h)

        # score: (batch, t, 1)
        score = self.v(torch.tanh(history_proj + current_proj))      # (b,t,1)

        # attn_weights: (batch, t)
        attn_weights = torch.softmax(score.squeeze(-1), dim=1)       # (b,t)

        # context: Σ α_i * h_i → (batch, hidden)
        context = torch.sum(history_states * attn_weights.unsqueeze(-1), dim=1)

        return context, attn_weights


# ======================================================
# Style-Conditioned LSTM Decoder
# ======================================================
class StyleConditionedLSTM(nn.Module):
    """
    Decoder-only LSTM + Retro-Attention + Style Embedding
    """

    def __init__(self, vocab_size, token_dim, style_dim, hidden_size,
                 num_layers=2, dropout=0.2):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, token_dim)
        # 3 种风格：0,1,2
        self.style_emb = nn.Embedding(3, style_dim)

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # LSTM 输入 = token_emb + style_emb
        self.lstm = nn.LSTM(
            input_size=token_dim + style_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.attn = RetroAttention(hidden_size)

        # 输出层：concat(h_t, context, style_vec) → vocab
        self.output_layer = nn.Linear(hidden_size * 2 + style_dim, vocab_size)

    def forward(self, input_ids, style_ids):
        """
        input_ids: (batch, seq)
        style_ids: (batch,)
        return:
            logits: (batch, seq, vocab_size)
        """
        batch, seq_len = input_ids.shape

        # token embedding: (b, seq, token_dim)
        token_emb = self.token_emb(input_ids)

        # style embedding: (b, style_dim) → (b, seq, style_dim)
        style_vec = self.style_emb(style_ids)                 # (b, style_dim)
        style_expanded = style_vec.unsqueeze(1).repeat(1, seq_len, 1)

        # LSTM 输入拼接
        lstm_inp = torch.cat([token_emb, style_expanded], dim=-1)  # (b, seq, token_dim+style_dim)

        # 一次性跑完所有时间步
        # outputs: (b, seq, hidden)
        outputs, _ = self.lstm(lstm_inp)

        logits_list = []

        for t in range(seq_len):
            current_state = outputs[:, t, :]           # (b, hidden)
            if t == 0:
                history_states = None                  # 第一个词没有历史
            else:
                history_states = outputs[:, :t, :]     # (b, t, hidden)

            # retro-attention
            context, _ = self.attn(current_state, history_states)   # (b, hidden)

            # 拼接 [h_t, context, style_vec]
            full_vec = torch.cat([current_state, context, style_vec], dim=-1)  # (b, 2h+style_dim)

            logits_t = self.output_layer(full_vec)     # (b, vocab)
            logits_list.append(logits_t.unsqueeze(1))  # (b,1,vocab)

        logits = torch.cat(logits_list, dim=1)         # (b, seq, vocab)
        return logits
