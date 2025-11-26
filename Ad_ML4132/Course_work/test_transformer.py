import torch

from model_transformer import TransformerStyleConfig, TransformerStyleLM

# ===== 测试配置（可根据你实际 vocab 大小修改） =====
vocab_size = 20000          # 测试时随便设一个
batch_size = 2
seq_len = 10

# ===== 构建模型（使用小模型参数，和你笔记本兼容） =====
config = TransformerStyleConfig(
    vocab_size=vocab_size,
    d_model=256,      # 3070-friendly
    n_heads=4,
    n_layers=2,
    dim_ff=1024,
    max_seq_len=512,
)

model = TransformerStyleLM(config)

# ===== 构造假数据 =====
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
style_ids = torch.tensor([0, 2])  # 两个风格

# padding mask: 全 1，表示没有 padding
attn_mask = torch.ones(batch_size, seq_len)

# ===== 运行 forward =====
logits = model(input_ids, style_ids, attn_mask)

print("input_ids shape:", input_ids.shape)
print("style_ids shape:", style_ids.shape)
print("logits shape:", logits.shape)  # (batch, seq_len, vocab)
