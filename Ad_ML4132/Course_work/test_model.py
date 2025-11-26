import torch
from model import StyleConditionedLSTM

vocab_size = 20000
token_dim = 64
style_dim = 16
hidden_size = 128

model = StyleConditionedLSTM(vocab_size, token_dim, style_dim, hidden_size)

input_ids = torch.randint(0, vocab_size, (2, 10))
style_ids = torch.tensor([0, 2])

logits = model(input_ids, style_ids)
print("logits shape:", logits.shape)
