"""
generate.py
使用训练好的 Style-Conditioned Retro-Attention LSTM 生成文本

功能：
- 加载 model_best.pt
- 根据 style_id 生成一段文本
- 支持简单的 temperature + top-k 采样
"""

import torch
import torch.nn.functional as F
from model import StyleConditionedLSTM


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = "./model_best(1).pt"


def load_model_and_vocab(ckpt_path=CHECKPOINT_PATH):
    """从保存的 checkpoint 中加载模型和词表"""
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    vocab = ckpt["vocab"]
    config = ckpt["config"]
    vocab_size = config["vocab_size"]
    token_dim = config["token_dim"]
    style_dim = config["style_dim"]
    hidden_size = config["hidden_size"]

    # 如果以后你在 train.py 里也把 num_layers / dropout 写进 config，就用 get 读出来
    num_layers = config.get("num_layers", 2)   # 本次训练我们知道是 2 层
    dropout = config.get("dropout", 0.2)

    model = StyleConditionedLSTM(
        vocab_size=vocab_size,
        token_dim=token_dim,
        style_dim=style_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 构建 id → token 的反向词表
    id2token = {idx: tok for tok, idx in vocab.items()}

    return model, vocab, id2token


def sample_from_logits(logits, temperature=1.0, top_k=20):
    """
    从 logits 中采样一个 token id
    logits: (vocab_size,)
    """
    logits = logits / max(temperature, 1e-6)

    if top_k is not None and top_k > 0:
        # 只保留 top_k 的概率，其他设为 -inf
        values, indices = torch.topk(logits, top_k)
        mask = torch.full_like(logits, float("-inf"))
        mask[indices] = logits[indices]
        logits = mask

    probs = F.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1)
    return next_id.item()


def decode_tokens(token_ids, id2token):
    """将 token id 序列解码为可读文本"""
    tokens = []
    for tid in token_ids:
        tok = id2token.get(tid, "<UNK>")
        if tok in ["<BOS>", "<EOS>"]:
            continue
        if tok == "<PARA>":
            tokens.append("\n\n")
        else:
            tokens.append(tok)

    # 简单拼接：空格连接单词，再把标点处理一下
    text = " ".join(tokens)
    # 处理标点前的空格
    text = text.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
    text = text.replace(" ;", ";").replace(" :", ":").replace(" '", "'")
    return text


def generate_text(model, vocab, id2token, style_id=0, max_len=50, temperature=1.0, top_k=20):
    """
    使用给定 style_id 生成一段文本
    style_id: 0 / 1 / 2
    """
    # 找到特殊 token 的 id
    bos_id = vocab.get("<BOS>", 0)
    eos_id = vocab.get("<EOS>", None)

    # 初始输入：只放一个 <BOS>
    generated_ids = [bos_id]

    # batch=1
    style_ids = torch.tensor([style_id], dtype=torch.long, device=DEVICE)

    for _ in range(max_len):
        input_ids = torch.tensor(generated_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)  # (1, T)

        with torch.no_grad():
            logits = model(input_ids, style_ids)  # (1, T, vocab)
            last_logits = logits[0, -1, :]        # (vocab,)

        next_id = sample_from_logits(last_logits, temperature=temperature, top_k=top_k)

        generated_ids.append(next_id)

        if eos_id is not None and next_id == eos_id:
            break

    text = decode_tokens(generated_ids, id2token)
    return text


def main():
    print(f"使用设备: {DEVICE}")
    print("加载模型和词表...")
    model, vocab, id2token = load_model_and_vocab()

    # 生成三种风格的示例
    for style_id, name in [(0, "Style 0 (Fellowship?)"),
                           (1, "Style 1 (Two Towers?)"),
                           (2, "Style 2 (ROTK?)")]:
        print("\n" + "=" * 60)
        print(f"生成风格 {style_id}: {name}")
        text = generate_text(
            model,
            vocab,
            id2token,
            style_id=style_id,
            max_len=1000,        # 生成长度可以再调
            temperature=0.5,   # 温度大一点更随机
            top_k=30,
        )
        print(text)
        print("=" * 60)


if __name__ == "__main__":
    main()
