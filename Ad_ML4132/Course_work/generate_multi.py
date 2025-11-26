"""
generate_multi.py
生成多段不同风格的小说，每段 300-400 个单词
前 3 段固定风格：0, 1, 2
后 2 段随机从 {0,1,2} 中选择
"""

import torch
import random
import torch.nn.functional as F
from model import StyleConditionedLSTM


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 你可以改成 "./model_best.pt" 来使用本地200段训练模型
CHECKPOINT_PATH = "./model_best(1).pt"


# ============================================================
# 加载模型 + 词表
# ============================================================

def load_model_and_vocab(ckpt_path=CHECKPOINT_PATH):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    vocab = ckpt["vocab"]
    config = ckpt["config"]

    model = StyleConditionedLSTM(
        vocab_size=config["vocab_size"],
        token_dim=config["token_dim"],
        style_dim=config["style_dim"],
        hidden_size=config["hidden_size"],
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.2),
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    id2token = {idx: tok for tok, idx in vocab.items()}

    return model, vocab, id2token


# ============================================================
# 采样函数
# ============================================================

def sample_from_logits(logits, temperature=1.0, top_k=50):
    logits = logits / max(temperature, 1e-6)

    # top-k 掩码
    if top_k is not None and top_k > 0:
        values, indices = torch.topk(logits, top_k)
        mask = torch.full_like(logits, float("-inf"))
        mask[indices] = logits[indices]
        logits = mask

    probs = torch.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, 1)
    return next_id.item()


# ============================================================
# 解码
# ============================================================

def decode_tokens(token_ids, id2token):
    tokens = []
    for tid in token_ids:
        tok = id2token.get(tid, "<UNK>")

        if tok in ["<BOS>", "<EOS>"]:
            continue

        if tok == "<PARA>":
            tokens.append("\n\n")
        else:
            tokens.append(tok)

    text = " ".join(tokens)
    text = (text.replace(" ,", ",").replace(" .", ".")
                 .replace(" !", "!").replace(" ?", "?")
                 .replace(" ;", ";").replace(" :", ":")
                 .replace(" ’", "’").replace(" '", "'"))
    return text.strip()


# ============================================================
# 单段生成（目标 300–400 个单词）
# ============================================================

def generate_paragraph(
    model, vocab, id2token,
    style_id,
    min_words=300,
    max_words=400,
    temperature=0.9,
    top_k=50,
):
    bos_id = vocab["<BOS>"]
    eos_id = vocab["<EOS>"]

    generated_ids = [bos_id]
    style_ids = torch.tensor([style_id], dtype=torch.long, device=DEVICE)

    for i in range(max_words):
        inp = torch.tensor(generated_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

        with torch.no_grad():
            logits = model(inp, style_ids)
            last_logits = logits[0, -1, :]

        next_id = sample_from_logits(last_logits, temperature=temperature, top_k=top_k)

        # 不允许太早结束（必须 >= min_words）
        if next_id == eos_id and i < min_words:
            continue

        generated_ids.append(next_id)

        if next_id == eos_id:
            break

    text = decode_tokens(generated_ids, id2token)
    return text


# ============================================================
# 多段小说生成器
# ============================================================

def generate_multi_paragraph(
    model,
    vocab,
    id2token,
    style_sequence,           # 如 [0,1,2,?,?]
    min_words=300,
    max_words=400,
):
    paragraphs = []

    for idx, style_id in enumerate(style_sequence, 1):
        print(f"\n正在生成第 {idx} 段（风格 {style_id}）...")
        para = generate_paragraph(
            model, vocab, id2token,
            style_id=style_id,
            min_words=min_words,
            max_words=max_words,
            temperature=1.0,
            top_k=50,
        )
        paragraphs.append(para)

    return "\n\n".join(paragraphs)


# ============================================================
# 主程序
# ============================================================

def main():
    print(f"使用设备：{DEVICE}")
    print("加载模型与词表...")

    model, vocab, id2token = load_model_and_vocab()

    # 前三段固定风格：0, 1, 2
    fixed_styles = [0, 1, 2]

    # 后两段随机风格
    random_styles = [random.choice([0, 1, 2]) for _ in range(2)]

    style_sequence = fixed_styles + random_styles

    print("本次生成段落的风格顺序：", style_sequence)

    novel = generate_multi_paragraph(
        model,
        vocab,
        id2token,
        style_sequence,
        min_words=300,
        max_words=400,
    )

    print("\n\n================ 生成结果 ==================\n")
    print(novel)
    print("\n================ 结束 ==================\n")


if __name__ == "__main__":
    main()
