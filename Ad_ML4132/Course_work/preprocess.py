"""

preprocess.py

任务：

1. 读取 LOTR 三本文本

2. 清洗文本（去章节号、空行，多空格）

3. 按段落切分

4. 注入特殊 token：<BOS> <EOS> <PARA>

5. 构建词表

6. 将段落转为 token id 序列

7. 保存 processed 数据到 data/processed/



需要实现的函数：

- load_raw_texts()

- clean_text(text)

- split_into_paragraphs(text)

- inject_special_tokens(paragraphs)

- build_vocab(paragraphs)

- paragraphs_to_token_ids(paragraphs, vocab)

- save_processed(paragraphs, vocab)



最终输出：

- paragraphs.json

- vocab.json

"""

import json
import re
import os
from typing import List, Dict, Tuple


def load_raw_texts(raw_dir="./data/raw"):
    """读取三本文本（自动处理编码问题），返回 [(style_id, text), ...]"""
    texts = []
    files = [
        "01 - The Fellowship Of The Ring.txt",
        "02 - The Two Towers.txt",
        "03 - The Return Of The King.txt"
    ]
    
    for idx, filename in enumerate(files, start=1):
        filepath = os.path.join(raw_dir, filename)
        if not os.path.exists(filepath):
            print(f"警告: 文件 {filepath} 不存在")
            continue

        # 尝试多种编码方式
        encodings_to_try = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

        text = None
        for enc in encodings_to_try:
            try:
                with open(filepath, "r", encoding=enc, errors="strict") as f:
                    text = f.read()
                print(f"成功使用编码 {enc} 读取 {filename}")
                break
            except Exception as e:
                print(f"使用编码 {enc} 读取失败，尝试下一种...")

        # 如果上述编码全部失败 → 最后兜底方案
        if text is None:
            with open(filepath, "r", encoding="latin-1", errors="ignore") as f:
                text = f.read()
            print(f"已使用兜底编码 latin-1(ignore) 读取 {filename}")

        texts.append((idx, text))

    return texts



def clean_text(text: str) -> str:
    """清洗字符、移除章节号"""
    # 移除章节号模式，如 "Chapter 1", "BOOK I", "_Chapter 1_", "Chapter I" 等
    # 匹配各种章节号格式
    chapter_patterns = [
        r'^\s*_?Chapter\s+\d+[._]?\s*.*?$',  # Chapter 1, _Chapter 1_
        r'^\s*_?Chapter\s+[IVX]+[._]?\s*.*?$',  # Chapter I, Chapter IV
        r'^\s*BOOK\s+[IVX]+[._]?\s*$',  # BOOK I, BOOK IV
        r'^\s*_?BOOK\s+[IVX]+[._]?\s*$',  # _BOOK I_
        r'^\s*PROLOGUE\s*$',
        r'^\s*FOREWORD\s*$',
        r'^\s*EPILOGUE\s*$',
        r'^\s*APPENDIX\s+[A-Z]?\s*$',
        r'^\s*THE FELLOWSHIP OF THE RING\s*$',
        r'^\s*THE TWO TOWERS\s*$',
        r'^\s*THE RETURN OF THE KING\s*$',
        r'^\s*being the first part of.*$',
        r'^\s*being the second part of.*$',
        r'^\s*being the third part of.*$',
        r'^.*The Lord of the Rings.*$',
    ]
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # 检查是否是章节号
        is_chapter = False
        for pattern in chapter_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                is_chapter = True
                break
        
        if not is_chapter:
            cleaned_lines.append(line)
    
    # 重新组合文本
    cleaned_text = '\n'.join(cleaned_lines)
    
    # 移除多余的空格（多个空格变为单个空格）
    cleaned_text = re.sub(r' +', ' ', cleaned_text)
    
    # 移除行首行尾空格
    cleaned_text = '\n'.join(line.strip() for line in cleaned_text.split('\n'))
    
    # 移除多个连续空行（保留单个空行用于段落分隔）
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    return cleaned_text.strip()


def split_into_paragraphs(text: str) -> List[str]:
    """按空行切自然段"""
    # 按双换行符（空行）切分段落
    paragraphs = re.split(r'\n\s*\n', text)
    
    # 过滤掉空段落和过短的段落（少于10个字符）
    paragraphs = [p.strip() for p in paragraphs if p.strip() and len(p.strip()) >= 10]
    
    return paragraphs


def inject_special_tokens(paragraphs: List[str]) -> List[List[str]]:
    """给段落加 <BOS>/<EOS>/<PARA> 并切词"""
    tokenized_paragraphs = []
    
    for idx, para in enumerate(paragraphs):
        # 简单的单词切分（按空格和标点）
        # 保留标点作为独立的token
        tokens = re.findall(r'\w+|[^\w\s]', para)
        
        # 添加特殊token：
        # - <PARA> 在每个段落开头（除了第一个段落）
        # - <BOS> 在段落内容开头
        # - <EOS> 在段落内容结尾
        tokenized_para = []
        if idx > 0:  # 不是第一个段落，添加 <PARA>
            tokenized_para.append('<PARA>')
        tokenized_para.extend(['<BOS>'] + tokens + ['<EOS>'])
        tokenized_paragraphs.append(tokenized_para)
    
    return tokenized_paragraphs


def build_vocab(paragraphs: List[List[str]]) -> Dict[str, int]:
    """构建词表并返回 vocab dict"""
    # 收集所有唯一的token
    all_tokens = set()
    for para in paragraphs:
        all_tokens.update(para)
    
    # 特殊token放在前面
    special_tokens = ['<BOS>', '<EOS>', '<PARA>', '<UNK>']
    
    # 构建词表：特殊token + 其他token（按字母顺序排序）
    vocab = {}
    idx = 0
    
    # 先添加特殊token（总是添加，即使不在数据中）
    for token in special_tokens:
        vocab[token] = idx
        idx += 1
    
    # 添加其他token（按字母顺序）
    other_tokens = sorted(all_tokens - set(special_tokens))
    for token in other_tokens:
        vocab[token] = idx
        idx += 1
    
    return vocab


def paragraphs_to_token_ids(paragraphs: List[List[str]], vocab: Dict[str, int]) -> List[List[int]]:
    """将 token 转成 id"""
    token_id_paragraphs = []
    unk_id = vocab.get('<UNK>', 0)
    
    for para in paragraphs:
        token_ids = []
        for token in para:
            token_id = vocab.get(token, unk_id)
            token_ids.append(token_id)
        token_id_paragraphs.append(token_ids)
    
    return token_id_paragraphs


def save_processed(paragraphs: List[List[int]], vocab: Dict[str, int], out_dir="./data/processed"):
    """保存成 json"""
    # 创建输出目录
    os.makedirs(out_dir, exist_ok=True)
    
    # 保存段落（token id序列）
    paragraphs_path = os.path.join(out_dir, "paragraphs.json")
    with open(paragraphs_path, 'w', encoding='utf-8') as f:
        json.dump(paragraphs, f, ensure_ascii=False, indent=2)
    
    # 保存词表
    vocab_path = os.path.join(out_dir, "vocab.json")
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    print(f"已保存处理后的数据到 {out_dir}")
    print(f"  - 段落数: {len(paragraphs)}")
    print(f"  - 词表大小: {len(vocab)}")


if __name__ == "__main__":
    # 主流程
    print("开始预处理 LOTR 文本...")
    
    # 1. 读取原始文本
    print("1. 读取原始文本...")
    raw_texts = load_raw_texts("./data/raw")
    print(f"   读取了 {len(raw_texts)} 本书")
    
    # 2. 清洗文本
    print("2. 清洗文本...")
    all_paragraphs = []
    for style_id, text in raw_texts:
        cleaned = clean_text(text)
        paragraphs = split_into_paragraphs(cleaned)
        print(f"   第 {style_id} 本书: {len(paragraphs)} 个段落")
        all_paragraphs.extend(paragraphs)
    
    print(f"   总共 {len(all_paragraphs)} 个段落")
    
    # 3. 注入特殊token并切词
    print("3. 注入特殊token并切词...")
    tokenized_paragraphs = inject_special_tokens(all_paragraphs)
    print(f"   已处理 {len(tokenized_paragraphs)} 个段落")
    
    # 4. 构建词表
    print("4. 构建词表...")
    vocab = build_vocab(tokenized_paragraphs)
    print(f"   词表大小: {len(vocab)}")
    
    # 5. 转换为token id
    print("5. 转换为token id...")
    token_id_paragraphs = paragraphs_to_token_ids(tokenized_paragraphs, vocab)
    
    # 6. 保存处理后的数据
    print("6. 保存处理后的数据...")
    save_processed(token_id_paragraphs, vocab, "./data/processed")
    
    print("预处理完成！")

