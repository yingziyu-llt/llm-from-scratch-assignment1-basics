import numpy as np
import os
import json
from cs336_basics.tokenizer import Tokenizer

def load_tokenizer(model_dir):
    # 兼容两种保存格式：vocab.json 里的 value 可能是 hex string 也可能是 raw string
    with open(os.path.join(model_dir, "vocab.json"), "r") as f:
        vocab_data = json.load(f)
        vocab = {}
        for k, v in vocab_data.items():
            try:
                # 尝试 hex 解码
                if isinstance(v, str):
                    vocab[int(k)] = bytes.fromhex(v)
                else:
                    # 如果不是 hex，回退到 latin1 编码 (不太常见，但以防万一)
                    vocab[int(k)] = v.encode('latin1')
            except ValueError:
                # 如果 hex 解码失败，说明原样保存的字符串
                vocab[int(k)] = v.encode('utf-8') if isinstance(v, str) else v

    merges = []
    merges_path = os.path.join(model_dir, "merges.txt")
    if os.path.exists(merges_path):
        with open(merges_path, "r") as f:
            for line in f:
                if not line.strip(): continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    p1, p2 = parts[0], parts[1]
                    merges.append((bytes.fromhex(p1), bytes.fromhex(p2)))
            
    return Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

def process_file(tokenizer, input_path, output_path):
    if not os.path.exists(input_path):
        print(f"[跳过] 文件不存在: {input_path}")
        return

    print(f"正在处理: {input_path} -> {output_path}")
    
    file_size = os.path.getsize(input_path)
    processed_bytes = 0
    token_list = []
    
    # 10MB 的块大小
    chunk_size = 10 * 1024 * 1024 
    
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            text = f.read(chunk_size)
            if not text:
                break
            
            ids = tokenizer.encode(text)
            token_list.extend(ids)
            
            processed_bytes += len(text.encode('utf-8'))
            progress = (processed_bytes / file_size) * 100
            print(f"\r进度: {progress:.1f}% ({len(token_list)/1e6:.2f}M tokens)", end="")
            
    print(f"\n正在保存为 NumPy 格式 (.npy)...")
    token_array = np.array(token_list, dtype=np.uint16)
    np.save(output_path, token_array)
    print(f"完成。保存大小: {os.path.getsize(output_path)/1024/1024:.2f} MB")

if __name__ == "__main__":
    # ---------------------------------------------------------
    # 1. TinyStories 处理
    # ---------------------------------------------------------
    print("=== TinyStories 预处理 ===")
    # 确保这里的路径指向你保存 TinyStories tokenizer 的文件夹
    ts_tokenizer_path = "models/tokenizer_tinystories" 
    
    if os.path.exists(ts_tokenizer_path):
        ts_tokenizer = load_tokenizer(ts_tokenizer_path)
        
        # 训练集
        process_file(ts_tokenizer, 
                     "assignment1-data/TinyStoriesV2-GPT4-train.txt", 
                     "assignment1-data/TinyStoriesV2-GPT4-train.npy")
        
        # 验证集
        process_file(ts_tokenizer, 
                     "assignment1-data/TinyStoriesV2-GPT4-valid.txt", 
                     "assignment1-data/TinyStoriesV2-GPT4-valid.npy")
    else:
        print(f"未找到 TinyStories Tokenizer 目录: {ts_tokenizer_path}，跳过。")

    # ---------------------------------------------------------
    # 2. OpenWebText 处理
    # ---------------------------------------------------------
    print("\n=== OpenWebText 预处理 ===")
    # 确保这里的路径指向你保存 OWT tokenizer 的文件夹
    owt_tokenizer_path = "models/tokenizer_owt" 
    
    if os.path.exists(owt_tokenizer_path):
        owt_tokenizer = load_tokenizer(owt_tokenizer_path)
        
        # 训练集 (大文件，耐心等待)
        process_file(owt_tokenizer, 
                     "assignment1-data/owt_train.txt", 
                     "assignment1-data/owt_train.npy")
        
        # 验证集
        process_file(owt_tokenizer, 
                     "assignment1-data/owt_valid.txt", 
                     "assignment1-data/owt_valid.npy")
    else:
        print(f"未找到 OWT Tokenizer 目录: {owt_tokenizer_path}，跳过。")