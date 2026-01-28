import json
from cs336_basics.tokenizer import Tokenizer

def load_tokenizer(model_dir):
    # 假设你保存为了 vocab.json 和 merges.txt
    with open(f"{model_dir}/vocab.json", "r") as f:
        # 确保 key 是 int
        vocab = {int(k): bytes.fromhex(v) if isinstance(v, str) else v.encode('latin1') for k, v in json.load(f).items()}
    
    merges = []
    with open(f"{model_dir}/merges.txt", "r") as f:
        for line in f:
            p1, p2 = line.strip().split()
            merges.append((bytes.fromhex(p1), bytes.fromhex(p2)))
            
    # 实例化你写的 Tokenizer 类
    # 注意：如果你的 Tokenizer 类初始化参数不同，请相应调整
    return Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

def analyze(name, tokenizer_path, test_file):
    print(f"=== 分析 Tokenizer: {name} ===")
    tokenizer = load_tokenizer(tokenizer_path)

    # 1. 寻找最长 Token
    longest_token_id = max(tokenizer.vocab.keys(), key=lambda k: len(tokenizer.vocab[k]))
    longest_token_bytes = tokenizer.vocab[longest_token_id]
    print(f"最长 Token (ID {longest_token_id}): 长度 {len(longest_token_bytes)}")
    print(f"内容 (尝试解码): {longest_token_bytes.decode('utf-8', errors='replace')}")

    # 2. 计算压缩比
    # 读取一小段测试文本（例如 1MB）
    with open(test_file, "r", encoding="utf-8") as f:
        text = f.read(1024 * 1024) # 读取 1MB
    
    encoded_ids = tokenizer.encode(text)
    original_bytes = len(text.encode("utf-8"))
    token_count = len(encoded_ids)
    
    compression_ratio = original_bytes / token_count
    print(f"原始字节数: {original_bytes}")
    print(f"Token 数量: {token_count}")
    print(f"压缩比 (Bytes/Token): {compression_ratio:.2f}")
    print("-" * 30)

# 使用示例
analyze("TinyStories", "models/tokenizer_tinystories", "assignment1-data/TinyStoriesV2-GPT4-valid.txt")
analyze("OpenWebText", "models/tokenizer_owt", "assignment1-data/owt_valid.txt")