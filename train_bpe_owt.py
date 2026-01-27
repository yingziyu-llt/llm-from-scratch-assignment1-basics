import os
import pickle
import json
from cs336_basics.tokenizer import train_tokenizer

# 1. 定义路径和参数
input_data_path = "assignment1-data/owt_train.txt" # 替换为你本地数据集的实际路径 
vocab_size = 32000 # 作业要求的词表大小 [cite: 1116]
special_tokens = ["<|endoftext|>"] # 必须添加的特殊 token 
output_dir = "models/tokenizer_owt"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. 执行训练
print(f"开始在 {input_data_path} 上训练 BPE Tokenizer...")
# 该函数返回 vocab (dict) 和 merges (list) [cite: 236]
vocab, merges = train_tokenizer(
    input_path=input_data_path,
    vocab_size=vocab_size,
    special_tokens=special_tokens
)

# 3. 序列化保存到磁盘 [cite: 251]
# 建议保存为作业要求的格式，以便后续 Tokenizer 类加载
with open(os.path.join(output_dir, "vocab.json"), "w", encoding="utf-8") as f:
    # 注意：vocab 里的 key 是 int，value 是 bytes，需要转换以便 JSON 序列化
    json.dump({k: v.hex() for k, v in vocab.items()}, f) 

with open(os.path.join(output_dir, "merges.txt"), "w", encoding="utf-8") as f:
    for pair in merges:
        # 保存为 token1_hex token2_hex 格式
        f.write(f"{pair[0].hex()} {pair[1].hex()}\n")

print(f"训练完成！结果已保存至 {output_dir}")
