import multiprocessing
import heapq
import os
from collections import Counter, defaultdict
from typing import BinaryIO
import pickle
import regex as re


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.id2token = vocab
        self.token2id = {v: k for k, v in self.vocab.items()}
        self.merges_ranks = {pair: i for i, pair in enumerate(merges)}
        self.pat = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            escaped_specials = [re.escape(s) for s in sorted_specials]
            self.special_pattern = re.compile(f"({'|'.join(escaped_specials)})")
            self.special_tokens_set = set(self.special_tokens)
        else:
            self.special_pattern = None
            self.special_tokens_set = set()
        self.cache = {}

    def _bpe_encode(self, token_bytes: bytes) -> list[int]:
        if token_bytes in self.cache:
            return self.cache[token_bytes]

        word = [bytes([b]) for b in token_bytes]
        while len(word) > 1:
            min_rank = float("inf")
            min_pair = None
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = self.merges_ranks.get(pair)
                if rank is not None and min_rank > rank:
                    min_rank = rank
                    min_pair = pair
            if min_pair is None:
                break
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == min_pair:
                    new_word.append(min_pair[0] + min_pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        ids = [self.token2id[token] for token in word]
        self.cache[token_bytes] = ids
        return ids

    def encode(self, tokens: str):
        ids = []
        if self.special_pattern:
            chunks = self.special_pattern.split(tokens)
        else:
            chunks = [tokens]

        for chunk in chunks:
            if chunk in self.special_tokens_set:
                ids.append(self.token2id[chunk.encode("utf-8")])
            elif chunk:
                for token_match in self.pat.finditer(chunk):
                    token_bytes = token_match.group().encode("utf-8")
                    ids.extend(self._bpe_encode(token_bytes))

        return ids

    def decode(self, ids: list[int]) -> str:
        res_bytes = []
        for i in ids:
            token = self.id2token.get(i)
            if token:
                res_bytes.append(token)
        combined = b"".join(res_bytes)
        return combined.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable):
        for text in iterable:
            yield from self.encode(text)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def process_chunk(input_path, start_pos, end_pos, special_tokens):
    """
    处理文件的一个块，返回该块内的 word_counts。
    """
    print(f"[{time.strftime('%H:%M:%S')}] 开始处理块 {start_pos}-{end_pos}")
    counts = Counter()
    # 使用字节模式读取，避免大文件一次性 load 到内存
    with open(input_path, "rb") as f:
        f.seek(start_pos)
        chunk_data = f.read(end_pos - start_pos).decode("utf-8", errors="ignore")
    print(f"[{time.strftime('%H:%M:%S')}] 读取完块 {start_pos}-{end_pos}，开始分词...")

    # 按照作业要求，先根据 special_tokens 分割，确保不跨界合并
    if special_tokens:
        escaped_special = [re.escape(t) for t in special_tokens]
        split_pattern = f"({'|'.join(escaped_special)})"
        parts = re.split(split_pattern, chunk_data)
    else:
        parts = [chunk_data]

    special_tokens_set = set(special_tokens) if special_tokens else set()
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    for part in parts:
        if part in special_tokens_set or not part:
            continue
        # 使用 finditer 比 findall 在某些情况下更省内存 [cite: 7]
        for match in re.finditer(PAT, part):
            token_bytes = match.group().encode("utf-8")
            counts[token_bytes] += 1
    filename = "chunk_{}_{}.cnt".format(start_pos, end_pos)
    print(f"[{time.strftime('%H:%M:%S')}] 处理完块 {start_pos}-{end_pos}，生成临时文件 {filename}。")
    
    with open(filename, "wb") as f:
        pickle.dump(counts, f)
    return filename

class DataBlock:
    def __init__(self, count, words):
        self.count = count
        self.words = words
    def __lt__(self, other):
        if self.count == other.count:
            return self.words > other.words
        return self.count > other.count

import time  # 新增
def train_tokenizer(input_path, vocab_size, special_tokens, num_procs=None):
    # [Log] 记录开始时间
    start_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] 开始训练 Tokenizer...")
    
    if num_procs is None:
        num_procs = min(multiprocessing.cpu_count(), 16) # 降到 16 甚至 8，稳一点
    

    # 1. 寻找边界
    print(f"[{time.strftime('%H:%M:%S')}] 正在寻找文件块边界...")
    split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"
    # 关键修改：把文件切成 64 份或者 128 份！
    # 这样每个块只有原来的 1/4 或 1/8 大小，绝对不会爆内存
    num_chunks = num_procs * 4 
    
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, split_token)    
    print(f"[{time.strftime('%H:%M:%S')}] 使用进程数: {num_procs}")
    print(f"[{time.strftime('%H:%M:%S')}] 找到 {len(boundaries)-1} 个文件块边界。")
    # 2. 并行执行
    print(f"[{time.strftime('%H:%M:%S')}] 启动进程池，开始预分词...")
    pool = multiprocessing.Pool(processes=num_procs)
    jobs = []
    for i in range(len(boundaries) - 1):
        jobs.append(
            pool.apply_async(
                process_chunk,
                (input_path, boundaries[i], boundaries[i + 1], special_tokens),
            )
        )
    pool.close()
    
    print(f"[{time.strftime('%H:%M:%S')}] 所有任务已分发，正在收集结果...")

    # 3. 合并结果 (关键卡顿点检测)
    final_word_counts = Counter()
    total_jobs = len(jobs)
    
    for i, job in enumerate(jobs):
        try:
            # 这里接收到的不再是巨大的 Counter，而是一个微小的文件名字符串
            temp_filename = job.get() 
            
            if (i + 1) % 5 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] 正在合并第 {i+1}/{total_jobs} 个块...")

            # 从磁盘加载数据
            with open(temp_filename, "rb") as f:
                chunk_counts = pickle.load(f)
            
            # 合并
            final_word_counts.update(chunk_counts)
            
            # 及时删除临时文件，清理磁盘空间
            os.remove(temp_filename)
            
            # 显式释放内存
            del chunk_counts
            
        except Exception as e:
            print(f"[ERROR] 处理任务 {i} 失败: {e}")
            continue
    print(f"[{time.strftime('%H:%M:%S')}] 等待进程池关闭...")
    pool.join()
    
    print(f"[{time.strftime('%H:%M:%S')}] 预分词完成。唯一单词数: {len(final_word_counts)}")

    # 准备 BPE
    print(f"[{time.strftime('%H:%M:%S')}] 正在构建初始统计数据...")
    word_counts = {
        tuple(bytes([b]) for b in k): v for k, v in final_word_counts.items()
    }
    
    # 显式释放大对象内存
    del final_word_counts 
    import gc
    gc.collect()

    merges = []
    pair_counts = defaultdict(int)
    pair_pos = defaultdict(set)

    print(f"[{time.strftime('%H:%M:%S')}] 正在统计初始 Pair 频率...")
    for word, count in word_counts.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += count
            pair_pos[pair].add(word)
    
    print(f"[{time.strftime('%H:%M:%S')}] 正在初始化堆 (Heapify)...")
    heap = [DataBlock(count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)

    # BPE 循环
    target_merges = vocab_size - 256 - len(special_tokens)
    print(f"[{time.strftime('%H:%M:%S')}] 开始 BPE 合并循环 (目标: {target_merges} 次)...")
    
    bpe_start_time = time.time()
    
    for i in range(target_merges):
        # 心跳包：每 100 次合并打印一次，证明程序活着
        if i % 100 == 0:
            elapsed = time.time() - bpe_start_time
            rate = (i + 1) / (elapsed + 1e-5)
            remaining = (target_merges - i) / rate
            print(f"[{time.strftime('%H:%M:%S')}] Merge {i}/{target_merges} | 速度: {rate:.2f} iter/s | 预计剩余: {remaining/60:.1f} min")

        most_common_pair = None
        while heap:
            data_block = heapq.heappop(heap)
            if pair_counts.get(data_block.words, 0) == data_block.count:
                most_common_pair = data_block.words
                break
        if most_common_pair is None:
            print(f"[{time.strftime('%H:%M:%S')}] 没有更多 Pair 可合并，提前退出。")
            break

        merges.append(most_common_pair)

        words_to_update = list(pair_pos[most_common_pair])

        for word in words_to_update:
            if word not in word_counts:
                continue

            count = word_counts[word]

            for idx in range(len(word) - 1):
                p = (word[idx], word[idx + 1])
                pair_counts[p] -= count
                if pair_counts[p] > 0:
                    heapq.heappush(heap, DataBlock(pair_counts[p], p))

            new_word_list = []
            idx = 0
            while idx < len(word):
                if idx < len(word) - 1 and (word[idx], word[idx + 1]) == most_common_pair:
                    new_word_list.append(most_common_pair[0] + most_common_pair[1])
                    idx += 2
                else:
                    new_word_list.append(word[idx])
                    idx += 1
            new_word = tuple(new_word_list)

            for idx in range(len(new_word) - 1):
                p = (new_word[idx], new_word[idx + 1])
                pair_counts[p] += count
                pair_pos[p].add(new_word)
                heapq.heappush(heap, DataBlock(pair_counts[p], p))

            if new_word in word_counts:
                word_counts[new_word] += count
            else:
                word_counts[new_word] = count
            del word_counts[word]

        del pair_counts[most_common_pair]
        del pair_pos[most_common_pair]

    print(f"[{time.strftime('%H:%M:%S')}] BPE 循环结束，构建最终词表...")
    
    vocab = {i: bytes([i]) for i in range(256)}

    # Merges
    for i, merge in enumerate(merges):
        vocab[256 + i] = merge[0] + merge[1]

    # Special tokens
    start_special_idx = 256 + len(merges)
    for i, token in enumerate(special_tokens):
        vocab[start_special_idx + i] = token.encode("utf-8")

    print(f"[{time.strftime('%H:%M:%S')}] 全部完成。总耗时: {(time.time() - start_time)/60:.2f} 分钟")
    return vocab, merges