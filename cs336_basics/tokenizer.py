import os
from collections import Counter, defaultdict
from typing import BinaryIO

import regex as re
import torch
import torch.nn as nn


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.id2token = vocab
        self.token2id = {v: k for k,v in self.vocab.items()}
        self.merges_ranks = {pair: i for i, pair in enumerate(merges)}
        self.pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
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
            min_rank = float('inf')
            min_pair = None
            for i in range(len(word) - 1):
                pair = (word[i],word[i + 1])
                rank = self.merges_ranks.get(pair)
                if rank is not None and min_rank > rank:
                    min_rank = rank
                    min_pair = pair
            if min_pair is None:
                break
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == min_pair:
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
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

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

def train_tokenizer(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 单线程版
    with open(input_path, "r", encoding="utf-8") as f:
        full_data = f.read()

    # 处理 Special Tokens
    if special_tokens:
        escaped_special = [re.escape(t) for t in special_tokens]
        split_pattern = f"({'|'.join(escaped_special)})"
        raw_chunks = re.split(split_pattern, full_data)
    else:
        raw_chunks = [full_data]

    # Pre-tokenization
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    special_tokens_set = set(special_tokens) if special_tokens else set()
    word_counts = Counter()

    for chunk in raw_chunks:
        if chunk in special_tokens_set or not chunk:
            continue
        pre_tokenized = re.findall(PAT, chunk)
        for token_str in pre_tokenized:
            token_bytes = token_str.encode("utf-8")
            word_counts[token_bytes] += 1
    
    word_counts = {tuple(bytes([b]) for b in k): v for k, v in word_counts.items()}
    
    merges = []
    pair_counts = defaultdict(int)
    pair_pos = defaultdict(set)
    
    for word, count in word_counts.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += count
            pair_pos[pair].add(word)
    
    for _ in range(vocab_size - 256 - len(special_tokens)):
        if not pair_counts:
            break
        most_common_pair = max(pair_counts, key=lambda x: (pair_counts[x], x))
        merges.append(most_common_pair)
        
        words_to_update = list(pair_pos[most_common_pair])
        
        for word in words_to_update:
            if word not in word_counts:
                continue
                
            count = word_counts[word]
            
            for i in range(len(word) - 1):
                p = (word[i], word[i + 1])
                pair_counts[p] -= count
            
            new_word_list = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == most_common_pair:
                    new_word_list.append(most_common_pair[0] + most_common_pair[1])
                    i += 2
                else:
                    new_word_list.append(word[i])
                    i += 1
            new_word = tuple(new_word_list)
            
            for i in range(len(new_word) - 1):
                p = (new_word[i], new_word[i + 1])
                pair_counts[p] += count
                pair_pos[p].add(new_word)
            
            if new_word in word_counts:
                word_counts[new_word] += count
            else:
                word_counts[new_word] = count
            del word_counts[word]
            
        del pair_counts[most_common_pair]
        del pair_pos[most_common_pair]

    vocab = {i: bytes([i]) for i in range(256)}
    
    # Merges
    for i, merge in enumerate(merges):
        vocab[256 + i] = merge[0] + merge[1]
    
    # Special tokens
    start_special_idx = 256 + len(merges)
    for i, token in enumerate(special_tokens):
        vocab[start_special_idx + i] = token.encode("utf-8")
        
    return vocab, merges