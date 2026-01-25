import math
import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import regex as re
import torch
import torch.nn as nn
from cs336_basics.tokenizer import Tokenizer, train_tokenizer
from jaxtyping import Bool, Float, Int
from torch import Tensor


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {'device': device, 'dtype': dtype}
        weight_data = torch.empty((in_features, out_features), **factory_kwargs)
        self.weight = nn.Parameter(weight_data)
        sigma = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(
            self.weight, 
            mean=0.0, 
            std=sigma, 
            a=-3 * sigma, 
            b=3 * sigma
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim)))
        nn.init.trunc_normal_(
            self.weight, 
            mean=0.0, 
            std=1, 
            a=-3, 
            b=3
        )

    def forward(self, input: Int[Tensor, " ..."]) -> Float[Tensor, " ... embedding_dim"]:
        return self.weight[input]

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
        return x * torch.sigmoid(x)
    
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
        self.silu = SiLU()

    def forward(self, in_features: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        x1 = self.w1(in_features)
        x2 = self.w3(in_features)
        return self.w2(self.silu(x1) * x2)

class Softmax(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
        exp_in_features = torch.exp(in_features - torch.max(in_features, dim=self.dim, keepdim=True).values)
        return exp_in_features / torch.sum(exp_in_features, dim=self.dim, keepdim=True)

def softmax(x: Tensor, dim: int = -1) -> Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(dim=dim, keepdim=True)

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d_model))

    def forward(self, in_features: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        in_dtype = in_features.dtype
        in_features = in_features.to(torch.float32)
        RMS = torch.sqrt(1 / self.d_model * torch.sum(in_features ** 2, dim=-1, keepdim=True) + self.eps)
        return (in_features / RMS * self.g).to(in_dtype)
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        powers = torch.arange(0, d_k, 2, device=device).float() / d_k
        inv_freq = 1.0 / (theta ** powers)
        t = torch.arange(max_seq_len, device=device).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[token_positions] 
        sin = self.sin_cached[token_positions]
        
        x_reshaped = x.view(*x.shape[:-1], -1, 2)
        x1 = x_reshaped[..., 0]
        x2 = x_reshaped[..., 1]
        
        out1 = x1 * cos - x2 * sin
        out2 = x2 * cos + x1 * sin
        
        # Combine back and flatten the last two dimensions
        out = torch.stack([out1, out2], dim=-1)
        return out.flatten(-2)

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    score = Q @ K.transpose(-2,-1) / math.sqrt(K.shape[-1])
    if mask is not None:
        score = score.masked_fill(~mask,-torch.inf)
    score = softmax(score)
    return score @ V

class MultiHeadAttention(nn.Module):
    def __init__(self,d_in: int,d_k: int, d_v: int,d_model: int,num_heads: int, use_rope: bool = False, theta: float = 0,max_seq_len: int = 0):
        super().__init__()
        self.q = Linear(d_in,d_k)
        self.k = Linear(d_in,d_k)
        self.v = Linear(d_in,d_v)
        self.o = Linear(d_v,d_model)
        self.num_heads = num_heads
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionalEmbedding(theta,d_k//num_heads,max_seq_len)
    def forward(
        self,
        in_features: Float[Tensor, " ... d_in"],
    ) -> Float[Tensor, " ... d_model"]:
        batch_shape = in_features.shape[:-1]
        Q = self.q(in_features).view(*batch_shape, self.num_heads, -1).transpose(-3,-2)
        K = self.k(in_features).view(*batch_shape, self.num_heads, -1).transpose(-3,-2)
        V = self.v(in_features).view(*batch_shape, self.num_heads, -1).transpose(-3,-2)
        if self.use_rope:
            seq_len = in_features.shape[-2]
            token_positions = torch.arange(seq_len, device=in_features.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        mask = torch.ones((in_features.shape[-2], in_features.shape[-2]), dtype=torch.bool, device=in_features.device).tril()
        attn_output = scaled_dot_product_attention(Q,K,V,mask)
        attn_output = attn_output.transpose(-3,-2).contiguous().view(*batch_shape, -1)
        return self.o(attn_output)
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_heads: int, use_rope: bool = False, theta: float = 0, max_seq_len: int = 0):
        super().__init__()
        self.mha = MultiHeadAttention(d_in=d_model,d_k=d_model,d_v=d_model,d_model=d_model,num_heads=num_heads,use_rope=use_rope,theta=theta,max_seq_len=max_seq_len)
        self.rms1 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model,d_ff)
        self.rms2 = RMSNorm(d_model)

    def forward(self, in_features: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        attn_output = self.mha(self.rms1(in_features))
        out1 = in_features + attn_output
        ffn_output = self.ffn(self.rms2(out1))
        out2 = out1 + ffn_output
        return out2
    
class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, d_ff: int, num_heads: int, num_layers: int, use_rope: bool = False, theta: float = 0, max_seq_len: int = 0):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, d_ff=d_ff, num_heads=num_heads, use_rope=use_rope, theta=theta, max_seq_len=max_seq_len)
            for _ in range(num_layers)
        ])
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
    def forward(self, input_ids: Int[Tensor, " batch seq_len"]) -> Float[Tensor, " batch seq_len vocab_size"]:
        x = self.token_embeddings(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
