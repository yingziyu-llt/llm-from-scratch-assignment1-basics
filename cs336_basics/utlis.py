import math
from typing import Iterable

import torch
from jaxtyping import Float, Int
from torch import Tensor


def crossEntropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    m,_ = torch.max(inputs,dim=-1,keepdim=True)
    lse = torch.log(torch.sum(torch.exp(inputs - m),dim=-1,keepdim = True)) + m
    target_logits = inputs[torch.arange(targets.shape[0]), targets].unsqueeze(1)
    loss = lse - target_logits
    return loss.mean()

def learning_rate_schedule(t,alpha_max,alpha_min,T_warmup,T_total):
    if t < T_warmup:
        return alpha_max * t / T_warmup
    elif t <= T_total:
        return alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + math.cos(math.pi * (t - T_warmup) / (T_total - T_warmup)))
    else:
        return alpha_min

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    eps = 1e-6
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
    return