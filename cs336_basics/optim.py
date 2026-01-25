import math
from typing import Callable, Optional

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self,params, lr=1e-3, weight_decay=1e-3, betas=(0.9,0.999), eps=1e-5):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
        super(AdamW,self).__init__(params,defaults)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                m = self.state[p].get('m', torch.zeros_like(p))
                v = self.state[p].get('v', torch.zeros_like(p))
                t = self.state[p].get('t', 0) + 1
                beta1, beta2 = group['betas']
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad * grad)
                alpha_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data -= alpha_t * m / (torch.sqrt(v) + group['eps'])
                p.data -= lr * group['weight_decay'] * p.data
                self.state[p]['m'] = m
                self.state[p]['v'] = v
                self.state[p]['t'] = t
        return loss
    