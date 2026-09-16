"""互补标签分类头（ReverseCLS）。

MutexMatch（TNNLS'24）与 RDA（ECCV'22）共用的反向头：Linear + Softmax。
保持原实现的 forward 返回 [x, logits, probs] 列表约定（三段解包）。

两个方法的互补标签来源不同：
- MutexMatch : argmin(softmax(logits))   —— 主头最不认为是的类
- RDA        : 从非真类中随机采样        —— 有标签数据上构造
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ReverseCLS(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.main = nn.Sequential(self.fc, nn.Softmax(dim=-1))

    def forward(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out


def argmin_complementary(logits: torch.Tensor) -> torch.Tensor:
    """MutexMatch 的互补标签：每个样本主头最不可能是的类。"""
    return torch.min(torch.softmax(logits.detach(), dim=-1), dim=-1)[1]


def random_complementary(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """RDA 的互补标签：有标签数据的非真类中均匀随机采样。"""
    res = []
    all_possible = set(range(num_classes))
    for yi in y.tolist():
        res.append(int(np.random.choice(list(all_possible - {yi}))))
    return torch.tensor(res, dtype=y.dtype, device=y.device)
