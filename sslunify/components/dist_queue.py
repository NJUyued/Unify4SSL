"""分布追踪队列：N 个 batch 的预测分布均值滑窗。

原实现分散在三处，同一数据结构不同用途：
- RDA（ECCV'22）  : distri / distri_reverse 两个 128 槽队列，驱动互惠分布对齐
- PRG（ICCV'23）  : CTT 内嵌的 distri（已并入 ClassTransitionTracker）
- USP（ICCV'25）  : DistAlign 的对齐分布队列（USB 式实现）
"""

from __future__ import annotations

import torch


class DistQueue:
    def __init__(self, num_classes: int, window: int = 128, init: str = "ones", device="cpu"):
        self.num_classes = num_classes
        self.window = window
        fill = 1.0 if init == "ones" else 0.0
        self.queue = torch.full((window, num_classes), fill)
        self.slot = 0
        self.device = device

    def to(self, device):
        self.device = device
        self.queue = self.queue.to(device)
        return self

    @torch.no_grad()
    def update(self, probs_mean: torch.Tensor):
        """probs_mean: 本 batch softmax 概率均值（C 维，已 detach）。"""
        self.queue[self.slot] = probs_mean
        self.slot = (self.slot + 1) % self.window

    @torch.no_grad()
    def mean(self) -> torch.Tensor:
        return self.queue.mean(dim=0)

    @property
    def distri(self) -> torch.Tensor:
        """兼容 RDA 原实现的字段名。"""
        return self.queue

    def state_dict(self):
        return {"queue": self.queue.cpu(), "slot": self.slot}

    def load_state_dict(self, state):
        self.queue = state["queue"].to(self.device)
        self.slot = state["slot"]
