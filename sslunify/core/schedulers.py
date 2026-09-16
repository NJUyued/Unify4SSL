"""Cosine LR schedule with linear warmup（与原始仓库 7/16 warmup 语义一致）。"""

from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(optimizer, num_training_steps, num_cycles=7.0 / 16.0, num_warmup_steps=0, last_epoch=-1):
    """num_warmup_steps=0 时：前 1/16·(2π/7) 段为线性 warmup（cos 函数平移后的上升段）。"""

    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)
