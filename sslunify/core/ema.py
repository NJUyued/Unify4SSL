"""EMA（评估模型动量更新）。

原来每个方法类里都手写一份 _eval_model_update（5 处重复），统一到这里。
"""

from __future__ import annotations

import torch


@torch.no_grad()
def init_eval_model(train_model, eval_model):
    for param_q, param_k in zip(train_model.parameters(), eval_model.parameters()):
        param_k.data.copy_(param_q.detach().data)
        param_k.requires_grad = False
    eval_model.eval()


@torch.no_grad()
def eval_model_update(train_model, eval_model, ema_m=0.999):
    """p_eval = p_eval * m + p_train * (1 - m)；buffers 直接拷贝。"""
    for param_train, param_eval in zip(train_model.parameters(), eval_model.parameters()):
        param_eval.copy_(param_eval * ema_m + param_train.detach() * (1 - ema_m))
    for buffer_train, buffer_eval in zip(train_model.buffers(), eval_model.buffers()):
        buffer_eval.copy_(buffer_train)
