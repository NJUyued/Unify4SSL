"""Loss utilities.

ce_loss / consistency_loss 语义与原始仓库 train_utils.py 完全一致：
- ce_loss 支持 hard / soft 标签两种模式（soft 即 KL 形式的手动实现）
- consistency_loss 是 FixMatch 式阈值一致性损失，作为各方法的公共基元
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ce_loss(logits, targets, use_hard_labels=True, reduction="none"):
    """CE loss，支持 hard（int 索引）与 soft（概率分布）标签。

    与原始实现一致：soft 分支为手动 log_softmax + nll（无 KL 的 target_entropy 项）。
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits.float(), dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits.float(), dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == "none":
            return nll_loss
        if reduction == "mean":
            return torch.mean(nll_loss)
        if reduction == "sum":
            return torch.sum(nll_loss)
        raise ValueError(f"invalid reduction: {reduction}")


def consistency_loss(logits_s, targets, mask, use_hard_labels=True):
    """FixMatch 式 masked consistency：对强视图按伪标签（hard/soft）+ mask 计算 CE。"""
    if mask.dim() == targets.dim():
        mask = mask.view(-1)
    loss = ce_loss(logits_s, targets, use_hard_labels=use_hard_labels, reduction="none")
    return (loss * mask.float()).mean()


def entropy_loss(x):
    """预测熵（原 RDA/Mutex 仓库 utils 中的同名函数）。"""
    return -(torch.log(x.softmax(dim=-1) + 1e-5).softmax(dim=-1) * torch.log(x.softmax(dim=-1) + 1e-5)).mean()
