"""Meters, logging, metrics（AverageMeter / TBLog / accuracy / GM）。"""

from __future__ import annotations

import os

import numpy as np
import torch


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TBLog:
    """TensorBoard logger wrapper（与原始实现一致：dict 直接 update 到 writer）。"""

    def __init__(self, tb_dir, file_name):
        self.tb_dir = tb_dir
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(os.path.join(tb_dir, file_name))
        except Exception:
            self.writer = None

    def update(self, tb_dict, it, suffix=None):
        if self.writer is None:
            return
        if suffix is None:
            suffix = ""
        for key, value in tb_dict.items():
            v = value.item() if isinstance(value, torch.Tensor) and value.numel() == 1 else value
            self.writer.add_scalar(suffix + key, v, it)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def GM(y_pred, y_true):
    """Geometric mean of per-class sensitivity（类不均衡场景指标，PRG 引入）。"""
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    classes = np.unique(y_true)
    sensitivities = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        if tp + fn == 0:
            continue
        sensitivities.append(tp / (tp + fn))
    if not sensitivities:
        return 0.0
    prod = np.prod(sensitivities)
    return float(np.sign(prod) * np.abs(prod) ** (1.0 / len(sensitivities)))
