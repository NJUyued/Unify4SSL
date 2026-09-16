"""Common utilities shared across the unified framework.

net_builder / Get_Scalar / logger 等，语义与原始各仓库 utils.py 保持一致。
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch


class Get_Scalar:
    """Constant scalar schedule（原代码中 T / p_cutoff 的调度器形式）。"""

    def __init__(self, value):
        self.value = value

    def get_value(self, it=None):
        return self.value

    def __call__(self, it=None):
        return self.value


def get_logger(name="sslunify", save_path=None, level="INFO"):
    """与原始 get_logger 语义一致：无 logger 时返回 print 兼容对象。"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fh = logging.FileHandler(os.path.join(save_path, f"{name}.log"))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setattr_cls_from_kwargs(cls, kwargs):
    for key in kwargs.keys():
        value = kwargs[key]
        setattr(cls, key, value)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def expmo(momentum, n):
    """EMA 有效窗口长度（仅用于日志）。"""
    return 1.0 / (1.0 - momentum) if momentum < 1.0 else float("inf")


def net_builder(net_name, from_name: bool, net_conf=None, **extra):
    """统一骨干构建入口。

    与原始各仓库的 net_builder 相比：
    - 统一了 wrn / resnet18 / resnet50 / resnet101 / cnn13 / preresnet / resnet32
      （USP 的 CIFAR 持续学习骨干）的注册；
    - 所有骨干遵循统一 forward 约定：
        forward(x)                -> logits
        forward(x, ood_test=True) -> (logits, feature)   # MutexMatch / RDA 双头所需
        output_num()              -> 特征维度
    """
    from sslunify import nets

    net_name = net_name.lower()
    builder = nets.get_builder(net_name)
    net_conf = net_conf or {}

    def build(num_classes, **overrides):
        kwargs = {**net_conf, **overrides}
        return builder(num_classes=num_classes, **kwargs)

    return build
