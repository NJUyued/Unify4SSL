"""Optimizer construction (SGD, no weight decay on BN)."""

from __future__ import annotations

import torch


def get_optimizer(net, optim="sgd", lr=0.03, momentum=0.9, weight_decay=5e-4, nesterov=False, bn_wd_skip=True):
    """与原始 get_SGD 一致：bn_wd_skip=True 时 BN 参数的 weight_decay 置 0。

    net 可为单个 nn.Module、模块列表或参数迭代器（迭代器时退化为整体 weight decay）。
    """
    assert optim in ("sgd", "adam", "adamw")
    if not isinstance(net, torch.nn.Module) and not (
        isinstance(net, (list, tuple)) and net and isinstance(net[0], torch.nn.Module)
    ):
        params = net if not isinstance(net, (list, tuple)) else list(net)
        if optim == "sgd":
            return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        if optim == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    modules = [net] if isinstance(net, torch.nn.Module) else list(net)

    def _get_params():
        if bn_wd_skip:
            decay, no_decay = [], []
            seen = set()
            for m in modules:
                for name, param in m.named_parameters():
                    if id(param) in seen or not param.requires_grad:
                        continue
                    seen.add(id(param))
                    if ("bn" in name or "norm" in name or "downsample.1" in name) and param.requires_grad:
                        no_decay.append(param)
                    else:
                        decay.append(param)
            return decay, no_decay
        seen = set()
        params = []
        for m in modules:
            for p in m.parameters():
                if id(p) not in seen and p.requires_grad:
                    seen.add(id(p))
                    params.append(p)
        return params, []

    decay, no_decay = _get_params()
    groups = [{"params": decay, "weight_decay": weight_decay}]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if optim == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=momentum, nesterov=nesterov)
    if optim == "adam":
        return torch.optim.Adam(groups, lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)
