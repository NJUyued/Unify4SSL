"""方法注册表：name -> SSLMethod 子类。

用法：
    from sslunify.methods import build_method
    method = build_method("prg", ctx)
"""

from __future__ import annotations

from .base import SSLMethod, TrainContext
from .fixmatch import FixMatch
from .prg import PRG
from .mutexmatch import MutexMatch
from .rda import RDA
from .soc import SoC

METHOD_REGISTRY = {
    FixMatch.name: FixMatch,
    PRG.name: PRG,
    MutexMatch.name: MutexMatch,
    RDA.name: RDA,
    SoC.name: SoC,
}

SINGLE_TASK_METHODS = set(METHOD_REGISTRY.keys())  # USP 为持续学习范式，单独构造


def build_method(name: str, ctx: TrainContext) -> SSLMethod:
    if name not in METHOD_REGISTRY:
        raise KeyError(f"unknown method: {name}, available: {sorted(METHOD_REGISTRY)}")
    return METHOD_REGISTRY[name](ctx)


__all__ = [
    "SSLMethod",
    "TrainContext",
    "METHOD_REGISTRY",
    "SINGLE_TASK_METHODS",
    "build_method",
    "FixMatch",
    "PRG",
    "MutexMatch",
    "RDA",
    "SoC",
]
