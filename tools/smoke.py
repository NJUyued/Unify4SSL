#!/usr/bin/env python
"""端到端冒烟测试：每个方法在 CIFAR-10 上跑若干迭代。

用法：
    python tools/smoke.py [--iters 20] [--data-dir /tmp/sslunify_data_check]
                          [--synthetic]

验证点：
1. 五个单数据集方法（fixmatch/prg/mutexmatch/rda/soc*）+ USP 各自端到端跑通
2. 损失为有限值、评估返回 top-1
3. MNAR 协议（prg mismatch）与失配协议（rda mismatch）可构造

* SoC 需要 Semi-Aves/Semi-Fungi 数据集，冒烟用 CIFAR-10 替代验证方法逻辑
  （alpha/num_tracked_batch 按小配置调小）。

网络不可用时的兜底：--synthetic 或自动回退（真实 CIFAR 下载失败时）使用
随机噪声合成的 CIFAR pickle（格式与 torchvision 期望一致，绕过 md5 完整性
检查）。合成数据只验证训练机制的连通性，指标数值无意义。
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from tools.train import COMMON_DEFAULTS, METHOD_DEFAULTS


# --------------------------------------------------------------- 合成 CIFAR 兜底

def _install_synthetic_cifar(data_dir: str, name: str = "cifar10", per_class: int = 60,
                             seed: int = 0) -> None:
    """写随机噪声合成的 CIFAR pickle（torchvision 原生目录布局）。"""
    rng = np.random.RandomState(seed)
    num_classes = 10 if name == "cifar10" else 100

    def make(n_per_class):
        labels = np.repeat(np.arange(num_classes), n_per_class)
        data = rng.randint(0, 256, size=(len(labels), 3 * 32 * 32), dtype=np.uint8)
        return data, labels

    if name == "cifar10":
        base = os.path.join(data_dir, "cifar-10-batches-py")
        os.makedirs(base, exist_ok=True)
        data, labels = make(per_class)
        for i, idx in enumerate(np.array_split(np.arange(len(labels)), 5), start=1):
            with open(os.path.join(base, f"data_batch_{i}"), "wb") as f:
                pickle.dump({"data": data[idx], "labels": labels[idx].tolist()}, f)
        data_t, labels_t = make(max(per_class // 2, 10))
        with open(os.path.join(base, "test_batch"), "wb") as f:
            pickle.dump({"data": data_t, "labels": labels_t.tolist()}, f)
        with open(os.path.join(base, "batches.meta"), "wb") as f:
            pickle.dump({"label_names": [f"class_{i}" for i in range(num_classes)]}, f)
    else:
        base = os.path.join(data_dir, "cifar-100-python")
        os.makedirs(base, exist_ok=True)
        data, labels = make(per_class)
        with open(os.path.join(base, "train"), "wb") as f:
            pickle.dump({"data": data, "fine_labels": labels.tolist()}, f)
        data_t, labels_t = make(max(per_class // 2, 10))
        with open(os.path.join(base, "test"), "wb") as f:
            pickle.dump({"data": data_t, "fine_labels": labels_t.tolist()}, f)
        with open(os.path.join(base, "meta"), "wb") as f:
            pickle.dump({"fine_label_names": [f"class_{i}" for i in range(num_classes)]}, f)


def _patch_cifar_offline() -> None:
    """绕过 torchvision CIFAR 的 md5 完整性检查与下载（合成数据用）。"""
    import torchvision.datasets as tvd

    if getattr(tvd.CIFAR10, "_sslunify_offline_patched", False):
        return
    tvd.CIFAR10._check_integrity = lambda self: True
    tvd.CIFAR10.download = lambda self: None

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    tvd.CIFAR10._load_meta = _load_meta
    tvd.CIFAR10._sslunify_offline_patched = True


def ensure_cifar(data_dir: str, name: str, synthetic: bool = False) -> bool:
    """确保 data_dir 下有可用的 <name> 数据。

    返回 True 表示启用了合成数据兜底。本地已有真实数据时无需网络。
    """
    import torchvision.datasets as tvd

    cls = getattr(tvd, name.upper())
    if not synthetic:
        try:
            cls(data_dir, train=True, download=True)
            cls(data_dir, train=False, download=True)
            return False
        except Exception as e:
            print(f"[smoke] real {name} unavailable ({type(e).__name__}: {e}); "
                  f"falling back to synthetic data")
    _install_synthetic_cifar(data_dir, name)
    _patch_cifar_offline()
    print(f"[smoke] WARNING: using SYNTHETIC {name} — metrics are meaningless, "
          f"connectivity/pipeline checks only")
    return True



def make_cfg(method: str, data_dir: str, iters: int, **overrides) -> types.SimpleNamespace:
    values = dict(COMMON_DEFAULTS)
    values.update(METHOD_DEFAULTS.get(method, {}))
    values.update(
        dict(
            dataset="cifar10",
            data_dir=data_dir,
            num_labels=40,
            batch_size=4,
            uratio=2,
            eval_batch_size=64,
            num_train_iter=iters,
            num_eval_iter=max(iters, 1),
            num_workers=0,
            save_dir="/tmp/sslunify_smoke",
            seed=1,
        )
    )
    values.update(overrides)
    values["method"] = method
    values["num_classes"] = overrides.get("num_classes", 10)
    if values.get("save_name") is None:
        values["save_name"] = f"smoke_{method}"
    values["tb_dir"] = os.path.join(values["save_dir"], values["save_name"], "tensorboard")
    return types.SimpleNamespace(**values)


def run_single_task(method: str, data_dir: str, iters: int, device, **overrides) -> dict:
    from sslunify.core import SSLTrainer, get_cosine_schedule_with_warmup, get_optimizer
    from sslunify.data import get_data_loaders, get_ssl_dsets
    from sslunify.methods import TrainContext, build_method
    from sslunify.utils.common import net_builder, set_seed

    cfg = make_cfg(method, data_dir, iters, **overrides)
    set_seed(cfg.seed)
    cfg.device = str(device)
    cfg.net_builder = net_builder(
        cfg.net,
        from_name=False,
        net_conf={
            "depth": cfg.depth,
            "widen_factor": cfg.widen_factor,
            "leaky_slope": cfg.leaky_slope,
            "dropout": cfg.dropout,
        },
    )

    lb, ulb, ev = get_ssl_dsets(cfg)
    loaders = get_data_loaders(cfg, lb, ulb, ev)

    ctx = TrainContext(cfg, print_fn=print)
    ctx.device = device
    m = build_method(method, ctx)
    ctx.optimizer = get_optimizer(m.optimizer_modules(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    ctx.scheduler = get_cosine_schedule_with_warmup(ctx.optimizer, cfg.num_train_iter)

    trainer = SSLTrainer(m, cfg, loaders, tb_log=None, logger=None)
    result = trainer.train()
    finite = all(
        math.isfinite(float(v)) for v in result.values() if isinstance(v, (int, float)) and not isinstance(v, bool)
    )
    assert finite, f"{method}: non-finite metrics {result}"
    acc = float(result["eval/top-1-acc"])
    print(f"[SMOKE OK] {method}: top-1={acc:.4f}")
    return {"method": method, "top1": acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--data-dir", type=str, default="/tmp/sslunify_data_check")
    ap.add_argument("--methods", type=str, default="fixmatch,prg,mutexmatch,rda,soc")
    ap.add_argument("--usp", action="store_true", help="run USP continual-learning smoke (slow)")
    ap.add_argument("--synthetic", action="store_true",
                    help="force synthetic CIFAR data (no download; metrics meaningless)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("/tmp/sslunify_smoke", exist_ok=True)
    ensure_cifar(args.data_dir, "cifar10", synthetic=args.synthetic)
    if args.usp:
        ensure_cifar(args.data_dir, "cifar100", synthetic=args.synthetic)
    results = []

    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        overrides = {}
        if method == "soc":  # 小配置：CIFAR-10 代替 Semi-Aves 验证方法逻辑
            overrides = dict(alpha=2.5, num_tracked_batch=32, num_eval_iter=args.iters)
        results.append(run_single_task(method, args.data_dir, args.iters, device, **overrides))

    # MNAR / mismatch 协议构造检查
    from sslunify.data import get_ssl_dsets

    for protocol in ("prg", "cadr", "rda"):
        cfg = make_cfg("prg" if protocol != "rda" else "rda", args.data_dir, args.iters,
                       mismatch=protocol, n0=10 if protocol != "cadr" else 0,
                       gamma=20 if protocol == "cadr" else 0)
        lb, ulb, ev = get_ssl_dsets(cfg)
        print(f"[SMOKE OK] protocol {protocol}: lb={len(lb)}, ulb={len(ulb)}, eval={len(ev)}")

    if args.usp:
        from sslunify.cl import SessionTrainer

        cfg = make_cfg("usp", args.data_dir, iters=1,
                       dataset="cifar100", net="resnet32", num_classes=100,
                       nb_cl=50, nb_cl_fg=50, k_shot=5, u_ratio=1,
                       epochs=1, epochs_new=1, warmup_epochs=0,
                       proto_dim=64, nb_protos=5, u_iter=2, buffer_size=100,
                       batch_size=16, eval_batch_size=64)
        cfg.device = str(device)
        trainer = SessionTrainer(cfg)
        result = trainer.train()
        print(f"[SMOKE OK] usp: {result}")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
