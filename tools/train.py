#!/usr/bin/env python
"""统一训练入口。

用法：
    python tools/train.py -c configs/prg/cifar10.yaml                # yaml 配置
    python tools/train.py -c configs/prg/cifar10.yaml --num_labels 40   # 覆盖
    python tools/train.py --method usp -c configs/usp/cifar100.yaml   # 持续学习

配置优先级：CLI 覆盖 > yaml > 方法默认值。
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from sslunify.core import SSLTrainer, TBLog, get_cosine_schedule_with_warmup, get_optimizer
from sslunify.methods import TrainContext, build_method
from sslunify.utils.common import get_logger, net_builder

# ---------------------------------------------------------------- 默认值

COMMON_DEFAULTS = dict(
    # 数据
    dataset="cifar10",
    data_dir="./data",
    num_labels=4000,
    batch_size=64,
    uratio=7,
    eval_batch_size=1024,
    num_workers=4,
    input_size=32,
    # 训练
    epoch=1,
    num_train_iter=2**20,
    num_eval_iter=1000,
    ema_m=0.999,
    ulb_loss_ratio=1.0,
    hard_label=True,
    amp=False,
    lr=0.03,
    momentum=0.9,
    weight_decay=5e-4,
    optim="sgd",
    # 骨干
    net="wrn",
    depth=28,
    widen_factor=2,
    leaky_slope=0.1,
    dropout=0.0,
    # 失配/噪声协议
    mismatch="none",
    n0=0,
    gamma=0.0,
    noisy="none",
    noisy_ratio=0.0,
    fold=0,
    # 杂项
    seed=1,
    save_dir="./saved_models",
    save_name=None,
    resume=False,
    load_path=None,
    tb_dir=None,
)

# 各方法特有默认值（忠实于各原始仓库 argparse）
METHOD_DEFAULTS = {
    "fixmatch": dict(T=1.0, p_cutoff=0.95),
    "prg": dict(T=1.0, p_cutoff=0.95, Nb=128, alpha=1.0, last=False),
    "mutexmatch": dict(T=0.5, p_cutoff=0.95, k=0),
    "rda": dict(T=0.5, p_cutoff=None),  # RDA 无阈值
    "soc": dict(
        T=1.0,
        p_cutoff=None,
        alpha=2.5,
        num_tracked_batch=5120,
        net="resnet50",
        batch_size=32,
        uratio=5,
        lr=0.01,
        input_size=224,
        num_train_iter=400000,
        unlabel="in",
        trainval=True,
        pretrained=False,
    ),
    "usp": dict(
        dataset="cifar100",
        net="resnet32",
        k_shot=30,
        nb_cl_fg=10,
        nb_cl=10,
        u_ratio=7,
        epochs=200,
        epochs_new=200,
        warmup_epochs=10,
        proto_dim=64,
        nb_protos=5,
        p_cutoff=0.95,
        u_iter=100,
        buffer_size=5120,
        use_ulb_aug=True,
        kd_mode="logits",
        use_hard_labels=True,
        use_ulb_kd=True,
        ulb_kd_mode="similarity",
        flip_on_means=True,
        kd_only_old=True,
        use_conloss=True,
        unlabels_predict_mode="cosine",
        lambda_ce=1.0,
        lambda_kd=1.0,
        lambda_con=1.0,
        lambda_cons=1.0,
        lambda_ukd=1.0,
    ),
}


DATASET_NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
    "svhn": 10,
    "stl10": 10,
    "miniimage": 100,
    "tinyimage": 200,
    "semi_aves": 200,
    "semi_fungi": 1394,
}


def build_cfg(method: str, yaml_path: str | None, cli_overrides: dict) -> types.SimpleNamespace:
    import yaml

    values = dict(COMMON_DEFAULTS)
    values.update(METHOD_DEFAULTS.get(method, {}))
    if yaml_path:
        with open(yaml_path) as f:
            values.update(yaml.safe_load(f) or {})
    values.update({k: v for k, v in cli_overrides.items() if v is not None})
    if values.get("num_classes") is None:
        if values["dataset"] not in DATASET_NUM_CLASSES:
            raise ValueError(f"unknown dataset {values['dataset']}, set num_classes explicitly in yaml")
        values["num_classes"] = DATASET_NUM_CLASSES[values["dataset"]]
    if values.get("save_name") is None:
        values["save_name"] = f"{method}_{values['dataset']}"
    if values.get("tb_dir") is None:
        values["tb_dir"] = os.path.join(values["save_dir"], values["save_name"], "tensorboard")
    values["method"] = method
    return types.SimpleNamespace(**values)


def parse_args():
    p = argparse.ArgumentParser("Unify4SSL unified trainer")
    p.add_argument("-c", "--config", type=str, default=None, help="yaml 配置文件")
    p.add_argument("--method", type=str, default=None, help="fixmatch/prg/mutexmatch/rda/soc/usp")
    # 常用 CLI 覆盖项；其余经 yaml 传入
    p.add_argument("--num_labels", type=int, default=None)
    p.add_argument("--num_train_iter", type=int, default=None)
    p.add_argument("--num_eval_iter", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--uratio", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--save_name", type=str, default=None)
    p.add_argument("--mismatch", type=str, default=None)
    p.add_argument("--noisy", type=str, default=None)
    p.add_argument("--gpu", type=int, default=None, help="cuda device id；缺省自动")
    args = p.parse_args()
    method = args.method
    if method is None and args.config:
        method = os.path.basename(os.path.dirname(args.config))
    if method is None:
        p.error("需要 --method 或 -c configs/<method>/xxx.yaml")
    overrides = {k: v for k, v in vars(args).items() if k not in ("config", "method") and v is not None}
    return method, args.config, overrides


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    method, yaml_path, overrides = parse_args()
    cfg = build_cfg(method, yaml_path, overrides)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if overrides.get("gpu") is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{overrides['gpu']}")
    cfg.device = str(device)

    os.makedirs(os.path.join(cfg.save_dir, cfg.save_name), exist_ok=True)
    logger = get_logger(cfg.save_name, os.path.join(cfg.save_dir, cfg.save_name))
    cfg.print_fn = logger.info
    logger.info(f"method={method}, device={device}, cfg={vars(cfg)}")

    cfg.net_builder = net_builder(cfg.net, from_name=False, net_conf={
        "depth": cfg.depth, "widen_factor": cfg.widen_factor,
        "leaky_slope": cfg.leaky_slope, "dropout": cfg.dropout,
    })

    # -------------------------------------------------- USP：持续学习范式
    if method == "usp":
        from sslunify.cl import SessionTrainer

        trainer = SessionTrainer(cfg)
        result = trainer.train()
        logger.info(f"USP finished: {result}")
        return result

    # -------------------------------------------------- 单数据集范式
    from sslunify.data import get_data_loaders, get_ssl_dsets

    lb_dset, ulb_dset, eval_dset = get_ssl_dsets(cfg)
    loaders = get_data_loaders(cfg, lb_dset, ulb_dset, eval_dset)

    ctx = TrainContext(cfg, print_fn=logger.info)
    ctx.device = device
    ssl_method = build_method(method, ctx)

    # 优化器：按方法声明的模块构建（双头方法覆盖 optimizer_modules）
    ctx.optimizer = get_optimizer(ssl_method.optimizer_modules(), optim=cfg.optim, lr=cfg.lr,
                                  momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    ctx.scheduler = get_cosine_schedule_with_warmup(ctx.optimizer, cfg.num_train_iter)

    tb_log = TBLog(cfg.tb_dir, cfg.save_name)
    trainer = SSLTrainer(ssl_method, cfg, loaders, tb_log=tb_log, logger=logger)

    if cfg.resume and cfg.load_path:
        trainer.load_model(cfg.load_path)

    result = trainer.train()
    logger.info(f"training finished: {result}")
    return result


if __name__ == "__main__":
    main()
