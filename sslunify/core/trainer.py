"""统一训练器：收敛 5 个仓库中重复的训练循环样板。

一个 trainer 驱动所有方法（FixMatch / PRG / MutexMatch / RDA / SoC；
USP 的持续学习循环在 cl/session.py，复用本模块的评估与存档逻辑）。

循环语义与原实现保持一致：
- zip(lb_loader, ulb_loader) 双流迭代，ulb loader 为无限迭代器（num_iters 预采样）
- 单次前向：cat(x_lb, x_ulb_w, x_ulb_s)（由方法在 train_step 内部完成）
- backward → optimizer.step → scheduler.step → zero_grad → EMA 更新
- 每 num_eval_iter 次评估，追踪 best，2^19 后评估间隔降为 1000
"""

from __future__ import annotations

import contextlib
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score

from ..core.ema import eval_model_update
from ..core.meters import GM
from ..utils.common import Get_Scalar

try:
    from torch.amp import GradScaler, autocast
    _AMP_KW = {"device_type": "cuda"}
except ImportError:  # torch < 2.0
    from torch.cuda.amp import GradScaler, autocast
    _AMP_KW = {}


class SSLTrainer:
    def __init__(self, method, cfg, loaders, tb_log=None, logger=None):
        self.method = method
        self.cfg = cfg
        self.loaders = loaders
        self.tb_log = tb_log
        self.print_fn = print if logger is None else logger.info
        self.it = 0
        self.best_eval_acc, self.best_it = 0.0, 0
        self.t_fn = Get_Scalar(cfg.T)
        self.p_fn = Get_Scalar(cfg.p_cutoff if getattr(cfg, "p_cutoff", None) is not None else 1.0)

    # ---------------------------------------------------------------- train

    def train(self):
        cfg = self.cfg
        method = self.method
        ctx = method.ctx
        ngpus_per_node = torch.cuda.device_count()
        method.train_model.train()

        scaler = GradScaler(enabled=cfg.amp)
        amp_cm = (lambda: autocast(**_AMP_KW)) if cfg.amp else contextlib.nullcontext

        best_eval_acc, best_it = 0.0, 0

        for (x_lb, y_lb), (x_ulb_w, x_ulb_s, y_ulb, idx) in zip(
            self.loaders["train_lb"], self.loaders["train_ulb"]
        ):
            if self.it > cfg.num_train_iter:
                break

            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]
            assert num_ulb == x_ulb_s.shape[0]

            batch = {
                "x_lb": x_lb.to(ctx.device, non_blocking=True),
                "y_lb": y_lb.long().to(ctx.device, non_blocking=True),
                "x_ulb_w": x_ulb_w.to(ctx.device, non_blocking=True),
                "x_ulb_s": x_ulb_s.to(ctx.device, non_blocking=True),
                "y_ulb": y_ulb.to(ctx.device, non_blocking=True),
                "idx": idx,
                "num_lb": num_lb,
                "num_ulb": num_ulb,
                "T": self.t_fn(self.it),
                "p_cutoff": self.p_fn(self.it),
            }

            with amp_cm():
                total_loss, tb_dict = method.train_step(ctx, batch)

            if cfg.amp:
                scaler.scale(total_loss).backward()
                scaler.step(ctx.optimizer)
                scaler.update()
            else:
                total_loss.backward()
                ctx.optimizer.step()

            ctx.scheduler.step()
            method.train_model.zero_grad()

            with torch.no_grad():
                eval_model_update(method.train_model, method.eval_model, cfg.ema_m)

            tb_dict["lr"] = ctx.optimizer.param_groups[0]["lr"]

            if self.it % cfg.num_eval_iter == 0:
                eval_dict = self.evaluate()
                tb_dict.update(eval_dict)
                save_path = os.path.join(cfg.save_dir, cfg.save_name)
                if tb_dict.get("eval/top-1-acc", 0.0) > best_eval_acc:
                    best_eval_acc = tb_dict["eval/top-1-acc"]
                    best_it = self.it
                self.print_fn(
                    f"{self.it} iteration, USE_EMA: True, {tb_dict}, BEST_EVAL_ACC: {best_eval_acc}, at {best_it} iters"
                )

            if not getattr(cfg, "multiprocessing_distributed", False) or (
                cfg.multiprocessing_distributed and cfg.rank % max(ngpus_per_node, 1) == 0
            ):
                if self.it == best_it:
                    self.save_model("model_best.pth", os.path.join(cfg.save_dir, cfg.save_name))
                if self.tb_log is not None:
                    self.tb_log.update(tb_dict, self.it)

            self.it += 1
            ctx.it = self.it
            if self.it > 2**19:
                cfg.num_eval_iter = 1000

        eval_dict = self.evaluate()
        eval_dict.update({"eval/best_acc": best_eval_acc, "eval/best_it": best_it})
        return eval_dict

    # ---------------------------------------------------------------- eval

    @torch.no_grad()
    def evaluate(self, loader_key="eval"):
        cfg = self.cfg
        method = self.method
        eval_model = method.eval_model
        eval_model.eval()

        eval_loader = self.loaders[loader_key]
        device = method.ctx.device

        total_loss, total_acc, total_num = 0.0, 0.0, 0.0
        total_top5 = 0.0
        y_true, y_pred = [], []
        topk = 5 if self.num_classes_for_eval() >= 5 else 1

        for x, y in eval_loader:
            y = y.long()
            x, y = x.to(device), y.to(device)
            num_batch = x.shape[0]
            total_num += num_batch
            logits = method.eval_forward(method.ctx, eval_model, x)
            max_idx = torch.max(logits, dim=-1)[1]
            if topk > 1:
                top5_hit = logits.topk(topk, dim=1)[1].eq(y.view(-1, 1)).any(dim=1)
                total_top5 += top5_hit.float().sum()
            y_true.extend(y.cpu().tolist())
            y_pred.extend(max_idx.cpu().tolist())
            loss = F.cross_entropy(logits, y, reduction="mean")
            acc = torch.sum(max_idx == y)
            total_loss += loss.detach() * num_batch
            total_acc += acc.detach()

        metrics = {
            "eval/loss": total_loss / total_num,
            "eval/top-1-acc": total_acc / total_num,
            "eval/precision": precision_score(y_true, y_pred, average="macro", zero_division=1),
            "eval/recall": recall_score(y_true, y_pred, average="macro"),
            "eval/GM": GM(y_pred, y_true),
        }
        if topk > 1:
            metrics["eval/top-5-acc"] = total_top5 / total_num
        metrics.update(method.extra_metrics(method.ctx, y_true, y_pred))
        return metrics

    def num_classes_for_eval(self):
        return self.method.num_classes

    # ---------------------------------------------------------------- ckpt

    def save_model(self, save_name, save_path):
        os.makedirs(save_path, exist_ok=True)
        cfg = self.cfg
        method = self.method
        save_filename = os.path.join(save_path, save_name)
        train_model = method.train_model.module if hasattr(method.train_model, "module") else method.train_model
        eval_model = method.eval_model.module if hasattr(method.eval_model, "module") else method.eval_model
        torch.save(
            {
                "train_model": train_model.state_dict(),
                "eval_model": eval_model.state_dict(),
                "optimizer": method.ctx.optimizer.state_dict(),
                "scheduler": method.ctx.scheduler.state_dict(),
                "it": self.it,
                "method_state": method.method_state_dict(method.ctx),
            },
            save_filename,
        )
        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):
        checkpoint = torch.load(load_path, map_location="cpu")
        method = self.method
        train_model = method.train_model.module if hasattr(method.train_model, "module") else method.train_model
        eval_model = method.eval_model.module if hasattr(method.eval_model, "module") else method.eval_model
        train_model.load_state_dict(checkpoint["train_model"], strict=False)
        eval_model.load_state_dict(checkpoint["eval_model"], strict=False)
        if "optimizer" in checkpoint and getattr(method.ctx, "optimizer", None) is not None:
            method.ctx.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint and getattr(method.ctx, "scheduler", None) is not None:
            method.ctx.scheduler.load_state_dict(checkpoint["scheduler"])
        if "it" in checkpoint:
            self.it = checkpoint["it"]
            method.ctx.it = self.it
        if "method_state" in checkpoint:
            method.load_method_state(method.ctx, checkpoint["method_state"])
        self.print_fn(f"checkpoint loaded: {load_path}")
