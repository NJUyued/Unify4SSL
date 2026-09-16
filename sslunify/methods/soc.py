"""SoC（AAAI 2024）。

《Roll with the Punches: Expansion and Shrinkage of Soft Label Selection
for Semi-Supervised Fine-Grained Visual Classification》
置信度感知的类子集软标签选择：对类别转移亲和矩阵做多粒度 k-means，
样本置信度决定粒度——低置信用粗粒度簇（选择集扩张，软标签更平），
高置信用细粒度簇（收缩，趋近硬标签）。无阈值，全量无标签样本参与。

组件复用：ClassTransitionTracker（与 PRG 共享）、kmeans。
原文默认：alpha=2.5（Semi-Aves）/ 4（Semi-Fungi）、num_tracked_batch=5120。
"""

from __future__ import annotations

import numpy as np
import torch

from ..components import ClassTransitionTracker, kmeans
from ..core.losses import ce_loss
from .base import SSLMethod


class SoC(SSLMethod):
    name = "soc"

    def setup(self, ctx):
        cfg = self.cfg
        self.num_granularity = round(self.num_classes / cfg.alpha) + 1  # C
        self.ctt = ClassTransitionTracker(self.num_classes, window=cfg.num_tracked_batch, device=ctx.device)
        self.label_dics = [{} for _ in range(self.num_granularity)]
        self.clusters = [[] for _ in range(self.num_granularity)]
        # 每个粒度级的初始中心：随机抽 i+2 个类（与原实现一致）
        self.centroids = [
            np.random.choice([i for i in range(self.num_classes)], i + 2, False).tolist()
            for i in range(self.num_granularity)
        ]
        self.c_flag = True

    def train_step(self, ctx, batch):
        cfg = self.cfg
        model = self.train_model
        x_lb, y_lb = batch["x_lb"], batch["y_lb"]
        x_ulb_w, x_ulb_s = batch["x_ulb_w"], batch["x_ulb_s"]
        idx = batch["idx"]
        num_lb = batch["num_lb"]

        inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
        logits, _ = model(inputs, ood_test=True)
        logits_x_lb = logits[:num_lb]
        logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
        pseudo_label = torch.softmax(logits_x_ulb_w, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)

        # CTT 更新（SoC 不追踪 distri，仅转移矩阵；由 ClassTransitionTracker 统一承担）
        self.ctt.update(idx, max_idx, pseudo_label.detach().mean(0))

        # k-means：首次评估点全粒度重建一次，此后每迭代轮转重聚一个粒度
        affinity = self.ctt.affinity().cpu().numpy()
        if ctx.it % cfg.num_eval_iter == 0 and self.c_flag:
            self.c_flag = False
            for i in range(self.num_granularity):
                self.label_dics[i], self.clusters[i], self.centroids[i] = kmeans(affinity, i + 2, self.centroids[i])
        c_count = ctx.it % self.num_granularity
        self.label_dics[c_count], self.clusters[c_count], self.centroids[c_count] = kmeans(
            affinity, c_count + 2, self.centroids[c_count]
        )
        del logits

        sup_loss = ce_loss(logits_x_lb, y_lb, reduction="mean")
        cos_loss = consistency_loss_soc(
            logits_x_ulb_w, logits_x_ulb_s, self.label_dics, self.clusters, cfg.alpha, self.num_classes
        )
        total_loss = sup_loss + cfg.ulb_loss_ratio * cos_loss

        tb_dict = {
            "train/sup_loss": sup_loss.detach(),
            "train/cos_loss": cos_loss.detach(),
            "train/total_loss": total_loss.detach(),
        }
        return total_loss, tb_dict

    def method_state_dict(self, ctx):
        return {"ctt": self.ctt.state_dict(), "centroids": self.centroids}

    def load_method_state(self, ctx, state):
        if "ctt" in state:
            self.ctt.load_state_dict(state["ctt"])
        if "centroids" in state:
            self.centroids = state["centroids"]


def _normalize(x):
    return (x / torch.sum(x)).detach()


def consistency_loss_soc(logits_w, logits_s, label_dics, clusters, alpha, num_classes):
    """SoC 的软标签选择一致性损失（与原 soc_utils.py 数学一致）。"""
    logits_w = logits_w.detach()
    pseudo_label = torch.softmax(logits_w, dim=-1)
    _, max_idx = torch.max(pseudo_label, dim=-1)

    num_cluster = round(num_classes / alpha)
    filter_value = float(0)
    p_temp = pseudo_label
    for i, p in enumerate(pseudo_label):
        max_probs_p, _ = torch.max(p, dim=-1)
        conf_idx = round(max_probs_p.cpu().item() * num_cluster)
        indices_to_remain = clusters[conf_idx][label_dics[conf_idx][max_idx[i].cpu().item()]]
        indices_to_remove = list(set(range(num_classes)) - set(indices_to_remain))
        p[indices_to_remove] = filter_value
        p = _normalize(p)
        p_temp[i] = p
    pseudo_label = p_temp
    loss_super = ce_loss(logits_s, pseudo_label, use_hard_labels=False, reduction="none")
    return loss_super.mean()
