"""MutexMatch（TNNLS 2024）。

《MutexMatch: Semi-Supervised Learning with Mutex-Based Consistency Regularization》
双头（TPC 常规头 + TNC 互补头）互斥一致性：高置信样本走正学习
（FixMatch 式 CE），低置信样本走负学习（互补标签软 CE）——
FixMatch 丢弃的样本被转化为"易学的负监督"。

组件复用：ReverseCLS（与 RDA 共享）。
原文默认：T=0.5、p_cutoff=0.95、k=0（k>0 时对互补伪标签做 top-k 过滤）。

与 RDA 的关键差异（忠实保留）：
- 互补标签 = argmin(softmax(logits))，且在全 batch 上构造
- reverse_loss 在 detach 的特征上计算（全 batch）
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..components import ReverseCLS, argmin_complementary
from ..core.losses import ce_loss
from .base import SSLMethod


class TotalNet(nn.Module):
    """双头封装：骨干（含 TPC 头）+ TNC 互补头。MutexMatch 与 RDA 共用。"""

    def __init__(self, net_builder, num_classes):
        super().__init__()
        self.feature_extractor = net_builder(num_classes=num_classes)
        self.classifier_reverse = ReverseCLS(self.feature_extractor.output_num(), num_classes)

    def forward(self, x):
        return self.feature_extractor(x)


class MutexMatch(SSLMethod):
    name = "mutexmatch"

    def build_models(self, ctx):
        build = ctx.cfg.net_builder
        train_model = TotalNet(build, self.num_classes)
        eval_model = TotalNet(build, self.num_classes)
        return train_model, eval_model

    def optimizer_modules(self):
        return [self.train_model.feature_extractor, self.train_model.classifier_reverse]

    def train_step(self, ctx, batch):
        cfg = self.cfg
        model = self.train_model
        x_lb, y_lb = batch["x_lb"], batch["y_lb"]
        x_ulb_w, x_ulb_s = batch["x_ulb_w"], batch["x_ulb_s"]
        num_lb = batch["num_lb"]
        T, p_cutoff = batch["T"], batch["p_cutoff"]

        inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
        logits, feature = model.feature_extractor(inputs, ood_test=True)
        logits_x_lb = logits[:num_lb]
        logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)

        # TNC 头：全 batch 一次前向（带梯度），ulb 部分供一致性损失
        _, logits_reverse, _ = model.classifier_reverse(feature)
        logits_x_ulb_w_reverse, logits_x_ulb_s_reverse = logits_reverse[num_lb:].chunk(2)

        # reverse_loss 用 detach 特征（只在特征上监督 TNC）
        _, logits_reverse_separate, _ = model.classifier_reverse(feature.detach())

        # 互补标签：主头最不可能是的类（全 batch）
        min_idx_reverse = argmin_complementary(logits)
        del logits

        sup_loss = ce_loss(logits_x_lb, y_lb, reduction="mean")
        reverse_loss = ce_loss(logits_reverse_separate, min_idx_reverse, reduction="mean")

        assert 0 <= cfg.k <= self.num_classes
        unsup_loss, masked_reverse_loss, mask = consistency_loss_mutex(
            logits_x_ulb_w_reverse,
            logits_x_ulb_s_reverse,
            logits_x_ulb_w,
            logits_x_ulb_s,
            cfg.k,
            T=T,
            p_cutoff=p_cutoff,
            use_hard_labels=cfg.hard_label,
        )

        total_loss = (
            sup_loss
            + cfg.ulb_loss_ratio * unsup_loss
            + reverse_loss
            + cfg.ulb_loss_ratio * masked_reverse_loss
        )

        tb_dict = {
            "train/sup_loss": sup_loss.detach(),
            "train/unsup_loss": unsup_loss.detach(),
            "train/reverse_loss": reverse_loss.detach(),
            "train/masked_reverse_loss": masked_reverse_loss.detach(),
            "train/total_loss": total_loss.detach(),
            "train/mask_ratio": 1.0 - mask.detach(),
        }
        return total_loss, tb_dict


def consistency_loss_mutex(logits_x_ulb_w_reverse, logits_x_ulb_s_reverse, logits_w, logits_s, k, T=1.0, p_cutoff=0.0, use_hard_labels=True):
    """MutexMatch 的互斥一致性损失（与原 mutexmatch_utils.py 数学一致）。

    注意：与原实现一致，硬标签路径的伪标签 softmax 不除温度 T（T 仅在
    soft 分支使用；原代码 hard 分支即论文默认配置不含 /T）。
    """
    logits_w = logits_w.detach()
    logits_x_ulb_w_reverse = logits_x_ulb_w_reverse.detach()

    pseudo_label = torch.softmax(logits_w, dim=-1)
    max_probs, max_idx = torch.max(pseudo_label, dim=-1)
    mask = max_probs.ge(p_cutoff).float()
    mask_dis = max_probs.lt(p_cutoff).float()

    pseudo_label_reverse = torch.softmax(logits_x_ulb_w_reverse, dim=-1)

    if k != 0 and k != pseudo_label.size(1):
        # top-k 过滤：互补伪标签与强视图 logits 同时置 0（与原实现一致，原地修改）
        filter_value = float(0)
        indices_to_remove = pseudo_label_reverse < torch.topk(pseudo_label_reverse, k)[0][..., -1, None]
        pseudo_label_reverse[indices_to_remove] = filter_value
        logits_x_ulb_s_reverse[indices_to_remove] = filter_value

    if use_hard_labels:
        masked_loss = ce_loss(logits_s, max_idx, use_hard_labels=True, reduction="none") * mask
        masked_reverse_loss = (
            ce_loss(logits_x_ulb_s_reverse, pseudo_label_reverse, use_hard_labels=False, reduction="none") * mask_dis
        )
    else:
        pseudo_label = torch.softmax(logits_w / T, dim=-1)
        masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels=False, reduction="none") * mask
        masked_reverse_loss = (
            ce_loss(logits_x_ulb_s_reverse, pseudo_label_reverse, use_hard_labels=False, reduction="none") * mask_dis
        )
    return masked_loss.mean(), masked_reverse_loss.mean(), mask.mean()
