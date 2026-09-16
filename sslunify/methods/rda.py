"""RDA: Reciprocal Distribution Alignment（ECCV 2022）。

《RDA: Reciprocal Distribution Alignment for Robust Semi-supervised Learning》
双头（主头 + 互补头）互惠分布对齐：两个头分别维护批均预测分布滑窗，
互以"对方分布的补 / 自身分布"重加权伪标签——FlexMatch 式分布对齐的
双向互惠推广。无置信度阈值，全部无标签样本参与。

组件复用：ReverseCLS（与 MutexMatch 共享）、DistQueue（双队列）。
原文默认：T=0.5、无 p_cutoff、窗口 128（distri 初始化为全 1）。

与 MutexMatch 的关键差异（忠实保留）：
- 互补标签 = 有标签数据非真类随机采样
- reverse_loss 在有标签数据的非 detach 特征上计算
- normalize_d 对整个 batch 张量全局归一化（原实现语义，行和≠1）
"""

from __future__ import annotations

import torch

from ..components import DistQueue, ReverseCLS, random_complementary
from ..core.losses import ce_loss
from .base import SSLMethod
from .mutexmatch import TotalNet


class RDA(SSLMethod):
    name = "rda"

    def build_models(self, ctx):
        build = ctx.cfg.net_builder
        train_model = TotalNet(build, self.num_classes)
        eval_model = TotalNet(build, self.num_classes)
        return train_model, eval_model

    def optimizer_modules(self):
        return [self.train_model.feature_extractor, self.train_model.classifier_reverse]

    def setup(self, ctx):
        # 原实现：distri / distri_reverse 均初始化为全 1，窗口 128
        self.distri = DistQueue(self.num_classes, window=128, init="ones", device=ctx.device)
        self.distri_reverse = DistQueue(self.num_classes, window=128, init="ones", device=ctx.device)

    def train_step(self, ctx, batch):
        cfg = self.cfg
        model = self.train_model
        x_lb, y_lb = batch["x_lb"], batch["y_lb"]
        x_ulb_w, x_ulb_s = batch["x_ulb_w"], batch["x_ulb_s"]
        num_lb = batch["num_lb"]

        inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
        logits, feature = model.feature_extractor(inputs, ood_test=True)
        logits_x_lb = logits[:num_lb]
        logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
        pseudo_label = torch.softmax(logits_x_ulb_w, dim=-1)

        _, logits_reverse, _ = model.classifier_reverse(feature)
        logits_x_ulb_w_reverse, logits_x_ulb_s_reverse = logits_reverse[num_lb:].chunk(2)
        pseudo_label_reverse = torch.softmax(logits_x_ulb_w_reverse, dim=-1)

        # reverse_loss：有标签数据 + 随机互补标签（特征不 detach）
        _, logits_reverse_separate, _ = model.classifier_reverse(feature[:num_lb])
        res_torch = random_complementary(y_lb, self.num_classes)

        self.distri.update(pseudo_label.detach().mean(0))
        self.distri_reverse.update(pseudo_label_reverse.detach().mean(0))
        del logits

        sup_loss = ce_loss(logits_x_lb, y_lb, reduction="mean")
        reverse_loss = ce_loss(logits_reverse_separate, res_torch, reduction="mean")

        unsup_loss_ca, unsup_loss_cd = consistency_loss_rda(
            logits_x_ulb_w_reverse,
            logits_x_ulb_s_reverse,
            logits_x_ulb_w,
            logits_x_ulb_s,
            self.distri.distri,
            self.distri_reverse.distri,
        )

        total_loss = sup_loss + reverse_loss + cfg.ulb_loss_ratio * (unsup_loss_cd + unsup_loss_ca)

        tb_dict = {
            "train/sup_loss": sup_loss.detach(),
            "train/reverse_loss": reverse_loss.detach(),
            "train/unsup_loss_cd": unsup_loss_cd.detach(),
            "train/unsup_loss_ca": unsup_loss_ca.detach(),
            "train/total_loss": total_loss.detach(),
        }
        return total_loss, tb_dict

    def method_state_dict(self, ctx):
        return {"distri": self.distri.state_dict(), "distri_reverse": self.distri_reverse.state_dict()}

    def load_method_state(self, ctx, state):
        if "distri" in state:
            self.distri.load_state_dict(state["distri"])
        if "distri_reverse" in state:
            self.distri_reverse.load_state_dict(state["distri_reverse"])


def _normalize_d(x):
    return (x / torch.sum(x)).detach()


def consistency_loss_rda(logits_x_ulb_w_reverse, logits_x_ulb_s_reverse, logits_w, logits_s, distri, distri_reverse):
    """RDA 的互惠分布对齐一致性损失（与原 rda_utils.py 数学一致）。

    注意：
    - 原实现的 T 参数未参与 softmax（模板残留），此处保持一致；
    - 原实现 normalize_d 对整个 batch 张量求和归一化（行和≠1），此处保持一致。

    p_reverse_da = normalize(p_reverse · mean(1-distri) / mean(distri_reverse))
    p_da         = normalize(p         · mean(1-distri_reverse) / mean(distri))
    loss_cd = CE(logits_s, argmax(p_da))            （主头，硬标签）
    loss_ca = CE(logits_s_reverse, p_reverse_da)     （互补头，软标签）
    """
    logits_w = logits_w.detach()
    logits_x_ulb_w_reverse = logits_x_ulb_w_reverse.detach()
    distri = distri.detach()
    distri_reverse = distri_reverse.detach()

    pseudo_label = torch.softmax(logits_w, dim=-1)
    pseudo_label_reverse = torch.softmax(logits_x_ulb_w_reverse, dim=-1)

    distri_ = _normalize_d(torch.ones_like(distri) - distri)
    pseudo_label_reverse_da = _normalize_d(
        pseudo_label_reverse * (torch.mean(distri_, dim=0) / torch.mean(distri_reverse, dim=0))
    )
    distri_reverse_ = _normalize_d(torch.ones_like(distri_reverse) - distri_reverse)
    pseudo_label_da = _normalize_d(
        pseudo_label * (torch.mean(distri_reverse_, dim=0) / torch.mean(distri, dim=0))
    )
    _, max_idx = torch.max(pseudo_label_da, dim=-1)

    loss_cd = ce_loss(logits_s, max_idx, use_hard_labels=True, reduction="none")
    loss_ca = ce_loss(logits_x_ulb_s_reverse, pseudo_label_reverse_da, use_hard_labels=False, reduction="none")
    return loss_ca.mean(), loss_cd.mean()
