"""PRG: Pseudo-Rectifying Guidance（ICCV 2023）。

《Towards Semi-supervised Learning with Non-random Missing Labels》
针对 MNAR（标注/无标注分布失配）：伪标签经类别转移矩阵（Markov 游走）
修正后再过阈值，放大被忽视的稀有类概率。

组件复用：ClassTransitionTracker（与 SoC 共享）。
原文默认：T=1.0、p_cutoff=0.95、Nb=128、alpha=1、last=False（PRG^Last 用 alpha=3）。
"""

from __future__ import annotations

import torch

from ..components import ClassTransitionTracker
from ..core.losses import ce_loss
from .base import SSLMethod


def _normalize_d(x):
    return (x / torch.sum(x)).detach()


class PRG(SSLMethod):
    name = "prg"

    def setup(self, ctx):
        self.ctt = ClassTransitionTracker(self.num_classes, window=getattr(self.cfg, "Nb", 128), device=ctx.device)

    def train_step(self, ctx, batch):
        cfg = self.cfg
        x_lb, y_lb = batch["x_lb"], batch["y_lb"]
        x_ulb_w, x_ulb_s = batch["x_ulb_w"], batch["x_ulb_s"]
        idx = batch["idx"]
        num_lb = batch["num_lb"]
        T, p_cutoff = batch["T"], batch["p_cutoff"]

        inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
        logits = self.train_model(inputs)
        logits_x_lb = logits[:num_lb]
        logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
        del logits

        pseudo_label = torch.softmax(logits_x_ulb_w, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)

        # PRG^cur：损失前更新 CTT；PRG^last：损失后更新（用上一时刻状态修正）
        if not cfg.last:
            self._update_ctt(idx, max_idx, pseudo_label)

        sup_loss = ce_loss(logits_x_lb, y_lb, reduction="mean")

        H_prime = self.ctt.H_prime(alpha=cfg.alpha)
        unsup_loss, mask = consistency_loss_prg(
            logits_x_ulb_w,
            logits_x_ulb_s,
            H_prime,
            self.ctt.label_bank,
            idx,
            last=cfg.last,
            T=T,
            p_cutoff=p_cutoff,
            use_hard_labels=cfg.hard_label,
        )

        total_loss = sup_loss + cfg.ulb_loss_ratio * unsup_loss

        if cfg.last:
            pseudo_label = torch.softmax(logits_x_ulb_w, dim=-1)
            max_probs, max_idx = torch.max(pseudo_label, dim=-1)
            self._update_ctt(idx, max_idx, pseudo_label)

        tb_dict = {
            "train/sup_loss": sup_loss.detach(),
            "train/unsup_loss": unsup_loss.detach(),
            "train/total_loss": total_loss.detach(),
            "train/mask_ratio": 1.0 - mask.detach(),
        }
        return total_loss, tb_dict

    def _update_ctt(self, idx, max_idx, pseudo_label):
        self.ctt.update(idx, max_idx, pseudo_label.detach().mean(0))

    def method_state_dict(self, ctx):
        return {"ctt": self.ctt.state_dict()}

    def load_method_state(self, ctx, state):
        if "ctt" in state:
            self.ctt.load_state_dict(state["ctt"])


def consistency_loss_prg(logits_w, logits_s, H_prime, label_bank, idx, last=False, T=1.0, p_cutoff=0.0, use_hard_labels=True):
    """PRG 的伪标签修正一致性损失（与原 prg_utils.py 数学一致）。"""
    logits_w = logits_w.detach()
    pseudo_label = torch.softmax(logits_w / T, dim=-1)
    max_probs, max_idx = torch.max(pseudo_label, dim=-1)
    if last:
        for i in range(pseudo_label.size(0)):
            if idx[i].cpu().item() in label_bank.keys():
                pseudo_label[i, :] = _normalize_d(pseudo_label[i, :] * H_prime[label_bank[idx[i].cpu().item()], :])
    else:
        for i in range(pseudo_label.size(0)):
            pseudo_label[i, :] = _normalize_d(pseudo_label[i, :] * H_prime[max_idx[i], :])

    max_probs, max_idx = torch.max(pseudo_label, dim=-1)
    mask = max_probs.ge(p_cutoff).float()

    if use_hard_labels:
        masked_loss = ce_loss(logits_s, max_idx, use_hard_labels=True, reduction="none") * mask
    else:
        masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels=False, reduction="none") * mask
    return masked_loss.mean(), mask.mean()
