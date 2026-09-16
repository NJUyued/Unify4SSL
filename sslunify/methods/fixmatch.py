"""FixMatch 基线（Sohn et al., 2020）。

作为统一框架的回归验证锚点：其余方法均在其基础上修改伪标签环节，
保证 FixMatch 路径的可复现性即保证框架骨架忠实于原始代码库。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core.losses import ce_loss
from .base import SSLMethod


class FixMatch(SSLMethod):
    name = "fixmatch"

    def train_step(self, ctx, batch):
        cfg = self.cfg
        x_lb, y_lb = batch["x_lb"], batch["y_lb"]
        x_ulb_w, x_ulb_s = batch["x_ulb_w"], batch["x_ulb_s"]
        num_lb = batch["num_lb"]

        inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
        logits = self.train_model(inputs)
        logits_x_lb = logits[:num_lb]
        logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)

        pseudo_label = torch.softmax(logits_x_ulb_w.detach() / batch["T"], dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(batch["p_cutoff"]).float()

        sup_loss = ce_loss(logits_x_lb, y_lb, reduction="mean")
        unsup_loss = (ce_loss(logits_x_ulb_s, max_idx, reduction="none") * mask).mean()
        total_loss = sup_loss + cfg.ulb_loss_ratio * unsup_loss

        tb_dict = {
            "train/sup_loss": sup_loss.detach(),
            "train/unsup_loss": unsup_loss.detach(),
            "train/total_loss": total_loss.detach(),
            "train/mask_ratio": 1.0 - mask.mean(),
        }
        return total_loss, tb_dict
