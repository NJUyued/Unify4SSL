"""类别转移追踪器（CTT）。

PRG（ICCV'23）与 SoC（AAAI'24）各自仓库中逐行相同的实现，统一为一个组件：

- label_bank      : {样本 idx: 上一次伪标签类别}
- transition      : C×C×N 环形计数矩阵——样本伪标签从 i 翻转到 j 时
                    transition[i, j, slot] += 1
- distri          : N×C，各 batch softmax 均值的滑动窗口
- 槽位语义与原实现一致：写入 slot 后指针前移一位，并将新指向的槽位清零
  （即"即将开始收集"的槽位先清空）

两个方法的不同消费方式：
- PRG  : H_prime() —— 转移矩阵行归一化 + 对角先验 + 按类分布重缩放，用于
         Markov 游走式伪标签修正（放大被忽视的稀有类）
- SoC  : affinity() —— transition 对 N 求和得类-类亲和矩阵，供多粒度 k-means
         把易混淆类聚成簇
"""

from __future__ import annotations

import torch


class ClassTransitionTracker:
    def __init__(self, num_classes: int, window: int = 128, device="cpu", init_distri_ones: bool = False):
        self.num_classes = num_classes
        self.window = window
        self.label_bank: dict[int, int] = {}
        self.transition = torch.zeros(num_classes, num_classes, window)
        # RDA 系初始化为全 1，PRG/SoC 为全 0（保持各自原实现的差异）
        self.distri = (
            torch.ones(window, num_classes) if init_distri_ones else torch.zeros(window, num_classes)
        )
        self.slot = 0
        self.device = device

    def to(self, device):
        self.device = device
        self.transition = self.transition.to(device)
        self.distri = self.distri.to(device)
        return self

    @torch.no_grad()
    def update(self, idx, max_idx, probs_mean):
        """更新一次：idx/max_idx 为本 batch 的样本索引与当前伪标签（任意设备）。

        probs_mean: 本 batch 的 softmax 概率均值（C 维，已 detach）。
        """
        idx_list = idx.tolist() if torch.is_tensor(idx) else list(idx)
        max_list = max_idx.tolist() if torch.is_tensor(max_idx) else list(max_idx)
        for i, m in zip(idx_list, max_list):
            if i not in self.label_bank:
                self.label_bank[i] = m
            elif self.label_bank[i] != m:
                self.transition[self.label_bank[i], m, self.slot] += 1
                self.label_bank[i] = m
        self.distri[self.slot] = probs_mean.to(self.transition.device)
        self.slot = (self.slot + 1) % self.window
        self.transition[:, :, self.slot] = 0

    # ------------------------------------------------------------- PRG 视角

    @torch.no_grad()
    def H_prime(self, alpha: float = 1.0) -> torch.Tensor:
        """PRG 的修正转移矩阵 H' = H / mean(distri, dim=0)。

        H = row_norm(mean(transition, dim=2)) + alpha/(C-1) * I
        """
        ctt_mean = self.transition.mean(dim=2)
        diag = torch.diag(torch.ones(self.num_classes, self.num_classes) * (alpha / (self.num_classes - 1)))
        a_diag = torch.diag_embed(diag).to(ctt_mean.device)
        H = ctt_mean / (ctt_mean.abs().sum(1, keepdim=True) + 1e-16)
        H = H + a_diag
        return H / self.distri.mean(dim=0)

    # ------------------------------------------------------------- SoC 视角

    @torch.no_grad()
    def affinity(self) -> torch.Tensor:
        """SoC 的类-类亲和矩阵：transition 对窗口求和。"""
        return self.transition.sum(dim=2)

    # ------------------------------------------------------------- ckpt

    def state_dict(self):
        return {
            "label_bank": self.label_bank,
            "transition": self.transition.cpu(),
            "distri": self.distri.cpu(),
            "slot": self.slot,
        }

    def load_state_dict(self, state):
        self.label_bank = state["label_bank"]
        self.transition = state["transition"].to(self.device)
        self.distri = state["distri"].to(self.device)
        self.slot = state["slot"]
