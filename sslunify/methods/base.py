"""方法基类：统一接口定义。

原始 5 个仓库中每个方法类都是 300-400 行的全套（模型构建 + 训练循环 + 评估 + 存档），
其中约 80% 是逐字节相同的样板代码。统一框架中：

- SSLTrainer（core/trainer.py）持有循环骨架：数据迭代、设备搬运、AMP、backward、
  optimizer/scheduler.step、EMA、评估节奏、best 追踪、checkpoint；
- SSLMethod 只实现方法学差异部分：
    build_models   —— 模型结构（单骨干 / MutexMatch-RDA 的双头 TotalNet）
    train_step     —— 单次迭代的前向 + 状态更新 + 损失计算（原 train() 循环体的方法特有部分）
    eval_forward   —— 评估时如何前向（默认直接 eval_model(x)）
    extra_metrics  —— 方法特有的评估指标（如 PRG 的 GM / precision / recall）

批次 schema 统一为（在数据层归一，消除原代码中 miniimage 的特殊分支）：
    labeled:   (x_lb, y_lb)
    unlabeled: (x_ulb_w, x_ulb_s, y_ulb, idx)
"""

from __future__ import annotations

from ..core.ema import init_eval_model


class TrainContext:
    """贯穿一次训练的上下文：配置、模型、迭代状态、方法私有状态。

    method 可以在 setup() 中往 self.state 里挂任意跨迭代状态
    （PRG/SoC 的 CTT、RDA 的双头分布队列、MutexMatch 的互补标签头状态等）。
    """

    def __init__(self, cfg, print_fn=print):
        self.cfg = cfg
        self.it = 0
        self.print_fn = print_fn
        self.state = {}


class SSLMethod:
    name: str = "base"

    def __init__(self, ctx: TrainContext):
        self.ctx = ctx
        self.cfg = ctx.cfg
        self.num_classes = ctx.cfg.num_classes
        self.train_model, self.eval_model = self.build_models(ctx)
        init_eval_model(self.train_model, self.eval_model)
        self.setup(ctx)

    # ------------------------------------------------------------------ hooks

    def build_models(self, ctx):
        """默认：单骨干 train_model + eval_model（EMA 影子）。双头方法覆写。"""
        build = ctx.cfg.net_builder
        train_model = build(num_classes=self.num_classes)
        eval_model = build(num_classes=self.num_classes)
        return train_model, eval_model

    def setup(self, ctx):
        """方法私有跨迭代状态初始化（CTT / 分布队列 / label bank 等）。"""

    def trainable_parameters(self):
        """返回参与优化器的参数（双头方法需要合并两个模块）。"""
        return self.train_model.parameters()

    def optimizer_modules(self):
        """返回优化器应覆盖的模块列表（供 BN 免 weight decay 逻辑用 name 匹配）。

        双头方法覆写为 [feature_extractor, classifier_reverse, ...]。
        """
        return [self.train_model]

    def train_step(self, ctx, batch):
        """单次迭代：前向 + 状态更新 + 损失。

        batch: dict(x_lb, y_lb, x_ulb_w, x_ulb_s, y_ulb, idx)，已在目标设备上。
        return: (total_loss, tb_dict)
        """
        raise NotImplementedError

    def eval_forward(self, ctx, model, x):
        """评估前向，默认直接 model(x) 返回 logits。"""
        return model(x)

    def extra_metrics(self, ctx, y_true, y_pred) -> dict:
        """方法特有评估指标（默认空）。"""
        return {}

    def method_state_dict(self, ctx) -> dict:
        """方法私有状态 checkpoint（CTT / 队列等），随模型一起保存。"""
        return {}

    def load_method_state(self, ctx, state: dict):
        """恢复方法私有状态。"""
