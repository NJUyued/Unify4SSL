"""Unify4SSL: unified framework for semi-supervised learning methods.

统一实现以下方法（均来自 NJUyued 的原始论文代码库）：
- FixMatch（基线锚点）
- RDA   (ECCV 2022)  互惠分布对齐
- PRG   (ICCV 2023)  非随机缺失标签下的伪标签修正
- SoC   (AAAI 2024)  细粒度 SSL 的软标签选择扩张收缩
- MutexMatch (TNNLS 2024) 互斥一致性正则
- USP   (ICCV 2025)  半监督持续学习（cl.SessionTrainer 驱动）
"""

__version__ = "0.1.0"
