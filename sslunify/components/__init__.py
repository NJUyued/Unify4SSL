"""共享方法学组件。

这是 Unify4SSL 的核心：原 5 个仓库中重复出现的算法数据结构收敛为可复用组件。
- ClassTransitionTracker：PRG 与 SoC 原本逐行相同的 CTT（类别转移追踪）
- DistQueue：RDA（双头分布）、PRG（类分布）、USP（DistAlign）共用的滑窗分布队列
- ReverseCLS：MutexMatch 与 RDA 共用的互补标签分类头
- kmeans：SoC 的多粒度聚类
"""

from .ctt import ClassTransitionTracker
from .dist_queue import DistQueue
from .heads import ReverseCLS, argmin_complementary, random_complementary
from .kmeans import multigranularity_kmeans, kmeans
