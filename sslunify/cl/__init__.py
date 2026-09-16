"""USP (ICCV 2025) 类增量持续学习子包。

提供 SessionTrainer：USP 半监督持续学习的 session 外循环驱动器。

USP = Unlabeled learning, Stability, and Plasticity，核心为
Divide-and-Conquer 三件套（实现见 methods/usp.py）：
- FSR : Feature-Space Regularization——视觉特征与 ETF 等角紧框架锚对齐
- DCP : Divide-and-Conquer Pseudo-labeling——高置信走分类器路径（FixMatch
        式 DA 对齐 + 阈值掩码），低置信走原型余弦分类器路径
- CUD : Confidence-aware Unlabeled Distillation——无标注数据上的
        原型相似度蒸馏（teacher=旧模型特征对旧原型 softmax）

移植范围（orig/USP4SSCL/）：
- train_semi.py                  -> cl/session.py（单骨干可扩张 fc 路径）
- utils_incremental/incremental_train_and_eval_semi.py -> methods/usp.py
- utils_incremental/dist_align.py / etf.py -> cl/dist_align.py / cl/etf.py
- resnet32_cifar.py / resnet.py  -> cl/nets.py
- dataloder.py / utils_pytorch.py 的数据划分 -> cl/data.py

未移植（原仓库中明确不在统一框架范围内的路径）：
- train_semi_der.py + der_net.py：DER 双骨干扩张变体（--model resnet32 的
  DER 版本，run_cifar.sh 中以 train_semi_der.py 调用）；
- train_cub.py + dataloader/cub200/：CUB 旧管线（get_data_file + label2id
  文本 split 驱动，与 CIFAR/ImageNet 的类区间划分不同源）。
"""

from .session import SessionTrainer

__all__ = ["SessionTrainer"]
