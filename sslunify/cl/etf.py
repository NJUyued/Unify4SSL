"""ETF（等角紧框架）锚生成。

移植自 orig/USP4SSCL/utils_incremental/etf.py（与 utils_pytorch.py 中同名
函数逐字节一致，原代码两处重复实现，此处收敛为一份）。

生成流程：
1. 随机矩阵 (feat_in, num_classes) 做 QR 分解取正交因子 orth_vec；
2. etf_vec = orth_vec @ (I - J/C) * sqrt(C/(C-1))，其中 J 为全 1 矩阵。

得到的 etf_vec 形状 (feat_in, num_classes)，转置后即 C 个类锚（列向量），
满足任意两锚内积为 -1/(C-1)、各锚模长为 1。
"""

from __future__ import annotations

import math

import numpy as np
import torch


def generate_random_orthogonal_matrix(feat_in: int, num_classes: int) -> torch.Tensor:
    """随机正交矩阵：QR 分解 (feat_in, num_classes) 随机矩阵。"""
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)  # (feat_in, min(feat_in, num_classes))
    orth_vec = torch.tensor(orth_vec).float()
    assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.0e-7), \
        "生成的矩阵不是正交矩阵"
    return orth_vec


def generate_etf_vector(in_channels: int, num_classes: int) -> torch.Tensor:
    """ETF 向量，形状 (in_channels, num_classes)，每列一个类锚。"""
    orth_vec = generate_random_orthogonal_matrix(in_channels, num_classes)

    i_nc_nc = torch.eye(num_classes)
    one_nc_nc = torch.mul(torch.ones(num_classes, num_classes), (1 / num_classes))

    etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                        math.sqrt(num_classes / (num_classes - 1)))

    return etf_vec
