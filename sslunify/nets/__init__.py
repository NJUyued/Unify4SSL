"""统一骨干网络注册表。

移植来源（4 个原始半监督仓库的骨干收敛）：
- wrn.py         : WideResNet（MutexMatch 基准版；RDA/SoC 同源，PRG 的 sel
                   分支收敛去除）+ WideResNetVar（SoC wrn_var.py）
- cnn13.py       : CNN13（MutexMatch / PRG / RDA / SoC 四家完全一致）
- resnet_cifar.py: USP 的 CIFAR ResNet（cifar_resnet.py，resnet20/32/44/56/110）
- resnet_tv.py   : torchvision 风格 ResNet18/50/101（SoC resnet.py）
- preresnet.py   : PreAct ResNet（PRG/RDA prern.py，两者完全一致）

统一接口契约（sslunify/utils/common.py 的 net_builder 依赖）：
    builder = nets.get_builder(net_name)          # net_name 已 .lower()
    model   = builder(num_classes=..., **net_conf)  # net_conf 典型 kwargs:
                                                    #   depth / widen_factor /
                                                    #   leaky_slope / dropout
                                                    # （兼容原仓库写法 dropRate /
                                                    #   drop_rate / bn_momentum）

骨干类统一 forward 约定：
    forward(x)                -> logits
    forward(x, ood_test=True) -> (logits, feature)   # MutexMatch / RDA 双头所需
    output_num()              -> 特征维度（int）

注：USP 持续学习专用骨干（cifar_resnet_t / resnet32_cifar 改造版，forward 返回
四元组）在 sslunify/cl/nets.py，经 build_cl_net 使用，不经本注册表。
"""

from .cnn13 import CNN13, build_CNN13, build_cnn13
from .preresnet import (
    ResNet as PreActResNet,
    ResNet18 as PreActResNet18,
    build_prern,
    build_preresnet,
)
from .resnet_cifar import (
    CifarResNet,
    build_resnet110,
    build_resnet20,
    build_resnet32,
    build_resnet44,
    build_resnet56,
)
from .resnet_tv import (
    ResNet as ResNetTV,
    build_ResNet50,
    build_resnet101,
    build_resnet18,
    build_resnet50,
)
from .wrn import (
    WideResNet,
    WideResNetVar,
    build_WideResNet,
    build_WideResNetVar,
    build_wrn,
    build_wrn_var,
)

__all__ = [
    "get_builder",
    # 网络类
    "WideResNet", "WideResNetVar", "CNN13", "CifarResNet", "ResNetTV", "PreActResNet",
    # 原仓库 builder 类（逐行保留，供需要类式 API 的调用方使用）
    "build_WideResNet", "build_WideResNetVar", "build_CNN13", "build_ResNet50", "build_prern",
    # 统一 builder 函数
    "build_wrn", "build_wrn_var", "build_cnn13", "build_preresnet",
    "build_resnet18", "build_resnet50", "build_resnet101",
    "build_resnet20", "build_resnet32", "build_resnet44", "build_resnet56", "build_resnet110",
]

_BUILDERS = {
    # Wide ResNet（FlexMatch 系）
    "wrn": build_wrn,
    "wideresnet": build_wrn,
    # Wide ResNet-Var（SoC，WRN-37-2 变体）
    "wrnvar": build_wrn_var,
    "wrn_var": build_wrn_var,
    "wideresnetvar": build_wrn_var,
    # CNN-13
    "cnn13": build_cnn13,
    # torchvision 风格 ResNet（SoC resnet.py，7x7 stem）
    "resnet18": build_resnet18,
    "resnet50": build_resnet50,
    "resnet101": build_resnet101,
    # USP CIFAR ResNet（cifar_resnet.py）
    "resnet20": build_resnet20,
    "resnet32": build_resnet32,
    "resnet44": build_resnet44,
    "resnet56": build_resnet56,
    "resnet110": build_resnet110,
    # PreAct ResNet（PRG/RDA prern.py）
    "preresnet": build_preresnet,
    "preactresnet": build_preresnet,
}


def get_builder(net_name):
    """按注册名返回 builder 函数：builder(num_classes=..., **net_conf)。

    注册名（小写）：
        wrn / wideresnet                      —— WideResNet
        wrnvar / wrn_var / wideresnetvar      —— WideResNetVar（SoC 变体）
        cnn13                                 —— CNN13
        resnet18 / resnet50 / resnet101       —— torchvision 风格（SoC）
        resnet20 / resnet32 / resnet44 / resnet56 / resnet110 —— USP CIFAR
        preresnet / preactresnet              —— PreAct ResNet（默认 ResNet18）
    """
    name = str(net_name).lower()
    if name not in _BUILDERS:
        raise ValueError(
            f"net '{net_name}' not supported. available: {sorted(_BUILDERS)}"
        )
    return _BUILDERS[name]
