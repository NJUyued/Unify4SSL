"""USP 的 CIFAR ResNet 骨干（resnet20/32/44/56/110，通道零填充下采样 A 版）。

移植来源：orig/USP4SSCL/cifar_resnet.py（该文件在原仓库仅被 der_net.py 的
get_convnet 以 name in {resnet32, resnet20} 使用）。DownsampleA/B/C/D 与
ResNetBasicblock 逐行保留。

与 sslunify/cl/nets.py 的关系：cl/nets.py 移植的是另一个文件
orig/USP4SSCL/resnet32_cifar.py（CL 专用改造版：trans/trans_non 投影头、
DownsampleB、last_phase 去末层 ReLU、return_feats 四元组 forward），由
cl.SessionTrainer 经 build_cl_net 使用，与本文件互不替代；本文件是通用
注册入口（nets.get_builder('resnet32') 等）所需的那份纯骨干。

统一 forward 契约下的必要适配（其余逐行保真）：
- 原 CifarResNet.__init__ 无 num_classes 参数（fc 硬编码 Linear(64, 10)）；
  移植版增加 num_classes=10 形参（默认值即原硬编码值）。
- 原 forward 返回 dict {'fmaps': [x_1, x_2, x_3], 'features': features} 且
  从不调用 fc；移植版按统一契约返回
      forward(x)                -> logits            (= fc(features))
      forward(x, ood_test=True) -> (logits, feature)
  fmaps 中间特征不再从 forward 返回（DER 变体走 cl/ 专用骨干）。
- 增加 output_num() -> 64 * expansion。

统一 forward 契约：
    forward(x)                -> logits
    forward(x, ood_test=True) -> (logits, feature)
    output_num()              -> 特征维度（64）
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DownsampleA(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class DownsampleB(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleB, self).__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class DownsampleC(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleC, self).__init__()
        assert stride != 1 or nIn != nOut
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownsampleD(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleD, self).__init__()
        assert stride == 2
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=2, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ResNetBasicblock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBasicblock, self).__init__()

        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu(residual + basicblock, inplace=True)


class CifarResNet(nn.Module):
    """
    ResNet optimized for the Cifar Dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """

    def __init__(self, block, depth, channels=3, num_classes=10):
        super(CifarResNet, self).__init__()

        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6

        self.conv_1_3x3 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)

        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.out_dim = 64 * block.expansion
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, ood_test=False):
        x = self.conv_1_3x3(x)  # [bs, 16, 32, 32]
        x = F.relu(self.bn_1(x), inplace=True)

        x_1 = self.stage_1(x)  # [bs, 16, 32, 32]
        x_2 = self.stage_2(x_1)  # [bs, 32, 16, 16]
        x_3 = self.stage_3(x_2)  # [bs, 64, 8, 8]

        pooled = self.avgpool(x_3)  # [bs, 64, 1, 1]
        features = pooled.view(pooled.size(0), -1)  # [bs, 64]

        outputs = self.fc(features)

        if ood_test:
            return outputs, features
        return outputs

    def output_num(self):
        return self.out_dim

    @property
    def last_conv(self):
        return self.stage_3[-1].conv_b


def resnet20mnist():
    """Constructs a ResNet-20 model for MNIST."""
    model = CifarResNet(ResNetBasicblock, 20, 1)
    return model


def resnet32mnist():
    """Constructs a ResNet-32 model for MNIST."""
    model = CifarResNet(ResNetBasicblock, 32, 1)
    return model


def resnet20():
    """Constructs a ResNet-20 model for CIFAR-10."""
    model = CifarResNet(ResNetBasicblock, 20)
    return model


def resnet32():
    """Constructs a ResNet-32 model for CIFAR-10."""
    model = CifarResNet(ResNetBasicblock, 32)
    return model


def resnet44():
    """Constructs a ResNet-44 model for CIFAR-10."""
    model = CifarResNet(ResNetBasicblock, 44)
    return model


def resnet56():
    """Constructs a ResNet-56 model for CIFAR-10."""
    model = CifarResNet(ResNetBasicblock, 56)
    return model


def resnet110():
    """Constructs a ResNet-110 model for CIFAR-10."""
    model = CifarResNet(ResNetBasicblock, 110)
    return model


# ------------------------------------------------------------------ 统一 builder

def _make_cifar_resnet_builder(default_depth):
    """注册名内编码深度（resnet32 -> 32）。

    net_conf 的 depth 仅在是合法 CIFAR-ResNet 深度（(depth-2)%6==0，即
    20/32/44/56/110/...）时生效；否则（如框架 tools/train.py 对所有骨干统一
    传 depth=28）回落到注册名对应的深度，避免 assert 崩溃。
    """

    def builder(num_classes, depth=None, channels=3, **kwargs):
        d = default_depth
        if depth is not None and (depth - 2) % 6 == 0:
            d = depth
        return CifarResNet(ResNetBasicblock, d, channels=channels, num_classes=num_classes)

    return builder


build_resnet20 = _make_cifar_resnet_builder(20)
build_resnet32 = _make_cifar_resnet_builder(32)
build_resnet44 = _make_cifar_resnet_builder(44)
build_resnet56 = _make_cifar_resnet_builder(56)
build_resnet110 = _make_cifar_resnet_builder(110)
