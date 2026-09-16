"""Wide ResNet 骨干（FlexMatch 系 WRN + SoC 的 WRN-Var 变体）。

移植来源（同源收敛，以 MutexMatch 版为基准）：
- orig/MutexMatch4SSL/models/nets/wrn.py   —— 基准版（BasicBlock / NetworkBlock /
  WideResNet / build_WideResNet 逐行保留）
- orig/RDA4RobustSSL/models/nets/wrn.py    —— 与 MutexMatch 版仅空行差异（diff 确认）
- orig/SoC4SS-FGVC/models/nets/wrn.py      —— 与基准版唯一语义差异：forward 的
  ood_test 默认值为 True；统一框架约定 forward(x) 只返回 logits，
  故移植版统一为 ood_test=False（调用方 SoC 显式传 ood_test=True，行为不变）
- orig/PRG4SSL-MNAR/models/nets/wrn.py     —— 基准版 + 多出 forward(..., sel=1)
  截断分支（sel==1 硬编码 avg_pool2d(out, 8)；sel==3 走 block1+bn3）。全仓库
  grep 确认 sel 从未被任何调用方传入（prg.py 只调 train_model(inputs)），
  且 sel==1 默认分支在 32x32 输入下与基准版逐位等价，故收敛时按基准版去掉 sel。
- orig/SoC4SS-FGVC/models/nets/wrn_var.py  —— WideResNetVar：4 个 block / 5 级通道、
  conv 带 bias、kaiming(leaky_relu) 初始化、BN momentum/eps 硬编码 0.001，
  结构上无法参数化进 WideResNet，作为独立类整体保留（含未被调用的
  mish / PSBatchNorm2d 定义，忠实原文件）。

统一 forward 契约（框架其余部分依赖）：
    forward(x)                -> logits
    forward(x, ood_test=True) -> (logits, feature)
    output_num()              -> 特征维度
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

momentum = 0.001


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, bn_momentum=0.1, leaky_slope=0.0, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=bn_momentum)
        self.relu1 = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=bn_momentum)
        self.relu2 = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        if self.equalInOut:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, bn_momentum=0.1, leaky_slope=0.0, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, bn_momentum, leaky_slope, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, bn_momentum, leaky_slope, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, bn_momentum, leaky_slope, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, bn_momentum=0.1, leaky_slope=0.0, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, bn_momentum, leaky_slope, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, bn_momentum, leaky_slope, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, bn_momentum, leaky_slope, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=bn_momentum)
        self.bn3 = nn.BatchNorm2d(nChannels[1], momentum=bn_momentum)
        self.relu = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.__in_features = self.fc.in_features

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, ood_test=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        if out.size(3) == 8:
            out = F.avg_pool2d(out, 8)
        else:
            out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.nChannels)
        output = self.fc(out)
        if ood_test:
            return output, out
        else:
            return output

    def output_num(self):
        return self.__in_features


class build_WideResNet:
    """原仓库 builder 类（MutexMatch/PRG/RDA/SoC 四家一致），逐行保留。

    原始用法：build_WideResNet() 构造后由 utils.net_builder 用
    setattr_cls_from_kwargs 把 net_conf（depth/widen_factor/leaky_slope/
    bn_momentum/dropRate）写到实例属性上，再调用 .build(num_classes)。
    """

    def __init__(self, depth=28, widen_factor=2, bn_momentum=0.01, leaky_slope=0.0, dropRate=0.0):
        self.depth = depth
        self.widen_factor = widen_factor
        self.bn_momentum = bn_momentum
        self.dropRate = dropRate
        self.leaky_slope = leaky_slope

    def build(self, num_classes):
        return WideResNet(depth=self.depth,
                          num_classes=num_classes,
                          widen_factor=self.widen_factor,
                          bn_momentum=self.bn_momentum,
                          leaky_slope=self.leaky_slope,
                          dropRate=self.dropRate)


# ------------------------------------------------------------------ WRN-Var
# 以下整体移植自 orig/SoC4SS-FGVC/models/nets/wrn_var.py（含原文件中未被
# 调用的 mish / PSBatchNorm2d 定义），注册名 wrnvar / wrn_var。

def mish(x):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function (https://arxiv.org/abs/1908.08681)"""
    return x * torch.tanh(F.softplus(x))


class PSBatchNorm2d(nn.BatchNorm2d):
    """How Does BN Increase Collapsed Neural Network Filters? (https://arxiv.org/abs/2001.11216)"""

    def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, x):
        return super().forward(x) + self.alpha


class VarBasicBlock(nn.Module):
    """wrn_var 的 BasicBlock（与 WRN 的 BasicBlock 结构不同：conv 带 bias、
    activate_before_residual、BN momentum/eps 固定 0.001），原名 BasicBlock，
    此处加 Var 前缀避免与本模块 WRN 的 BasicBlock 混淆（仅类名变化，
    结构/初始化/forward 与原文件逐行一致，state_dict 键不受影响）。"""

    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        super(VarBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001, eps=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001, eps=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=True) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class VarNetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        super(VarNetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNetVar(nn.Module):
    def __init__(self, first_stride, num_classes, depth=28, widen_factor=2, drop_rate=0.0, is_remix=False):
        super(WideResNetVar, self).__init__()
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor, 128 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = VarBasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=True)
        # 1st block
        self.block1 = VarNetworkBlock(
            n, channels[0], channels[1], block, first_stride, drop_rate, activate_before_residual=True)
        # 2nd block
        self.block2 = VarNetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = VarNetworkBlock(
            n, channels[2], channels[3], block, 2, drop_rate)
        # 4th block
        self.block4 = VarNetworkBlock(
            n, channels[3], channels[4], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[4], momentum=0.001, eps=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.fc = nn.Linear(channels[4], num_classes)
        self.channels = channels[4]
        self.__in_features = self.fc.in_features

        # rot_classifier for Remix Match
        self.is_remix = is_remix
        if is_remix:
            self.rot_classifier = nn.Linear(self.channels, 4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x, ood_test=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)
        output = self.fc(out)

        if ood_test:
            return output, out
        else:
            if self.is_remix:
                rot_output = self.rot_classifier(out)
                return output, rot_output
            else:
                return output

    def output_num(self):
        return self.__in_features


class build_WideResNetVar:
    """原 SoC wrn_var.py 的 builder 类，逐行保留（bn_momentum / leaky_slope /
    use_embed 被接受但不传入 WideResNetVar，与原实现一致）。"""

    def __init__(self, first_stride=1, depth=28, widen_factor=2, bn_momentum=0.01, leaky_slope=0.0, dropRate=0.0,
                 use_embed=False, is_remix=False):
        self.first_stride = first_stride
        self.depth = depth
        self.widen_factor = widen_factor
        self.bn_momentum = bn_momentum
        self.dropRate = dropRate
        self.leaky_slope = leaky_slope
        self.use_embed = use_embed
        self.is_remix = is_remix

    def build(self, num_classes):
        return WideResNetVar(
            first_stride=self.first_stride,
            depth=self.depth,
            num_classes=num_classes,
            widen_factor=self.widen_factor,
            drop_rate=self.dropRate,
            is_remix=self.is_remix,
        )


# ------------------------------------------------------------------ 统一 builder
# get_builder('wrn') / get_builder('wrnvar') 返回的函数签名：
#     builder(num_classes=..., **net_conf)
# net_conf 兼容两套参数名（自动别名归一）：
# - 原仓库 train 脚本：{'depth', 'widen_factor', 'leaky_slope', 'bn_momentum', 'dropRate'}
# - 框架 tools/train.py：{'depth', 'widen_factor', 'leaky_slope', 'dropout'}

def _norm_drop(dropRate=0.0, drop_rate=None, dropout=None):
    """dropout / drop_rate / dropRate 三种写法归一（后出现者优先级：
    dropout > drop_rate > dropRate，未提供时用原 builder 类默认 dropRate）。"""
    if dropout is not None:
        return dropout
    if drop_rate is not None:
        return drop_rate
    return dropRate


def build_wrn(num_classes, depth=28, widen_factor=2, bn_momentum=0.01, leaky_slope=0.0,
              dropRate=0.0, drop_rate=None, dropout=None, **kwargs):
    """WideResNet（注册名 wrn / wideresnet）。默认值与原 build_WideResNet 一致。"""
    return build_WideResNet(
        depth=depth,
        widen_factor=widen_factor,
        bn_momentum=bn_momentum,
        leaky_slope=leaky_slope,
        dropRate=_norm_drop(dropRate, drop_rate, dropout),
    ).build(num_classes)


def build_wrn_var(num_classes, first_stride=1, depth=28, widen_factor=2, bn_momentum=0.01,
                  leaky_slope=0.0, dropRate=0.0, drop_rate=None, dropout=None,
                  use_embed=False, is_remix=False, **kwargs):
    """WideResNetVar（注册名 wrnvar / wrn_var / wideresnetvar，SoC 的 WRN-37-2 变体）。"""
    return build_WideResNetVar(
        first_stride=first_stride,
        depth=depth,
        widen_factor=widen_factor,
        bn_momentum=bn_momentum,
        leaky_slope=leaky_slope,
        dropRate=_norm_drop(dropRate, drop_rate, dropout),
        use_embed=use_embed,
        is_remix=is_remix,
    ).build(num_classes)
