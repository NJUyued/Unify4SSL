"""USP 持续学习骨干网络。

移植自 orig/USP4SSCL/：
- resnet32_cifar.py : CIFAR ResNet（改造版：trans 投影到 ETF 维度 dim、
  return_feats 返回四元组、可扩张 fc；本次移植补充 fc_session=None 属性
  以对齐 resnet.py 的接口——原 resnet32_cifar.py 无此属性，
  use_session_labels 路径在 resnet32 上会 AttributeError）；
- resnet.py         : 224px torchvision 风格 resnet18（ImageNet-100 用），
  含 trans / trans_non / fc_session 与 ImageNet 预训练加载。

forward 契约（与原实现逐字节一致，cl/methods 代码依赖此约定）：
    forward(x)                          -> logits
    forward(x, return_feats=True)       -> (outputs, feats, con_feats, non_feats)
    forward(x, return_feats_list=True)  -> 同上（CIFAR 版两flag等价）

其中 feats 为池化后的骨干特征（resnet32: 64 维 / resnet18: 512 维，
fc 与原型分类器所在空间），con_feats = trans(feats)（ETF 对齐空间，dim 维），
non_feats 为最后一个 stage 的空间特征图（der 变体用，此处保留输出）。

fc 扩张（session 递进时由 SessionTrainer 执行）：
    new_fc = nn.Linear(in_features, out_features + nb_cl)
    new_fc.weight.data[:out_features] = old_fc.weight.data
    new_fc.bias.data[:out_features]   = old_fc.bias.data
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.hub import load_state_dict_from_url
except ImportError:  # pragma: no cover
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
}


# ------------------------------------------------------------------ CIFAR ResNet
# 以下 4 个 Downsample 与 ResNetBasicblock 为原文件逐行移植。

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

    def __init__(self, inplanes, planes, stride=1, downsample=None, last=False):
        super(ResNetBasicblock, self).__init__()

        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.last = last

    def forward(self, x):
        residual = x

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = residual + basicblock
        if not self.last:
            out = F.relu(out, inplace=True)

        return out


class CifarResNet(nn.Module):
    """ResNet optimized for the Cifar Dataset (原文件逐行移植 + fc_session 属性)。

    trans: 64 -> dim 的投影（FSR/ETF 对齐空间）；
    trans_non: 64 -> dim（der 变体的独立投影头，保留以保证参数结构一致）。
    """

    def __init__(self, block, depth, channels=3, use_proto_classifer=False, no_trans=False,
                 temperature=0.1, dim=128, no_linear=False):
        super(CifarResNet, self).__init__()

        assert (depth - 2) % 6 == 0, "depth should be one of 20, 32, 44, 56, 110"
        layer_blocks = (depth - 2) // 6

        self.conv_1_3x3 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)

        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2, last_phase=True)
        self.avgpool = nn.AvgPool2d(8)
        self.out_channels = 64 * block.expansion
        self.fc = nn.Linear(64 * block.expansion, 10)

        self.use_proto_classifer = use_proto_classifer
        self.temperature = temperature

        if self.use_proto_classifer:
            print("Using Proto Classifier, temperature:", self.temperature)

        if no_trans:
            print("Not use trans!")
            self.trans = nn.Identity()
        else:
            if no_linear:
                print("Not linear trans!")
                self.trans = nn.Sequential(
                    nn.Linear(self.out_channels, self.out_channels),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.out_channels, dim),
                )
            else:
                self.trans = nn.Linear(self.out_channels, dim)

        self.trans_non = nn.Linear(self.out_channels, dim)

        # 补充：对齐 resnet.py 的接口（use_session_labels 路径引用）
        self.fc_session = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, last_phase=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleB(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        if last_phase:
            for _ in range(1, blocks - 1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, last=True))
        else:
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, return_feats=False, return_feats_list=False):
        x = self.conv_1_3x3(x)  # [bs, 16, 32, 32]
        x = F.relu(self.bn_1(x), inplace=True)

        x_1 = self.stage_1(x)  # [bs, 16, 32, 32]
        x_2 = self.stage_2(x_1)  # [bs, 32, 16, 16]
        x_3 = self.stage_3(x_2)  # [bs, 64, 8, 8]

        non_feats = x_3

        pooled = self.avgpool(x_3)  # [bs, 64, 1, 1]
        feats = pooled.view(pooled.size(0), -1)  # [bs, 64]
        outputs = self.fc(feats)

        con_feats = self.trans(feats)

        if return_feats or return_feats_list:
            return outputs, feats, con_feats, non_feats
        return outputs

    @property
    def last_conv(self):
        return self.stage_3[-1].conv_b


def resnet20(num_classes, pretrained=False, progress=True, use_proto_classifer=False, **kwargs):
    """Constructs a ResNet-20 model for CIFAR."""
    model = CifarResNet(ResNetBasicblock, 20, use_proto_classifer=use_proto_classifer, **kwargs)
    num_ftrs = model.fc.in_features
    if use_proto_classifer:
        model.fc = nn.Linear(num_ftrs, num_classes, bias=False)
    else:
        model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def resnet32(num_classes, pretrained=False, progress=True, use_proto_classifer=False, **kwargs):
    """Constructs a ResNet-32 model for CIFAR."""
    model = CifarResNet(ResNetBasicblock, 32, use_proto_classifer=use_proto_classifer, **kwargs)
    num_ftrs = model.fc.in_features
    if use_proto_classifer:
        print("Using Proto Classifier")
        model.fc = nn.Linear(num_ftrs, num_classes, bias=False)
    else:
        print("Using Params Classifier")
        model.fc = nn.Linear(num_ftrs, num_classes)
    return model


# ------------------------------------------------------------------ 224px ResNet
# 移植自 orig/USP4SSCL/resnet.py（torchvision 风格，仅保留 BasicBlock/resnet18）。

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """224px ResNet（resnet.py 逐行移植；trans/trans_non/fc_session 齐全）。"""

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, use_proto_classifer=False, no_trans=False,
                 temperature=0.1, dim=128, no_linear=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_channels = 512 * block.expansion

        self.fc = nn.Linear(self.out_channels, num_classes)

        self.use_proto_classifer = use_proto_classifer
        self.temperature = temperature

        if self.use_proto_classifer:
            print("Using Proto Classifier, temperature:", self.temperature)

        if no_trans:
            print("Not use trans!")
            self.trans = nn.Identity()
        else:
            if no_linear:
                print("Not linear trans!")
                self.trans = nn.Sequential(
                    nn.Linear(self.out_channels, self.out_channels),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.out_channels, dim),
                )
            else:
                self.trans = nn.Linear(self.out_channels, dim)

        self.trans_non = nn.Linear(self.out_channels, dim)

        self.fc_session = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, return_feats=False, return_feats_list=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        feats0 = x

        x = self.maxpool(x)

        x = self.layer1(x)
        feats1 = x
        x = self.layer2(x)
        feats2 = x
        x = self.layer3(x)
        feats3 = x
        x = self.layer4(x)
        feats4 = x

        x = self.avgpool(x)
        feats = torch.flatten(x, 1)

        outputs = self.fc(feats)

        if self.fc_session is not None:
            session_outputs = self.fc_session(feats)
        else:
            session_outputs = None

        con_feats = self.trans(feats)
        non_feats = self.trans_non(feats)

        if return_feats_list:
            return outputs, feats, con_feats, [feats0, feats1, feats2, feats3, feats4]

        if return_feats:
            return outputs, feats, con_feats, feats4
        return outputs

    def forward(self, x, return_feats=False, return_feats_list=False):
        return self._forward_impl(x, return_feats, return_feats_list)

    def get_logits(self, feats):
        logits = self.fc(feats)
        return logits


def _resnet(num_classes, arch, block, layers, pretrained, progress, use_proto_classifer, **kwargs):
    model = ResNet(block, layers, use_proto_classifer=use_proto_classifer, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict, strict=False)

    num_ftrs = model.fc.in_features
    if use_proto_classifer:
        print("Using Proto Classifier")
        model.fc = nn.Linear(num_ftrs, num_classes, bias=False)
    else:
        print("Using Params Classifier")
        model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def resnet18(num_classes, pretrained=False, progress=True, use_proto_classifer=False, **kwargs):
    """ResNet-18 model（224px，ImageNet-100 持续学习骨干，支持 ImageNet 预训练）。"""
    return _resnet(num_classes, "resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   use_proto_classifer, **kwargs)


# ------------------------------------------------------------------ builder

def build_cl_net(name, num_classes, **kwargs):
    """USP 骨干构建入口：resnet32 / resnet20（CIFAR 32px）、resnet18（224px）。"""
    name = name.lower()
    builders = {
        "resnet32": resnet32,
        "resnet20": resnet20,
        "resnet18": resnet18,
    }
    if name not in builders:
        raise ValueError(f"model {name} not supported (available: {sorted(builders)})")
    return builders[name](num_classes, **kwargs)


def feature_model(net):
    """特征子模型 = list(net.children())[:-3]（原 train_semi.py 的构造方式）。

    CifarResNet children: conv_1_3x3, bn_1, stage_1..3, avgpool, fc, trans, trans_non
        -> [:-3] 输出 (B, 64, 1, 1)（squeeze 后 64 维原型空间）
    ResNet children: conv1..avgpool, fc, trans, trans_non
        -> [:-3] 输出 (B, 512, 1, 1)
    """
    return nn.Sequential(*list(net.children())[:-3])
