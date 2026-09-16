"""USP 类增量数据管线。

移植自 orig/USP4SSCL/：
- dataloder.py 的 BaseDataset / BaseDataset_flag / UnlabelDataset
  （weak/strong 双视图；CIFAR 分支与 TransformLoader 分支均保留）；
- utils_pytorch.py 的 get_data_file_cifar（k_shot 有标注 + 其余无标注池）、
  get_data_file_imagenet100（文件路径 + percentage 划分）；
- utils/randaugment.py 的 RandAugment 直接复用框架内
  sslunify/data/randaugment.py（两份实现逐字节一致）。

返回元组契约（与原实现一致，训练/评估代码按位置解包）：
    BaseDataset        -> (image_w, image_s, label)
    BaseDataset_flag   -> (index, image_w, image_s, label, flag, on_flag)
    UnlabelDataset     -> (image_w, image_s, gt)   # gt 仅用于监控
"""

from __future__ import annotations

import math
import os
import random
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision import transforms

from ..data.randaugment import RandAugment

# ------------------------------------------------------------------ transforms


class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [(ImageEnhance.Brightness, transformdict["Brightness"]),
                           (ImageEnhance.Contrast, transformdict["Contrast"]),
                           (ImageEnhance.Color, transformdict["Color"])]
        # 原实现按 transformtypedict 的键序遍历；此处固定 Brightness/Contrast/Color，
        # 与 dataloder.py 中唯一的使用方式 dict(Brightness=0.4, Contrast=0.4, Color=0.4) 等价。

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))
        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (randtensor[i] * 2.0 - 1.0) + 1
            out = transformer(out).enhance(r).convert("RGB")
        return out


class TransformLoader:
    def __init__(self, image_size,
                 normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param

    def parse_transform(self, transform_type):
        if transform_type == "ImageJitter":
            method = ImageJitter(self.jitter_param)
            return method
        method = getattr(transforms, transform_type)

        if transform_type == "RandomResizedCrop":
            return method(self.image_size)
        elif transform_type == "CenterCrop":
            return method(self.image_size)
        elif transform_type == "Resize":
            return method([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        elif transform_type == "Normalize":
            return method(**self.normalize_param)
        else:
            return method()

    def get_composed_transform(self, phase="train"):
        if phase == "train":
            transform_list = ["RandomResizedCrop", "ImageJitter", "RandomHorizontalFlip", "ToTensor", "Normalize"]
        elif phase == "test":
            transform_list = ["Resize", "CenterCrop", "ToTensor", "Normalize"]
        elif phase == "reserved":
            transform_list = ["RandomResizedCrop", "ImageJitter", "RandomHorizontalFlip", "Normalize"]
        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform


def get_transform(phase, image_size, normalize_param):
    trans_loader = TransformLoader(image_size, normalize_param=normalize_param)
    return trans_loader.get_composed_transform(phase)


# 归一化参数与原 dataloder.py 一致（cifar10 用 ImageNet 统计量为原实现行为，保留）
_CIFAR100_NORM = dict(mean=[x / 255 for x in [129.3, 124.1, 112.4]],
                      std=[x / 255 for x in [68.2, 65.4, 70.4]])
_IMAGENET_NORM = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def _cifar_transforms(dataset, image_size, phase):
    normalize_param = _CIFAR100_NORM if dataset == "cifar100" else _IMAGENET_NORM
    if phase == "train":
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomCrop(image_size, padding=int(image_size * (1 - 0.875)), padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**normalize_param),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(**normalize_param),
        ])
    strong_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size, padding=int(image_size * (1 - 0.875)), padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(**normalize_param),
    ])
    return transform, strong_transform


def _miniimagenet_transforms(image_size):
    normalize_param = _IMAGENET_NORM
    transform = transforms.Compose([
        transforms.Resize((int(math.floor(image_size / 0.875)), int(math.floor(image_size / 0.875)))),
        transforms.RandomCrop((image_size, image_size)),
        ImageJitter(dict(Brightness=0.4, Contrast=0.4, Color=0.4)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize_param),
    ])
    strong_transform = transforms.Compose([
        transforms.Resize((int(math.floor(image_size / 0.875)), int(math.floor(image_size / 0.875)))),
        transforms.RandomCrop((image_size, image_size)),
        ImageJitter(dict(Brightness=0.4, Contrast=0.4, Color=0.4)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(**normalize_param),
    ])
    return transform, strong_transform


def _build_transforms(dataset, image_size, phase):
    """按原 dataloder.py 的分支结构构建 (transform, strong_transform)。"""
    if dataset in ("cifar100", "cifar10"):
        return _cifar_transforms(dataset, image_size, phase)
    elif dataset == "miniimagenet":
        return _miniimagenet_transforms(image_size)
    else:  # imagenet100 / cub 等 TransformLoader 分支
        transform = get_transform(phase, image_size, _IMAGENET_NORM)
        # 原实现：deepcopy(transform) 后 insert(-2, RandAugment(3, 10))
        strong_transform = transforms.Compose(
            list(transform.transforms[:-2]) + [RandAugment(3, 10)] + list(transform.transforms[-2:]))
        return transform, strong_transform


# ------------------------------------------------------------------ datasets


class BaseDataset(Dataset):
    """弱/强双视图数据集，__getitem__ -> (image_w, image_s, label)。"""

    def __init__(self, phase, image_size, dataset="cifar100", autoaug=False):
        self.data = []
        self.targets = []
        self.dataset = dataset
        self.transform, self.strong_transform = _build_transforms(dataset, image_size, phase)

    def __getitem__(self, index):
        path, label = self.data[index], self.targets[index]
        if self.dataset in ("cifar100", "cifar10"):
            image = self.transform(Image.fromarray(path))
            image_s = self.strong_transform(Image.fromarray(path))
        else:
            image = self.transform(Image.open(path).convert("RGB"))
            image_s = self.strong_transform(Image.open(path).convert("RGB"))
        label = int(label)
        return image, image_s, label

    def __len__(self):
        return len(self.data)


class BaseDataset_flag(Dataset):
    """带 index / flag / on_flag 的训练数据集（flag=1 有标注来源，on_flag=1 基类）。"""

    def __init__(self, phase, image_size, dataset="cifar100", autoaug=False):
        self.data = []
        self.targets = []
        self.flags = []
        self.on_flags = []
        self.dataset = dataset
        self.transform, self.strong_transform = _build_transforms(dataset, image_size, phase)

    def __getitem__(self, index):
        path, label = self.data[index], self.targets[index]
        flags = self.flags[index]
        on_flags = self.on_flags[index]
        if self.dataset in ("cifar100", "cifar10"):
            image = self.transform(Image.fromarray(path))
            image_s = self.strong_transform(Image.fromarray(path))
        else:
            image = self.transform(Image.open(path).convert("RGB"))
            image_s = self.strong_transform(Image.open(path).convert("RGB"))
        label = int(label)
        return index, image, image_s, label, flags, on_flags

    def __len__(self):
        return len(self.data)


class UnlabelDataset(Dataset):
    """无标注池数据集（weak/strong 双视图，label 为 ground truth 仅用于监控）。"""

    def __init__(self, image_size, unlabeled_num=None, dataset="cifar100", autoaug=False):
        self.data = []
        self.targets = []
        self.dataset = dataset
        self.transform, self.strong_transform = _build_transforms(dataset, image_size, "train")

        if unlabeled_num != -1 and unlabeled_num is not None:
            try:
                self.data = self.data[: unlabeled_num]
                self.targets = self.targets[: unlabeled_num]
            except Exception:
                pass

    def __getitem__(self, index):
        path, label = self.data[index], self.targets[index]
        if self.dataset in ("cifar100", "cifar10"):
            image = self.transform(Image.fromarray(path))
            image_s = self.strong_transform(Image.fromarray(path))
        else:
            image = self.transform(Image.open(path).convert("RGB"))
            image_s = self.strong_transform(Image.open(path).convert("RGB"))
        return image, image_s, label

    def __len__(self):
        return len(self.data)


# ------------------------------------------------------------------ 数据划分

_CIFAR_CACHE: Dict[Tuple[str, str], object] = {}


def _load_cifar(data_dir, dataset):
    key = (os.path.expanduser(data_dir), dataset)
    if key not in _CIFAR_CACHE:
        import torchvision

        print(f"==> Preparing {dataset} data..")
        if dataset == "cifar100":
            _CIFAR_CACHE[key] = (
                torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True),
                torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True),
            )
        elif dataset == "cifar10":
            _CIFAR_CACHE[key] = (
                torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True),
                torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True),
            )
        else:
            raise ValueError("dataset must be cifar10 or cifar100")
    return _CIFAR_CACHE[key]


def _select_from_default(data, targets, index, num_per_class=None, return_ulb=False):
    """原 utils_pytorch.SelectfromDefault：按类取前 num_per_class 个为有标注，
    其余进无标注池；num_per_class=None 时取全部。"""
    data_tmp, targets_tmp = [], []
    udata_tmp, utargets_tmp = [], []

    for i in index:
        ind_cl = np.where(targets == i)[0]
        if num_per_class is not None:
            if len(data_tmp) == 0:
                data_tmp = data[ind_cl][:num_per_class]
                targets_tmp = targets[ind_cl][:num_per_class]
                udata_tmp = data[ind_cl][num_per_class:]
                utargets_tmp = targets[ind_cl][num_per_class:]
            else:
                data_tmp = np.vstack((data_tmp, data[ind_cl][:num_per_class]))
                targets_tmp = np.hstack((targets_tmp, targets[ind_cl][:num_per_class]))
                udata_tmp = np.vstack((udata_tmp, data[ind_cl][num_per_class:]))
                utargets_tmp = np.hstack((utargets_tmp, targets[ind_cl][num_per_class:]))
        else:
            if len(data_tmp) == 0:
                data_tmp = data[ind_cl]
                targets_tmp = targets[ind_cl]
            else:
                data_tmp = np.vstack((data_tmp, data[ind_cl]))
                targets_tmp = np.hstack((targets_tmp, targets[ind_cl]))

    if return_ulb:
        return data_tmp, targets_tmp, udata_tmp, utargets_tmp
    return data_tmp, targets_tmp


def get_cifar_session_data(data_dir, dataset, class_index, k_shot):
    """CIFAR 单 session 数据划分（get_data_file_cifar 的收敛接口）。

    Args:
        class_index: 本 session 的类区间（如 arange(s*nb_cl, (s+1)*nb_cl)）。
        k_shot: 每类有标注样本数（labels_num）。
    Returns:
        X_train, Y_train: 每类前 k_shot 个训练样本（np 数组）；
        U_data, U_gt    : 同类其余训练样本（无标注池）；
        X_valid, Y_valid: 这些类的全部测试样本。
    """
    trainset, testset = _load_cifar(data_dir, dataset)
    X_train, Y_train, U_data, U_gt = _select_from_default(
        trainset.data, np.array(trainset.targets), class_index,
        num_per_class=k_shot, return_ulb=True)
    X_valid, Y_valid = _select_from_default(
        testset.data, np.array(testset.targets), class_index)
    return X_train, Y_train, U_data, U_gt, X_valid, Y_valid


# ------------------------------------------------------------------ ImageNet-100

def find_classes(directory: Union[str, os.PathLike]) -> Tuple[List[str], Dict[str, int]]:
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(directory, class_to_idx, percentage=-1,
                 extensions=(".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")):
    """原 utils_pytorch.make_dataset（percentage 子采样 + random.shuffle 文件序）。"""
    instances = []
    directory = os.path.expanduser(directory)
    lb_idx = {}
    for target_class in sorted(class_to_idx.keys()):
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            random.shuffle(fnames)
            if percentage != -1:
                fnames = fnames[: int(len(fnames) * percentage)]
            if percentage != -1:
                lb_idx[target_class] = fnames
            for fname in fnames:
                if fname.lower().endswith(extensions):
                    instances.append((os.path.join(root, fname), class_to_idx[target_class]))
    return instances, lb_idx


def get_imagenet100_session_data(root, class_index, percentage=None, k_shot=None, train=True):
    """ImageNet-100 单 session 数据划分（get_data_file_imagenet100 的收敛接口）。

    root 需为 ImageFolder 结构（train/ 与 val/ 下按类名分目录，类 id 按类名排序）。
    有标注样本数 labels_num = int(percentage * N_total / N_classes)（原实现），
    percentage 未给时退化为 k_shot。

    Returns:
        (X_train, Y_train, U_data, U_gt) 或 (X_valid, Y_valid)（train=False）。
    """
    setname = "train" if train else "val"
    directory = os.path.join(root, setname)
    classes, class_to_idx = find_classes(directory)
    samples, _ = make_dataset(directory, class_to_idx)
    if len(samples) == 0:
        raise RuntimeError(f"Found 0 files in subfolders of: {directory}")

    data = [s[0] for s in samples]
    targets = [s[1] for s in samples]

    if not train:
        return _select_from_imagenet(data, targets, class_index)

    labels_num = (int(percentage * len(data) / len(classes))
                  if percentage is not None else k_shot)
    return _select_from_imagenet(data, targets, class_index, num_per_class=labels_num, return_ulb=True)


def _select_from_imagenet(data, targets, index, num_per_class=None, return_ulb=False):
    """原 get_data_file_imagenet100.SelectfromClasses：每类前 num_per_class 为有标注。"""
    data_tmp, targets_tmp = [], []
    udata_tmp, utargets_tmp = [], []

    if num_per_class is not None:
        for i in index:
            num_tmp = 0
            ind_cl = np.where(np.array(targets) == i)[0]
            for j in ind_cl:
                if num_tmp < num_per_class:
                    data_tmp.append(data[j])
                    targets_tmp.append(targets[j])
                else:
                    udata_tmp.append(data[j])
                    utargets_tmp.append(targets[j])
                num_tmp += 1
    else:
        for i in index:
            ind_cl = np.where(np.array(targets) == i)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])

    if return_ulb:
        return np.array(data_tmp), np.array(targets_tmp), np.array(udata_tmp), np.array(utargets_tmp)
    return np.array(data_tmp), np.array(targets_tmp)
