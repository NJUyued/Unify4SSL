"""特征记忆库、类原型与 iCaRL herding exemplar buffer。

移植自 orig/USP4SSCL/utils_incremental/incremental_train_and_eval_semi.py：
- FeatureMemBank : mem_bank (dim, N) / labels_bank (N)，update_bank 按 index
                   覆写当前 batch 的归一化特征（USP 中只写不读，保留以维持
                   与原实现一致的状态流）；
- get_proto      : 逐类累加 feats 求均值原型，按 old_cn 切分为
                   (prototypes_old, prototypes_new, prototypes_all)，
                   normalize=False（USP 训练路径的调用方式，归一化在使用处
                   进行）；
- fill_pro_list  : 训练后用无标注数据按分类器置信度 top-k 补充每类样本池
                   （prototypes 列表，即后续 herding 的候选池）；
- select_exemplars_herding : iCaRL herding 选样（train_semi.py 主循环里的
                   dr_herding 过程）。

exemplar buffer：buffer_size（run_cifar.sh 默认 5120）按
nb_protos_cl = buffer_size // ((session+1) * nb_cl) 均摊到每个已见类。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class FeatureMemBank:
    """特征记忆库：USP 原 update_bank 的状态部分。

    mem_bank: (dim, N) 按样本 index 存归一化特征；
    labels_bank: (N,) 存对应标签。
    """

    def __init__(self, dim: int, num_samples: int, device):
        mem_bank = torch.randn(dim, num_samples).to(device)
        self.mem_bank = F.normalize(mem_bank, dim=0).detach()
        self.labels_bank = torch.zeros(num_samples, dtype=torch.long).to(device).detach()

    @torch.no_grad()
    def update_bank(self, feats: torch.Tensor, labels: torch.Tensor, index: torch.Tensor):
        """feats: (B, dim)（原实现此处未显式归一化，写入时按列归一化）。"""
        self.mem_bank[:, index] = F.normalize(feats).t().detach()
        self.labels_bank[index] = labels.detach()


@torch.no_grad()
def get_proto(trainloader, tg_model, old_cn: int, device, normalize: bool = False):
    """逐类特征均值原型。

    遍历 trainloader 的弱视图，按标签累加 con_feats（trans 后特征），
    求均值得到每类原型；label >= old_cn 的进 prototypes_new，其余进
    prototypes_old。normalize=False 时保留原始尺度（USP 训练路径的调用
    方式；使用处自行 F.normalize）。

    返回 (prototypes_old, prototypes_new, prototypes_all)，
    无旧类时 prototypes_old 为空 tensor (0,)。
    """
    tg_model.eval()
    class_features = {}
    class_counts = {}

    for batch in trainloader:
        # BaseDataset_flag: (indexs, inputs, inputs_s, targets, flags, on_flags)
        indexs, inputs, inputs_s, targets, flags, on_flags = batch
        inputs, targets = inputs.to(device), targets.to(device)
        if len(inputs) == 1:
            continue
        outputs, raw_feats, feats, session_outputs = tg_model(inputs, return_feats=True)

        for i in range(len(targets)):
            label = targets[i].item()
            feature = feats[i]

            if label not in class_features:
                class_features[label] = torch.zeros_like(feature)
                class_counts[label] = 0

            class_features[label] += feature
            class_counts[label] += 1

    prototypes = []
    prototypes_new = []
    prototypes_old = []
    for label in sorted(class_features.keys()):
        class_mean = class_features[label] / class_counts[label]
        if normalize:
            class_mean = F.normalize(class_mean, p=2, dim=0)
        prototypes.append(class_mean)
        if label >= old_cn:
            prototypes_new.append(class_mean)
        else:
            prototypes_old.append(class_mean)

    if len(prototypes_old) == 0:
        prototypes_old = torch.tensor([])
    else:
        prototypes_old = torch.stack(prototypes_old, dim=0)

    if len(prototypes_new) == 0:
        prototypes_new = torch.tensor([])
    else:
        prototypes_new = torch.stack(prototypes_new, dim=0)

    prototypes = torch.stack(prototypes, dim=0)

    prototypes_old, prototypes_new, prototypes = (
        prototypes_old.to(device), prototypes_new.to(device), prototypes.to(device)
    )

    return prototypes_old, prototypes_new, prototypes


@torch.no_grad()
def fill_pro_list(pro_list, tg_model, val_loader, device, k: int, old_cn: int):
    """按分类器置信度为每个新类补充 top-k 无标注样本到样本池。

    移植自原 fill_pro_list（打印部分精简，选择逻辑逐行一致）：
    对 label in [old_cn, num_classes)，取该类 logits-softmax 置信度最高的
    k 个样本 index，从 dataset.data 追加到 pro_list[label]。
    """
    tg_model.eval()
    all_gt = []
    all_index = []
    all_outputs = []
    dataset = val_loader.dataset
    with torch.no_grad():
        for batch in val_loader:
            index, inputs, _, gt, _, _ = batch
            inputs = inputs.to(device)
            gt = gt.to(device)
            outputs, _, feats, _ = tg_model(inputs, return_feats=True)
            outputs = torch.softmax(outputs, dim=1)

            all_gt.extend(gt.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
            all_index.extend(index.cpu().numpy())

    all_gt = np.array(all_gt)
    all_outputs = np.array(all_outputs)
    all_index = np.array(all_index)

    for label in range(old_cn, all_outputs.shape[1]):
        class_confidences = all_outputs[:, label]
        top_k_indices = np.argsort(class_confidences)[-k:]
        selected_index = all_index[top_k_indices]
        pro_list[label] = np.concatenate((pro_list[label], dataset.data[selected_index]), axis=0)

    return pro_list


@torch.no_grad()
def compute_features(tg_feature_model, evalloader, num_samples: int, num_features: int, device):
    """特征抽取（iCaRL class-means 与 herding 共用）。"""
    tg_feature_model.eval()

    features = np.zeros([num_samples, num_features])
    start_idx = 0
    with torch.no_grad():
        for batch in evalloader:
            inputs = batch[1] if isinstance(batch, (tuple, list)) else batch[0]
            inputs = inputs.to(device)
            features[start_idx:start_idx + inputs.shape[0], :] = np.squeeze(
                tg_feature_model(inputs).data.cpu().numpy())
            start_idx = start_idx + inputs.shape[0]
    assert start_idx == num_samples
    return features


def select_exemplars_herding(prototypes_list, tg_feature_model, evalset_builder,
                             eval_batch_size: int, num_features: int, nb_protos_cl: int,
                             last_iter: int, nb_cl: int, session: int, device,
                             eval_batch_fetch=None):
    """iCaRL herding 选样（train_semi.py 主循环中 dr_herding 过程的收敛实现）。

    Args:
        prototypes_list: 每类的候选样本池（图像数组列表，可能被重复填充到等长）；
        tg_feature_model: 特征子模型（backbone 去掉最后 3 个 children）；
        evalset_builder: callable(data) -> Dataset（test 变换）；
        eval_batch_fetch: callable(loader) -> batch tuple（默认取 batch[1] 为输入）。
    Returns:
        alpha_dr_herding: list over sessions of (nb_cl, num_pool) 排序矩阵，
                          值为 1+iter_herding（选中序）或 0（未选中）。
    """
    import torch.utils.data as tud

    alpha_dr_herding = []
    dr_herding = []

    # 与原实现一致：先把本 session 各类候选池填充到等长
    start_idx = last_iter * nb_cl
    end_idx = (session + 1) * nb_cl
    max_length = max(len(prototypes_list[i]) for i in range(start_idx, end_idx))
    for i in range(start_idx, end_idx):
        lst = prototypes_list[i]
        extended_list = list(lst) * (max_length // len(lst)) + list(lst)[:max_length % len(lst)]
        prototypes_list[i] = np.array(extended_list)

    for iter_dico in range(last_iter * nb_cl, (session + 1) * nb_cl):
        evalset = evalset_builder(prototypes_list[iter_dico])
        evalloader = tud.DataLoader(evalset, batch_size=eval_batch_size,
                                    shuffle=False, num_workers=0)
        num_samples = len(evalset)
        mapped_prototypes = compute_features(tg_feature_model, evalloader, num_samples,
                                             num_features, device)
        D = mapped_prototypes.T
        D = D / np.linalg.norm(D, axis=0)

        herding = np.zeros(len(prototypes_list[iter_dico]), np.float32)
        dr_herding.append(herding)
        # Herding procedure : ranking of the potential exemplars
        mu = np.mean(D, axis=1)
        index2 = iter_dico % nb_cl
        dr_herding[index2] = dr_herding[index2] * 0
        w_t = mu
        iter_herding = 0
        iter_herding_eff = 0
        while not (np.sum(dr_herding[index2] != 0) == min(nb_protos_cl, 500)) and iter_herding_eff < 1000:
            tmp_t = np.dot(w_t, D)
            ind_max = np.argmax(tmp_t)
            iter_herding_eff += 1
            if dr_herding[index2][ind_max] == 0:
                dr_herding[index2][ind_max] = 1 + iter_herding
                iter_herding += 1
            w_t = w_t + mu - D[:, ind_max]

        if (iter_dico + 1) % nb_cl == 0:
            alpha_dr_herding.append(np.array(dr_herding))
            dr_herding = []

    return alpha_dr_herding
