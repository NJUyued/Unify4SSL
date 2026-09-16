"""USP SessionTrainer：类增量 session 外循环。

移植自 orig/USP4SSCL/train_semi.py（单骨干可扩张 fc 路径）+
utils_incremental/incremental_train_and_eval_semi.py 的训练循环。

session 语义（与原实现一致）：
    start_session = nb_cl_fg // nb_cl - 1
    for session in range(start_session, num_classes // nb_cl):
        类区间 = arange(session * nb_cl, (session + 1) * nb_cl)
    首个 session 的类区间覆盖 0..(start_session+1)*nb_cl（即整个 base session）。

每 session：
1. session > start_session 时：ref_model = deepcopy(tg_model)，fc 扩张
   out_features += nb_cl（旧权重原样拷贝）；
2. 数据重建：本 session k_shot 有标注 + 无标注池 + 历史 exemplar（herding
   选择）；testloader 为累积测试集；
3. optimizer/scheduler 重建（SGD momentum 0.9 + cosine warmup，
   num_training_steps = epochs * u_iter）；增量 session 用 new_lr/new_wd；
4. 训练循环（methods/usp.py 的损失）：每 epoch 末 update_proto（get_proto）；
5. exemplar 更新：fill_pro_list（无标注补充）+ iCaRL herding 选样；
6. 评估：累积 / base / novel 三档准确率 -> acc_matrix。

评估采用分类器 top-1（原 compute_accuracy_train 的 cnn_acc 路径，
即最终模型在累积测试集上的分类器精度；uad/etf/composite 等辅助精度
未纳入 acc_matrix——原 train_semi.py 报告的是 uad_acc，但其依赖
flip 重复前向恒等（原代码第二遍用同一 transform，D2==D），
故此处取语义等价的分类器精度作为主指标）。

用法：
    from sslunify.cl import SessionTrainer
    trainer = SessionTrainer(cfg)   # cfg: argparse.Namespace 或属性可访问对象
    result = trainer.train()        # {'acc_matrix', 'avg_acc', 'final_acc'}
"""

from __future__ import annotations

import copy
import os
import types
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

from ..core.optim import get_optimizer
from ..core.schedulers import get_cosine_schedule_with_warmup
from ..methods import usp
from .data import (BaseDataset, BaseDataset_flag, UnlabelDataset,
                   get_cifar_session_data, get_imagenet100_session_data)
from .dist_align import DistAlignQueueHook
from .memory import (FeatureMemBank, fill_pro_list, get_proto,
                     select_exemplars_herding)
from .nets import build_cl_net, feature_model

# ------------------------------------------------------------------ 默认配置
# 与 run_cifar.sh（cifar100 主配置）一致的默认值。

_DEFAULTS = dict(
    dataset="cifar100",
    num_classes=None,          # 缺省按 dataset 推断
    image_size=None,           # 缺省按 dataset 推断（cifar 32 / 其他 224）
    data_dir="./data",
    k_shot=30,
    nb_cl_fg=10,
    nb_cl=10,
    nb_protos=5,
    u_ratio=7,
    epochs=200,
    epochs_new=200,
    warmup_epochs=10,
    lr=0.03,
    new_lr=0.03,
    batch_size=64,
    eval_batch_size=256,
    u_iter=100,
    model="resnet32",
    proto_dim=64,              # class_means 的维度（原 args.proto_dim）
    dim=512,                   # ETF / trans 维度（原 args.dim）
    p_cutoff=0.95,
    q_cutoff=0.25,
    T=2.0,
    beta=0.25,
    buffer_size=5120,
    seed=1,
    device=None,               # None -> 自动
    weight_decay=5e-4,
    weight_decay_new=2e-4,
    momentum=0.9,
    include_unlabel=True,
    update_proto=True,
    use_conloss=True,
    kd_only_old=True,
    kd_mode="logits",
    ulb_kd_mode="similarity",
    use_ulb_kd=True,
    use_ulb_aug=True,
    use_hard_labels=True,
    no_use_conloss_on_ulb=False,
    unlabels_predict_mode="cosine",
    flip_on_means=True,
    lambda_ce=1.0,
    lambda_kd=1.0,
    lambda_con=1.0,
    lambda_cons=1.0,
    lambda_ukd=1.0,
    percentage=0.01,           # imagenet100 有标注比例
    unlabeled_num=-1,
    save_dir="./saved_models",
    save_name="usp",
    num_workers=4,
    save_checkpoint=False,     # 原实现每 session torch.save 整模型；默认关闭
)

_DATASET_INFO = {
    "cifar100": dict(num_classes=100, image_size=32),
    "cifar10": dict(num_classes=10, image_size=32),
    "imagenet100": dict(num_classes=100, image_size=224),
}


class _Cfg:
    """属性访问包装：dict/Namespace -> 带默认值的 cfg。"""

    def __init__(self, cfg):
        base = dict(_DEFAULTS)
        if isinstance(cfg, dict):
            src = cfg
        else:
            src = {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_") and not callable(getattr(cfg, k))}
        unknown = set(src) - set(base)
        if unknown:
            # 容忍额外字段（如 tools 传入的公共配置），仅覆盖已知字段
            pass
        base.update({k: v for k, v in src.items() if k in base})
        info = _DATASET_INFO[base["dataset"]]
        if base["num_classes"] is None:
            base["num_classes"] = info["num_classes"]
        if base["image_size"] is None:
            base["image_size"] = info["image_size"]
        if base["device"] is None:
            base["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        assert base["nb_cl_fg"] % base["nb_cl"] == 0
        assert base["nb_cl_fg"] >= base["nb_cl"]
        self._d = base

    def __getattr__(self, name):
        try:
            return self._d[name]
        except KeyError:
            raise AttributeError(name)

    def __getitem__(self, name):
        return self._d[name]


class SessionTrainer:
    """USP 半监督持续学习训练器（session 外循环 + 每 session 训练循环）。"""

    def __init__(self, cfg):
        self.cfg = _Cfg(cfg)
        self.device = torch.device(self.cfg.device)
        self.print_fn = print

        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

        c = self.cfg
        # 类顺序（原实现 order = np.arange(num_classes)）
        self.order = np.arange(c.num_classes)

        # 样本池（每类图像数组；训练后 fill_pro_list 补充无标注样本）
        self.prototypes = [[] for _ in range(c.num_classes)]
        self.prototypes_flag = [[] for _ in range(c.num_classes)]
        self.prototypes_on_flag = [[] for _ in range(c.num_classes)]

        self.start_session = int(c.nb_cl_fg / c.nb_cl) - 1
        self.num_sessions = int(c.num_classes / c.nb_cl) - self.start_session

        # 累积数据
        self.X_train_cumuls, self.Y_train_cumuls = [], []
        self.X_valid_cumuls, self.Y_valid_cumuls = [], []
        self.X_valid_cumuls_base, self.Y_valid_cumuls_base = [], []
        self.X_valid_cumul_novel, self.Y_valid_cumul_novel = [], []
        self.X_protoset_cumuls, self.Y_protoset_cumuls = [], []

        # ETF 锚：(dim, num_classes) -> 转置为 (num_classes, dim)
        from .etf import generate_etf_vector

        etf_vec = generate_etf_vector(c.dim, c.num_classes)
        self.text_anchor = torch.tensor(etf_vec.T).to(self.device)

        self.tg_model = None
        self.ref_model = None
        self.acc_matrix = np.zeros((self.num_sessions, self.num_sessions))

    # ------------------------------------------------------------------ data

    def _session_data(self, session):
        c = self.cfg
        class_index = np.arange(session * c.nb_cl, (session + 1) * c.nb_cl)
        if c.dataset in ("cifar100", "cifar10"):
            X_train, Y_train, u_data, u_gt, X_valid, Y_valid = get_cifar_session_data(
                c.data_dir, c.dataset, class_index, c.k_shot)
        elif c.dataset == "imagenet100":
            X_train, Y_train, u_data, u_gt = get_imagenet100_session_data(
                c.data_dir, class_index, percentage=c.percentage, k_shot=c.k_shot, train=True)
            X_valid, Y_valid = get_imagenet100_session_data(
                c.data_dir, class_index, train=False)
        else:
            raise ValueError(f"dataset {c.dataset} not supported")
        return X_train, Y_train, u_data, u_gt, X_valid, Y_valid

    def _make_dataset(self, data, targets, kind, flags=None, on_flags=None):
        c = self.cfg
        if kind == "train_flag":
            ds = BaseDataset_flag("train", c.image_size, dataset=c.dataset)
            ds.data, ds.targets = data, targets
            ds.flags = np.asarray(flags, dtype=int) if flags is not None else np.ones(len(data), dtype=int)
            ds.on_flags = np.asarray(on_flags, dtype=int) if on_flags is not None else np.ones(len(data), dtype=int)
        elif kind == "test":
            ds = BaseDataset("test", c.image_size, dataset=c.dataset)
            ds.data, ds.targets = data, targets
        elif kind == "unlabel":
            ds = UnlabelDataset(c.image_size, unlabeled_num=c.unlabeled_num, dataset=c.dataset)
            ds.data, ds.targets = data, targets
        else:
            raise ValueError(kind)
        return ds

    # ------------------------------------------------------------------ train

    def train(self):
        c = self.cfg
        for session in range(self.start_session, int(c.num_classes / c.nb_cl)):
            self._run_session(session)
        return {
            "acc_matrix": self.acc_matrix,
            "avg_acc": float(self.acc_matrix.sum(axis=0).mean() / self.num_sessions),
            "final_acc": float(self.acc_matrix[-1].mean()),
        }

    def _run_session(self, session):
        c = self.cfg
        s_idx = session - self.start_session  # 0-based session 序号

        # ---------------- 模型：新建 / fc 扩张 + 旧模型快照 ----------------
        if session == self.start_session:
            self.tg_model = build_cl_net(c.model, num_classes=(session + 1) * c.nb_cl,
                                         dim=c.dim)
            self.ref_model = None
            last_iter = 0
        else:
            last_iter = session
            self.ref_model = copy.deepcopy(self.tg_model)
            in_features = self.tg_model.fc.in_features
            out_features = self.tg_model.fc.out_features
            new_fc = nn.Linear(in_features, out_features + c.nb_cl)
            new_fc.weight.data[:out_features] = self.tg_model.fc.weight.data
            new_fc.bias.data[:out_features] = self.tg_model.fc.bias.data
            self.tg_model.fc = new_fc

        self.tg_model = self.tg_model.to(self.device)
        if self.ref_model is not None:
            self.ref_model = self.ref_model.to(self.device)

        # ---------------- 本 session 数据 ----------------
        X_train, Y_train, unlabeled_data, unlabeled_gt, X_valid, Y_valid = \
            self._session_data(session)

        # 样本池登记（本 session 类；k_shot 有标注来源 flag=1）
        for orde in range(session * c.nb_cl, (session + 1) * c.nb_cl):
            self.prototypes[orde] = X_train[np.where(Y_train == orde)]
            self.prototypes_flag[orde] = np.ones(len(self.prototypes[orde]), dtype=int)
            if orde < c.nb_cl_fg:
                self.prototypes_on_flag[orde] = np.ones(len(self.prototypes[orde]), dtype=int)
            else:
                self.prototypes_on_flag[orde] = np.zeros(len(self.prototypes[orde]), dtype=int)

        self.X_train_cumuls.append(X_train)
        self.Y_train_cumuls.append(Y_train)
        self.X_valid_cumuls.append(X_valid)
        self.Y_valid_cumuls.append(Y_valid)

        # ---------------- 有标注训练集 = 本 session 新样本 + 历史 exemplar ----------------
        if session == self.start_session:
            X_flag = np.ones(len(X_train), dtype=int)
            X_on_flag = np.ones(len(X_train), dtype=int)
            self.X_valid_cumuls_base.append(X_valid)
            self.Y_valid_cumuls_base.append(Y_valid)
        else:
            if len(self.X_protoset_cumuls):
                X_protoset = np.concatenate(self.X_protoset_cumuls)
                Y_protoset = np.concatenate(self.Y_protoset_cumuls)
            else:
                X_protoset = np.zeros((0,) + X_train.shape[1:], dtype=X_train.dtype)
                Y_protoset = np.zeros(0, dtype=Y_train.dtype)
            X_flag = np.concatenate((np.ones(len(X_protoset), dtype=int),
                                     np.ones(len(X_train), dtype=int)))
            X_on_flag = np.concatenate((np.ones(len(X_protoset), dtype=int),
                                        np.ones(len(X_train), dtype=int)))
            X_train = np.concatenate((X_train, X_protoset), axis=0)
            Y_train = np.concatenate((Y_train, Y_protoset))

            if len(self.X_valid_cumul_novel):
                self.X_valid_cumuls_base.append(self.X_valid_cumul_novel)
                self.Y_valid_cumuls_base.append(self.Y_valid_cumul_novel)
            self.X_valid_cumul_novel = X_valid
            self.Y_valid_cumul_novel = Y_valid

        X_valid_cumul = np.concatenate(self.X_valid_cumuls)
        Y_valid_cumul = np.concatenate(self.Y_valid_cumuls)

        # ---------------- dataloader ----------------
        trainset = self._make_dataset(X_train, Y_train, "train_flag", flags=X_flag, on_flags=X_on_flag)
        batch_size = min(c.batch_size, len(trainset))
        sampler_x = tud.RandomSampler(trainset, replacement=True,
                                      num_samples=c.u_iter * batch_size)
        batch_sampler_x = tud.BatchSampler(sampler_x, batch_size, drop_last=True)
        trainloader = tud.DataLoader(trainset, batch_sampler=batch_sampler_x,
                                     num_workers=c.num_workers)

        testset = self._make_dataset(X_valid_cumul, Y_valid_cumul, "test")
        testloader = tud.DataLoader(testset, batch_size=c.eval_batch_size,
                                    shuffle=False, num_workers=c.num_workers)

        if c.include_unlabel and unlabeled_data is not None and len(unlabeled_data) > 0:
            ssl_trainset = self._make_dataset(unlabeled_data, unlabeled_gt, "unlabel")
            ssl_trainloader = tud.DataLoader(ssl_trainset,
                                             batch_size=c.u_ratio * batch_size,
                                             shuffle=True, num_workers=c.num_workers)
            # fill_pro_list 专用 loader：6 元组 (index, w, s, gt, flag, on_flag) +
            # test 相位变换（原 train_semi.py:495-500 的 BaseDataset_flag("test") 路径）
            fill_set = BaseDataset_flag("test", c.image_size, dataset=c.dataset)
            fill_set.data, fill_set.targets = unlabeled_data, unlabeled_gt
            fill_set.flags = np.ones(len(unlabeled_data), dtype=int)
            fill_set.on_flags = np.ones(len(unlabeled_data), dtype=int)
            fill_loader = tud.DataLoader(fill_set, batch_size=c.eval_batch_size,
                                         shuffle=False, num_workers=c.num_workers)
        else:
            ssl_trainloader = None
            fill_loader = None

        # ---------------- optimizer / scheduler ----------------
        if session > self.start_session:
            base_lr, epochs, wd = c.new_lr, c.epochs_new, c.weight_decay_new
        else:
            base_lr, epochs, wd = c.lr, c.epochs, c.weight_decay

        optimizer = get_optimizer(self.tg_model, "sgd", lr=base_lr,
                                  momentum=c.momentum, weight_decay=wd,
                                  nesterov=False, bn_wd_skip=False)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_training_steps=epochs * c.u_iter,
            num_warmup_steps=c.warmup_epochs * c.u_iter)

        print("Batch of classes number {0} arrives ... (session {1}, classes {2}-{3})".format(
            session, s_idx, session * c.nb_cl, (session + 1) * c.nb_cl))

        # ---------------- 训练 ----------------
        self._train_one_session(
            session=session, epochs=epochs, trainloader=trainloader,
            testloader=testloader, ssl_trainloader=ssl_trainloader,
            optimizer=optimizer, scheduler=scheduler,
            unlabeled_data=unlabeled_data)

        # ---------------- exemplar 更新（fill + herding） ----------------
        if fill_loader is not None:
            print("Filling buffer...")
            fill_pro_list(self.prototypes, self.tg_model, fill_loader,
                          self.device, c.k_shot, session * c.nb_cl)

        print("Updating exemplar set...")
        self._update_exemplars(session, last_iter)

        # ---------------- 评估：累积 / base / novel ----------------
        accs = self._evaluate_all(session, X_valid_cumul, Y_valid_cumul)
        for j, acc in enumerate(accs):
            self.acc_matrix[s_idx, j] = acc
        print(f"[session {s_idx}] acc_matrix row: {self.acc_matrix[s_idx].tolist()}")

        if c.save_checkpoint:
            os.makedirs(os.path.join(c.save_dir, c.save_name), exist_ok=True)
            path = os.path.join(c.save_dir, c.save_name, f"model_session_{s_idx}.pth")
            torch.save(self.tg_model.state_dict(), path)

    # ------------------------------------------------------------------ 训练循环

    def _train_one_session(self, session, epochs, trainloader, testloader,
                           ssl_trainloader, optimizer, scheduler, unlabeled_data):
        """incremental_train_and_eval 的训练循环部分（逐损失忠实移植）。"""
        c = self.cfg
        N = 128
        old_cn = session * c.nb_cl
        total_cn = (session + 1) * c.nb_cl
        is_incremental = session > self.start_session

        include_unlabel = c.include_unlabel and ssl_trainloader is not None

        # 记忆库（USP 状态流的一部分；ref_bank 仅增量 session 使用）
        mem_bank = FeatureMemBank(c.dim, len(trainloader.dataset), self.device)

        if is_incremental:
            self.ref_model.eval()
            num_old_classes = self.ref_model.fc.out_features
            assert num_old_classes == old_cn
            prototypes_ref_old, prototypes_ref_new, prototypes_ref = get_proto(
                trainloader, self.ref_model, old_cn, self.device, False)
        else:
            prototypes_ref_old = torch.tensor([]).to(self.device)
            prototypes_ref = torch.tensor([]).to(self.device)

        # USB 式分布对齐（对齐目标 = 本 session nb_cl 类的 uniform）
        distri = DistAlignQueueHook(num_classes=c.nb_cl, queue_length=N, p_target_type="uniform")

        if include_unlabel:
            ssl_iterator = iter(ssl_trainloader)

        prototypes_old, prototypes_new, pro = get_proto(trainloader, self.tg_model,
                                                        old_cn, self.device, False)

        for epoch in range(epochs):
            self.tg_model.train()
            total, correct = 0, 0
            train_loss = 0.0
            batch_idx = -1
            skip = False

            for batch_idx, (indexs, inputs, inputs_s, targets, flags, on_flags) in enumerate(trainloader):
                optimizer.zero_grad()
                indexs = indexs.to(self.device)
                inputs, inputs_s = inputs.to(self.device), inputs_s.to(self.device)
                targets = targets.to(self.device)

                num_lb = len(targets)
                if num_lb == 1:
                    continue

                outputs, raw_feats, feats, session_outputs = self.tg_model(inputs, return_feats_list=True)
                outputs_s, raw_feats_s, feats_s, session_outputs_s = self.tg_model(inputs_s, return_feats=True)
                mem_bank.update_bank(feats, targets, indexs)

                # 有标注 CE（分类器）
                suploss_lb = nn.CrossEntropyLoss(None)(outputs, targets.long())

                # FSR：有标注
                if c.use_conloss:
                    conloss_lb = usp.fsr_loss(feats, targets, self.text_anchor)
                else:
                    conloss_lb = torch.tensor(0.0).to(self.device)

                # 有标注 KD（仅增量 session）
                if is_incremental:
                    ref_outputs, ref_raw_feats, ref_feats, ref_session_outputs = \
                        self.ref_model(inputs, return_feats_list=True)
                    old_mask = targets < num_old_classes
                    if c.kd_mode == "logits":
                        suploss_kd = usp.labeled_kd_loss(
                            outputs, ref_outputs, targets, num_old_classes,
                            T=c.T, beta=c.beta, kd_only_old=c.kd_only_old)
                    elif c.kd_mode == "feats":
                        suploss_kd = usp.labeled_kd_feats_loss(
                            feats, ref_feats, targets, num_old_classes,
                            kd_only_old=c.kd_only_old)
                    else:
                        raise ValueError(f"kd_mode: {c.kd_mode} not supported")
                else:
                    suploss_kd = torch.tensor(0.0).to(self.device)
                    old_mask = targets < 0  # 空 mask

                # ---------------- 无标注分支 ----------------
                if include_unlabel and epoch >= c.warmup_epochs:
                    skip = False
                    try:
                        inputs_ulb, inputs_s_ulb, gt = next(ssl_iterator)
                    except StopIteration:
                        ssl_iterator = iter(ssl_trainloader)
                        inputs_ulb, inputs_s_ulb, gt = next(ssl_iterator)

                    num_ulb = len(gt)
                    if num_ulb == 1:
                        skip = True
                        continue

                    inputs_ulb, inputs_s_ulb = inputs_ulb.to(self.device), inputs_s_ulb.to(self.device)
                    gt = gt.to(self.device)

                    outputs_ulb, raw_feats_ulb, feats_ulb, session_outputs_ulb = \
                        self.tg_model(inputs_ulb, return_feats=True)
                    outputs_s_ulb, raw_feats_s_ulb, feats_s_ulb, session_outputs_s_ulb = \
                        self.tg_model(inputs_s_ulb, return_feats=True)
                    feats_ulb = F.normalize(feats_ulb, p=2, dim=1)
                    feats_s_ulb = F.normalize(feats_s_ulb, p=2, dim=1)

                    # DCP 高置信路径（分类器 + DA + 阈值）
                    consloss_ulb, mask, n_mask, predicted_classes, max_probs = \
                        usp.dcp_high_confidence_path(
                            outputs_ulb, outputs_s_ulb, distri, old_cn, total_cn,
                            c.p_cutoff, c.q_cutoff)

                    # FSR：无标注
                    if not c.no_use_conloss_on_ulb:
                        conloss_ulb = usp.fsr_loss_ulb(feats_ulb, predicted_classes,
                                                       mask, self.text_anchor)
                    else:
                        conloss_ulb = torch.tensor(0.0).to(self.device)

                    # CUD：无标注蒸馏
                    if is_incremental and c.use_ulb_kd:
                        if c.ulb_kd_mode == "logits":
                            ref_outputs_ulb = self.ref_model(inputs_ulb)
                            suploss_kd_ulb = usp.cud_logits_loss(
                                outputs_ulb, ref_outputs_ulb.detach(), num_old_classes,
                                c.T, num_ulb)
                        elif c.ulb_kd_mode == "feats":
                            _, ref_raw_feats_ulb, _, _ = self.ref_model(inputs_ulb, return_feats=True)
                            suploss_kd_ulb = usp.cud_feats_loss(raw_feats_ulb, ref_raw_feats_ulb)
                        elif c.ulb_kd_mode == "similarity":
                            _, _, ref_feats_ulb, _ = self.ref_model(inputs_s_ulb, return_feats=True)
                            if old_mask.sum() > 0:
                                suploss_kd_ulb = usp.cud_similarity_loss(
                                    ref_feats, feats, ref_feats_ulb, feats_ulb, prototypes_ref)
                            else:
                                suploss_kd_ulb = torch.tensor(0.0).to(self.device)
                        else:
                            raise ValueError(f"ulb_kd_mode: {c.ulb_kd_mode} not supported")
                    else:
                        suploss_kd_ulb = torch.tensor(0.0).to(self.device)

                    # DCP 低置信路径（原型）
                    if is_incremental and c.use_ulb_aug and epoch != 0:
                        consloss_ulb_aug, q_predict_class = usp.dcp_low_confidence_path(
                            feats_ulb, outputs_s_ulb, prototypes_old, prototypes_new,
                            old_cn, mask)
                    else:
                        consloss_ulb_aug = torch.tensor(0.0).to(self.device)

                    # 总损失
                    loss = usp.total_loss(
                        suploss_lb, suploss_kd, conloss_lb, conloss_ulb,
                        consloss_ulb, consloss_ulb_aug, suploss_kd_ulb,
                        lambda_ce=c.lambda_ce, lambda_kd=c.lambda_kd,
                        lambda_con=c.lambda_con, lambda_cons=c.lambda_cons,
                        lambda_ukd=c.lambda_ukd)
                else:
                    # warmup 前或无无标注数据
                    loss = (suploss_lb + c.lambda_kd * suploss_kd + c.lambda_con * conloss_lb)

                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                assert torch.isfinite(loss), f"loss is not finite at session {session} epoch {epoch}"

            # 每 epoch 更新原型
            if c.update_proto:
                prototypes_old, prototypes_new, pro = get_proto(
                    trainloader, self.tg_model, old_cn, self.device, False)

            test_loss, test_acc, test_old_acc, test_new_acc = usp.validate(
                self.tg_model, testloader, self.device, None, old_cn)
            if epoch % 10 == 0 or epoch == epochs - 1:
                print("Epoch: {}, Loss: {:.4f}, Train Acc: {:.4f}, Test Acc: {:.4f} "
                      "(old {:.4f} / new {:.4f})".format(
                          epoch, train_loss / (batch_idx + 1), 100.0 * correct / max(total, 1),
                          test_acc, test_old_acc, test_new_acc))

    # ------------------------------------------------------------------ exemplar

    def _update_exemplars(self, session, last_iter):
        """训练后 exemplar 选择：iCaRL herding（train_semi.py 主循环 686-746 行）。"""
        c = self.cfg
        nb_protos_cl = c.buffer_size // ((session + 1) * c.nb_cl)
        tg_feature_model = feature_model(self.tg_model).to(self.device)
        num_features = self.tg_model.fc.in_features

        def evalset_builder(data):
            ds = BaseDataset("test", c.image_size, dataset=c.dataset)
            ds.data = data
            ds.targets = np.zeros(len(ds))
            return ds

        alpha_dr_herding = select_exemplars_herding(
            self.prototypes, tg_feature_model, evalset_builder,
            eval_batch_size=min(c.eval_batch_size, 256), num_features=num_features,
            nb_protos_cl=nb_protos_cl, last_iter=last_iter, nb_cl=c.nb_cl,
            session=session, device=self.device)

        # 原实现 alpha_dr_herding 为跨 session 持久累积（外层列表）：
        # 每 session 只对 last_iter*nb_cl..(session+1)*nb_cl 的类重跑 herding
        # 并 append 一个 (nb_cl, pool) 数组；历史 session 的选择序沿用上次结果。
        if not hasattr(self, "_alpha_dr_herding") or self._alpha_dr_herding is None:
            self._alpha_dr_herding = alpha_dr_herding
        else:
            self._alpha_dr_herding = self._alpha_dr_herding + alpha_dr_herding

        X_protoset_cumuls, Y_protoset_cumuls = [], []
        for iteration2 in range(session + 1):
            for iter_dico in range(c.nb_cl):
                cl = iteration2 * c.nb_cl + iter_dico
                alph = self._alpha_dr_herding[iteration2][iter_dico]
                alph = (alph > 0) * (alph < nb_protos_cl + 1) * 1.0
                X_protoset_cumuls.append(self.prototypes[cl][np.where(alph == 1)[0]])
                Y_protoset_cumuls.append(
                    self.order[cl] * np.ones(len(np.where(alph == 1)[0])))

        # 拼接（过滤空类）
        data_parts = [p for p in X_protoset_cumuls if len(p) > 0]
        label_parts = [p for p in Y_protoset_cumuls if len(p) > 0]
        if data_parts:
            self.X_protoset_cumuls = data_parts
            self.Y_protoset_cumuls = label_parts
        else:
            self.X_protoset_cumuls, self.Y_protoset_cumuls = [], []

    # ------------------------------------------------------------------ eval

    @torch.no_grad()
    def _eval_top1(self, data, targets):
        testset = self._make_dataset(data, targets, "test")
        loader = tud.DataLoader(testset, batch_size=self.cfg.eval_batch_size,
                                 shuffle=False, num_workers=self.cfg.num_workers)
        _, acc, _, _ = usp.validate(self.tg_model, loader, self.device, None, 0)
        return acc

    def _evaluate_all(self, session, X_valid_cumul, Y_valid_cumul):
        """返回从 session 0 到当前的评估序列（供 acc_matrix 行填充）。

        acc_matrix[s_idx, j] = 在 session j 结束时的累积测试集（前 (j+1)*nb_cl 类
        中属于已见类的部分）上的 top-1。原实现每 session 只在当前累积测试集上
        评估一次并打印；矩阵语义为标准的持续学习协议（旧模型列在后面 session
        重测），此处按每 session 的三档评估 + 对历史 session 的累积子集重测。
        """
        c = self.cfg
        accs = []
        self.tg_model.eval()

        for j in range(session - self.start_session + 1):
            # 第 j 个 session 的累积测试集 = 前 (j+1) 个 session 的测试数据
            data = np.concatenate(self.X_valid_cumuls[: j + 1])
            labels = np.concatenate(self.Y_valid_cumuls[: j + 1])
            accs.append(self._eval_top1(data, labels))

        return accs
