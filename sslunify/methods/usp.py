"""USP（ICCV 2025）半监督持续学习方法实现。

移植自 orig/USP4SSCL/utils_incremental/incremental_train_and_eval_semi.py
的 incremental_train_and_eval（train_semi.py 单骨干路径）。方法为
Divide-and-Conquer 三件套 + 有标注旧类 KD：

- FSR（Feature-Space Regularization，use_conloss 分支）：
    labeled  : scores = F.linear(F.normalize(feats), F.normalize(text_anchor)) / 0.1
              conloss_lb = CE(scores, targets)
    unlabeled: feats_ulb 已归一化，scores = F.linear(feats_ulb, F.normalize(text_anchor)) / 0.1
              conloss_ulb = CE(scores, predicted_classes, none) * mask 后 .mean()
  注意原实现 feats 是 trans 后的 con_feats（forward 返回四元组的第 3 个）。

- DCP 高置信路径（分类器路径）：
    pseudo_label = softmax(outputs_ulb[:, old_cn:total_cn])
    pseudo_label = distri.dist_align(pseudo_label.detach())   # USB 队列 DA
    mask = max_probs.ge(p_cutoff)；n_mask = max_probs.le(q_cutoff)
    predicted_classes = argmax + old_cn
    consloss_ulb = ce_loss(outputs_s_ulb, predicted_classes, True, 'none') * mask -> mean()

- DCP 低置信路径（原型路径，use_ulb_aug 且 epoch != 0 且为增量 session）：
    prototypes = cat([prototypes_old, prototypes_new])
    q_cosine_scores = F.linear(feats_ulb, F.normalize(prototypes)) / 0.1
    q_predict_class = argmax(softmax(q_cosine_scores))
    consloss_ulb_aug = ce_loss(outputs_s_ulb, q_predict_class, True, 'none')
                        * (1 - mask) -> mean()

- CUD（无标注蒸馏，ulb_kd_mode='similarity'）：
    ref_feats_ulb = ref_model(inputs_s_ulb) 的 con_feats
    teacher: normalized(cat(ref_feats, ref_feats_ulb)) @ normalize(prototypes_ref).T
             -> softmax(/0.1)
    student: normalized(cat(feats, feats_ulb)) @ 同一 prototypes_ref.T
             -> log_softmax(/0.1)
    suploss_kd_ulb = sum(-teacher.detach() * student, dim=1).mean()
    仅当当前 batch 存在旧类样本（old_mask.sum() > 0）时计算。
    （feats/ref_feats 为当前 batch 有标注样本的 con_feats）

- 有标注 KD（kd_mode='logits'，kd_only_old=True）：
    仅旧类样本（targets < num_old_classes）：
    KLDiv(log_softmax(outputs[old][:, :old]/T), softmax(ref_outputs[old].detach()/T))
    * T^2 * beta * num_old_classes

- 总损失（增量 session、无标注分支启用后）：
    loss = λ_ce·suploss_lb + λ_kd·suploss_kd + λ_con·(conloss_lb + conloss_ulb)
           + λ_cons·(consloss_ulb + consloss_ulb_aug) + λ_ukd·suploss_kd_ulb
  （首 session 或 warmup 前：suploss_lb + λ_kd·suploss_kd + λ_con·conloss_lb）

各损失的行号锚点（incremental_train_and_eval_semi.py）：
FSR labeled 174-177 / FSR unlabeled 258-261 / DCP-high 230-242 /
CUD similarity 312-334 / DCP-low 345-371 / labeled-KD 186-195 / 总损失 375。

训练循环（epoch/batch 双层、update_bank、get_proto 每 epoch 更新、
DistAlign 队列、num_lb==1 跳过等）在 cl/session.py 的 SessionTrainer 中。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.losses import ce_loss


def fsr_loss(feats, targets, text_anchor):
    """FSR 有标注分支：ETF 锚余弦分类 CE。

    feats: (B, dim) trans 后特征（未归一化）；text_anchor: (num_classes, dim)。
    """
    scores = F.linear(F.normalize(feats, p=2, dim=1), F.normalize(text_anchor, p=2, dim=1)) / 0.1
    return F.cross_entropy(scores, targets.long())


def fsr_loss_ulb(feats_ulb, pseudo_classes, mask, text_anchor):
    """FSR 无标注分支：feats_ulb 已归一化，mask 为高置信掩码。"""
    scores = F.linear(feats_ulb, F.normalize(text_anchor, p=2, dim=1)) / 0.1
    conloss_ulb = F.cross_entropy(scores, pseudo_classes.long(), reduction="none") * mask
    return conloss_ulb.mean()


def dcp_high_confidence_path(outputs_ulb, outputs_s_ulb, distri, old_cn, total_cn,
                             p_cutoff, q_cutoff):
    """DCP 高置信分类器路径。

    Returns:
        (consloss_ulb, mask, n_mask, predicted_classes, max_probs)
        predicted_classes 已加 old_cn。
    """
    pseudo_label = torch.softmax(outputs_ulb[:, old_cn:total_cn], dim=-1)
    # DA
    pseudo_label = distri.dist_align(probs_x_ulb=pseudo_label.detach())
    max_probs, predicted_classes = torch.max(pseudo_label, dim=-1)
    mask = max_probs.ge(p_cutoff).float()
    n_mask = max_probs.le(q_cutoff).float()

    predicted_classes = predicted_classes + old_cn
    consloss_ulb = ce_loss(outputs_s_ulb, predicted_classes, True, reduction="none") * mask
    consloss_ulb = consloss_ulb.mean()
    return consloss_ulb, mask, n_mask, predicted_classes, max_probs


def dcp_low_confidence_path(feats_ulb, outputs_s_ulb, prototypes_old, prototypes_new,
                            old_cn, mask):
    """DCP 低置信原型路径。

    feats_ulb: (B, 64) 已归一化的池化特征（原型所在空间）；
    prototypes_old/new: 该空间内逐类均值原型（未归一化，使用处归一化）。
    consloss_ulb_aug = CE(outputs_s_ulb, q_predict_class) * (1-mask) -> mean()
    """
    prototypes = torch.cat([prototypes_old, prototypes_new], dim=0)
    q_cosine_scores = F.linear(feats_ulb, F.normalize(prototypes, p=2, dim=1)) / 0.1

    q_pseudo_label = torch.softmax(q_cosine_scores, dim=1)
    q_predict_class = q_pseudo_label.max(1)[1]

    consloss_ulb_aug = ce_loss(outputs_s_ulb, q_predict_class, True, reduction="none") \
        * torch.logical_not(mask.bool()).float()
    consloss_ulb_aug = consloss_ulb_aug.mean()
    return consloss_ulb_aug, q_predict_class


def cud_similarity_loss(ref_feats, feats, ref_feats_ulb, feats_ulb, prototypes_ref):
    """CUD：无标注蒸馏 similarity 模式（原实现 312-334 行）。

    teacher 输入 = cat(当前 batch 有标注的 ref con_feats, ref 无标注强视图 con_feats)
    student 输入 = cat(当前 batch 有标注的 con_feats, 无标注强视图 con_feats)
    概率均对 normalize(prototypes_ref) 计算余弦 logits / 0.1。
    返回标量损失（old_mask.sum()==0 时由调用方跳过）。
    """
    normalized_ref_feats_ulb = F.normalize(torch.cat((ref_feats, ref_feats_ulb)), p=2, dim=1)

    prototypes_ref = F.normalize(prototypes_ref, p=2, dim=1)

    teacher_logits = normalized_ref_feats_ulb @ prototypes_ref.T
    teacher_prob = F.softmax(teacher_logits / 0.1, dim=1)
    student_logits = F.normalize(torch.cat((feats, feats_ulb)), p=2, dim=1) @ prototypes_ref.T
    student_prob = F.log_softmax(student_logits / 0.1, dim=1)

    assert teacher_prob.size() == student_prob.size()
    suploss_kd_ulb = torch.sum(-teacher_prob.detach() * student_prob, dim=1).mean() * 1
    return suploss_kd_ulb


def cud_logits_loss(outputs_ulb, ref_outputs_ulb, num_old_classes, T, num_ulb):
    """CUD logits 模式（原实现 268-280 行，供 ulb_kd_mode='logits' 使用）。"""
    ref_predicted_classes = ref_outputs_ulb.max(1)[1].reshape(-1)
    gt_mask = torch.zeros_like(ref_outputs_ulb).scatter_(1, ref_predicted_classes.unsqueeze(1), 1).bool()
    pred_teacher_part2 = F.softmax(ref_outputs_ulb / T - 1000.0 * gt_mask, dim=1)
    log_pred_student_part2 = F.log_softmax(outputs_ulb[:, :num_old_classes] / T - 1000.0 * gt_mask, dim=1)
    return (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction="sum")
        * (T ** 2)
        / num_ulb
    )


def cud_feats_loss(raw_feats_ulb, ref_raw_feats_ulb):
    """CUD feats 模式（原实现 282-284 行）：骨干特征 MSE。"""
    return F.mse_loss(raw_feats_ulb, ref_raw_feats_ulb.detach())


def labeled_kd_loss(outputs, ref_outputs, targets, num_old_classes, T=2.0, beta=0.25,
                    kd_only_old=True):
    """有标注旧类 KD（kd_mode='logits'）。

    kd_only_old=True 时仅旧类样本（原实现 187-192 行），batch 内无旧类时返回 0。
    """
    old_mask = targets < num_old_classes
    if kd_only_old:
        if old_mask.sum() > 0:
            return nn.KLDivLoss(reduction="batchmean")(
                F.log_softmax(outputs[old_mask][:, :num_old_classes] / T, dim=1),
                F.softmax(ref_outputs[old_mask].detach() / T, dim=1)
            ) * T * T * beta * num_old_classes
        return torch.tensor(0.0).to(outputs.device)
    return nn.KLDivLoss(reduction="batchmean")(
        F.log_softmax(outputs[:, :num_old_classes] / T, dim=1),
        F.softmax(ref_outputs.detach() / T, dim=1)
    ) * T * T * beta * num_old_classes


def labeled_kd_feats_loss(feats, ref_feats, targets, num_old_classes, kd_only_old=True):
    """有标注 KD 的 feats 模式（原实现 197-204 行）。"""
    old_mask = targets < num_old_classes
    if kd_only_old:
        if old_mask.sum() > 0:
            return F.mse_loss(feats[old_mask], ref_feats[old_mask].detach()) * 1e3
        return torch.tensor(0.0).to(feats.device)
    return F.mse_loss(feats, ref_feats.detach()) * 1e3


def total_loss(suploss_lb, suploss_kd, conloss_lb, conloss_ulb, consloss_ulb,
               consloss_ulb_aug, suploss_kd_ulb, lambda_ce=1.0, lambda_kd=1.0,
               lambda_con=1.0, lambda_cons=1.0, lambda_ukd=1.0):
    """USP 总损失（原实现 375 行）。"""
    return (lambda_ce * suploss_lb
            + lambda_kd * suploss_kd
            + lambda_con * (conloss_lb + conloss_ulb)
            + lambda_cons * (consloss_ulb + consloss_ulb_aug)
            + lambda_ukd * suploss_kd_ulb)


def validate(tg_model, testloader, device, weight_per_class=None, old_cn=0):
    """测试集评估（原 validate 函数，session/old/new 指标拆分由调用方处理）。"""
    tg_model.eval()
    test_loss = 0
    correct = 0
    total = 0

    predicted_list = []
    gt_list = []

    with torch.no_grad():
        batch_idx = -1
        for batch_idx, batch in enumerate(testloader):
            inputs, _, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _, _, session_outputs = tg_model(inputs, return_feats=True)
            loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            predicted_list.append(predicted.cpu().numpy())
            gt_list.append(targets.cpu().numpy())

    if batch_idx < 0:
        return 0.0, 0.0, 0.0, 0.0

    predicted_list = np.concatenate(predicted_list)
    gt_list = np.concatenate(gt_list)

    old_mask = gt_list < old_cn
    new_mask = gt_list >= old_cn
    old_acc = (predicted_list[old_mask] == gt_list[old_mask]).mean() if old_mask.sum() > 0 else 0.0
    new_acc = (predicted_list[new_mask] == gt_list[new_mask]).mean() if new_mask.sum() > 0 else 0.0

    return (test_loss / (batch_idx + 1), 100.0 * correct / total,
            100.0 * old_acc, 100.0 * new_acc)
