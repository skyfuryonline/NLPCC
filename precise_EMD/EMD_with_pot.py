import torch
import numpy as np
import torch.nn.functional as F
import ot

def compute_wasserstein_loss(
        logits,  # 学生模型的输出 [batch_size, seq_length, student_vocab_size]
        teacher_logits,  # 教师模型的输出 [batch_size, seq_length, teacher_vocab_size]
        target,  # 目标标签 [batch_size, seq_length]
        padding_id=-100,  # 用于填充的ID
        reduction="sum",  # 指定如何处理多个样本的损失，"sum" 或 "mean"
        temp=2.0,  # 温度参数，用于调整概率分布的平滑度
        wasserstein_version=1,  # Wasserstein距离版本：1（W1）或 2（W2）
):
    """
    计算基于Wasserstein距离的知识蒸馏损失，使用emd_with_flow实现
    Args:
        logits: 学生模型的logits
        teacher_logits: 教师模型的logits（可能维度更大）
        target: 目标标签
        padding_id: 用于忽略填充部分的ID
        reduction: 损失缩减方式，"sum" 或 "mean"
        temp: 温度参数
        wasserstein_version: Wasserstein距离版本，1（W1，L1范数）或 2（W2，L2范数）
    Returns:
        torch.Tensor: 计算得到的Wasserstein损失
    """
    # 获取有效序列长度（忽略padding）
    mask = (target != padding_id).float()  # [batch_size, seq_length]

    # 获取学生和教师的词汇表大小
    student_vocab_size = logits.shape[-1]
    teacher_vocab_size = teacher_logits.shape[-1]
    max_vocab_size = max(student_vocab_size, teacher_vocab_size)

    # 将学生logits扩展到与教师相同的维度（填充负无穷）
    if student_vocab_size < teacher_vocab_size:
        padding = torch.full(
            (logits.shape[0], logits.shape[1], teacher_vocab_size - student_vocab_size),
            float('-inf'),
            device=logits.device,
            dtype=logits.dtype
        )
        logits_padded = torch.cat([logits, padding], dim=-1)
    else:
        logits_padded = logits

    # 转换为概率分布
    student_probs = F.softmax(logits_padded / temp, dim=-1)  # [batch_size, seq_length, max_vocab_size]
    teacher_probs = F.softmax(teacher_logits / temp, dim=-1)  # [batch_size, seq_length, teacher_vocab_size]

    if teacher_vocab_size < max_vocab_size:
        padding = torch.zeros(
            (teacher_logits.shape[0], teacher_logits.shape[1], max_vocab_size - teacher_vocab_size),
            device=teacher_logits.device,
            dtype=teacher_logits.dtype
        )
        teacher_probs = torch.cat([teacher_probs, padding], dim=-1)

    batch_size, seq_length, vocab_size = student_probs.shape
    losses = []

    # 对每个batch和序列位置计算Wasserstein距离
    for b in range(batch_size):
        for s in range(seq_length):
            if mask[b, s] == 0:  # 跳过padding位置
                continue

            # 获取当前学生和教师的概率分布，并转换为float64以兼容numpy
            student_dist = student_probs[b, s].to(torch.float32).detach().cpu().numpy().astype('float64')
            teacher_dist = teacher_probs[b, s].to(torch.float32).detach().cpu().numpy().astype('float64')

            # 构造距离矩阵（词汇表之间的绝对距离）
            distance_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float64)
            for i in range(vocab_size):
                for j in range(vocab_size):
                    distance_matrix[i, j] = abs(i - j) if wasserstein_version == 1 else (i - j) ** 2

            # 使用emd_with_flow计算EMD和运输计划
            transport_plan = ot.emd(student_dist, teacher_dist, distance_matrix)
            # 计算总成本
            emd_loss = np.sum(transport_plan * distance_matrix)

            if wasserstein_version == 2:
                emd_loss = np.sqrt(emd_loss)  # W2需要开平方根

            losses.append(emd_loss)

    # 处理损失
    if not losses:
        loss = torch.tensor(0.0, device=logits.device)
    else:
        loss = torch.tensor(losses, device=logits.device)
        if reduction == "sum":
            loss = loss.sum()
        elif reduction == "mean":
            loss = loss.mean()
        else:
            raise ValueError("reduction must be 'sum' or 'mean'")

    return loss