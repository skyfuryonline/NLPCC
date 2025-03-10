# emd_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.corpus import treebank


class EMDLossWithProjection(nn.Module):
    def __init__(self, student_vocab_size, teacher_vocab_size):
        super(EMDLossWithProjection, self).__init__()
        self.student_vocab_size = student_vocab_size
        self.teacher_vocab_size = teacher_vocab_size
        self.max_vocab_size = max(student_vocab_size, teacher_vocab_size)

        # 可训练的投影层：将学生 logits 投影到与教师相同的表示空间
        # 这里假设需要将学生 logits 的特征维度投影到教师的维度，但由于输入是 vocab_size，我们直接投影整个向量
        self.projection = nn.Linear(student_vocab_size, self.max_vocab_size, bias=False)

    def align_vocab(self, student_logits, teacher_logits, padding_id=0):
        """
        对齐学生和教师模型的词表大小，如果不一致则填充
        输入：
            student_logits: (seq_length, student_vocab_size)
            teacher_logits: (seq_length, teacher_vocab_size)
            padding_id: 填充标记的 ID
        输出：
            student_logits_aligned, teacher_logits_aligned: 对齐后的 logits
        """
        seq_length = student_logits.shape[0]

        # 如果词表大小不同，填充到最大词表大小
        if self.student_vocab_size < self.max_vocab_size:
            padding = torch.full((seq_length, self.max_vocab_size - self.student_vocab_size), float('-inf'),
                                 device=student_logits.device)
            student_logits_padded = torch.cat([student_logits, padding], dim=1)
        else:
            student_logits_padded = student_logits

        if self.teacher_vocab_size < self.max_vocab_size:
            padding = torch.full((seq_length, self.max_vocab_size - self.teacher_vocab_size), float('-inf'),
                                 device=teacher_logits.device)
            teacher_logits_padded = torch.cat([teacher_logits, padding], dim=1)
        else:
            teacher_logits_padded = teacher_logits

        # 投影学生 logits
        student_logits_aligned = self.projection(student_logits_padded)
        return student_logits_aligned, teacher_logits_padded

    def compute_ground_distance(self, student_logits, teacher_logits):
        """
        计算地面距离矩阵 D，基于 MSE
        输入：
            student_logits: (block_size, vocab_size)
            teacher_logits: (block_size, vocab_size)
        输出：
            D: (block_size, block_size)
        """
        block_size, vocab_size = student_logits.shape

        D = torch.zeros(block_size, block_size, device=student_logits.device)
        for i in range(block_size):
            for j in range(block_size):
                D[i, j] = F.mse_loss(student_logits[i], teacher_logits[j], reduction='mean')

        return D

    def compute_flow(self, D, w_T, w_S):
        """
        计算流量矩阵 F，简化版：假设均匀分配流量
        输入：
            D: (block_size, block_size)
            w_T, w_S: (block_size,) - 教师和学生的初始权重
        输出：
            F: (block_size, block_size)
        """
        block_size = D.shape[0]
        F = torch.ones(block_size, block_size, device=D.device) * (min(w_T.sum(), w_S.sum()) / (block_size ** 2))
        F = F * (w_T.unsqueeze(1) / F.sum(dim=1, keepdim=True).clamp(min=1e-6))  # 满足教师约束
        F = F * (w_S.unsqueeze(0) / F.sum(dim=0, keepdim=True).clamp(min=1e-6))  # 满足学生约束
        F = F.clamp(min=0)  # 确保非负
        return F

    def update_weights(self, D, F, w_T, w_S, temperature=1.0):
        """
        动态调整权重基于代价注意力机制
        输入：
            D: (block_size, block_size)
            F: (block_size, block_size)
            w_T, w_S: (block_size,)
            temperature: 温度参数
        输出：
            w_T_new, w_S_new: 更新后的权重
        """
        C_T = (D * F).sum(dim=1) / w_T.clamp(min=1e-6)  # (block_size,)
        w_T_bar = C_T.sum() / C_T.clamp(min=1e-6)  # scalar
        w_S_bar = C_T.sum() / C_T.clamp(min=1e-6)  # scalar (对称假设简化)
        w_T_new = F.softmax(w_T_bar / temperature * torch.ones_like(w_T), dim=0)
        w_S_new = F.softmax(w_S_bar / temperature * torch.ones_like(w_S), dim=0)
        return w_T_new, w_S_new

    def forward(self, student_logits, teacher_logits, temperature=1.0, reduction='mean', padding_id=0, block_size=64):
        """
        计算 EMD 损失，处理词表对齐和分块
        输入：
            student_logits: (seq_length, student_vocab_size)
            teacher_logits: (seq_length, teacher_vocab_size)
            temperature: 温度参数
            reduction: 'mean', 'sum', 或 'none'
            padding_id: 填充标记 ID
            block_size: 分块大小
        输出：
            loss: 根据 reduction 返回标量或张量
        """
        # 对齐词表并投影
        student_logits, teacher_logits = self.align_vocab(student_logits, teacher_logits, padding_id)
        seq_length, vocab_size = student_logits.shape

        # 创建掩码以屏蔽 padding
        mask = (student_logits.max(dim=1)[1] != padding_id).float()  # (seq_length,)

        # 分块计算
        num_blocks = (seq_length + block_size - 1) // block_size
        emd_total = 0.0
        valid_blocks = 0

        for i in range(num_blocks):
            start = i * block_size
            end = min((i + 1) * block_size, seq_length)
            block_size_actual = end - start

            if block_size_actual <= 0:
                continue

            student_block = student_logits[start:end]  # (block_size_actual, vocab_size)
            teacher_block = teacher_logits[start:end]  # (block_size_actual, vocab_size)
            block_mask = mask[start:end]

            if block_mask.sum() == 0:  # 全为 padding 跳过
                continue

            # 初始权重
            w_T = torch.ones(block_size_actual, device=student_logits.device) / block_size_actual
            w_S = torch.ones(block_size_actual, device=student_logits.device) / block_size_actual
            w_T = w_T * block_mask / block_mask.sum().clamp(min=1e-6)
            w_S = w_S * block_mask / block_mask.sum().clamp(min=1e-6)

            # 计算地面距离矩阵
            D = self.compute_ground_distance(student_block, teacher_block)

            # 计算流量矩阵
            F = self.compute_flow(D, w_T, w_S)

            # 更新权重并重新计算 F
            w_T, w_S = self.update_weights(D, F, w_T, w_S, temperature)
            F = self.compute_flow(D, w_T, w_S)

            # 计算 EMD
            WORK = (F * D).sum()
            total_flow = F.sum()
            emd = WORK / total_flow.clamp(min=1e-6)

            emd_total += emd * block_mask.mean()  # 加权考虑有效 token 比例
            valid_blocks += 1

        if valid_blocks == 0:
            emd_loss_value = torch.tensor(0.0, device=student_logits.device)
        else:
            emd_loss_value = emd_total / valid_blocks if reduction == 'mean' else emd_total

        if reduction == 'none':
            return emd_total
        return emd_loss_value


# 测试函数（可选）
def test_emd_loss():
    emd_loss_fn = EMDLossWithProjection(student_vocab_size=151665, teacher_vocab_size=152000)
    student_logits = torch.randn(128, 151665)
    teacher_logits = torch.randn(128, 152000)
    loss = emd_loss_fn(student_logits, teacher_logits)
    print(f"EMD Loss: {loss.item()}")


if __name__=="__main__":
    test_emd_loss()