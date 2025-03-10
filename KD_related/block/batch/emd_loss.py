# emd_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class EMDLossWithProjection(nn.Module):
    def __init__(self, student_vocab_size, teacher_vocab_size):
        super(EMDLossWithProjection, self).__init__()
        self.student_vocab_size = student_vocab_size
        self.teacher_vocab_size = teacher_vocab_size
        self.max_vocab_size = max(student_vocab_size, teacher_vocab_size)

        # 可训练的投影层，作用于 vocab_size 维度
        self.projection = nn.Linear(student_vocab_size, self.max_vocab_size, bias=False)

    def align_vocab(self, student_logits, teacher_logits, padding_id=0):
        """
        对齐学生和教师模型的词表大小，如果不一致则填充并投影
        输入：
            student_logits: (batch_size, seq_length, student_vocab_size)
            teacher_logits: (batch_size, seq_length, teacher_vocab_size)
            padding_id: 填充标记的 ID
        输出：
            student_logits_aligned, teacher_logits_aligned: (batch_size, seq_length, max_vocab_size)
        """
        batch_size, seq_length, _ = student_logits.shape

        # 填充学生 logits
        if self.student_vocab_size < self.max_vocab_size:
            padding = torch.full((batch_size, seq_length, self.max_vocab_size - self.student_vocab_size), float('-inf'),
                                 device=student_logits.device)
            student_logits_padded = torch.cat([student_logits, padding], dim=2)
        else:
            student_logits_padded = student_logits

        # 填充教师 logits
        if self.teacher_vocab_size < self.max_vocab_size:
            padding = torch.full((batch_size, seq_length, self.max_vocab_size - self.teacher_vocab_size), float('-inf'),
                                 device=teacher_logits.device)
            teacher_logits_padded = torch.cat([teacher_logits, padding], dim=2)
        else:
            teacher_logits_padded = teacher_logits

        # 投影学生 logits
        student_logits_aligned = self.projection(student_logits_padded)  # (batch_size, seq_length, max_vocab_size)
        return student_logits_aligned, teacher_logits_padded

    def compute_ground_distance(self, student_logits, teacher_logits):
        """
        计算地面距离矩阵 D，基于 MSE，支持批次维度
        输入：
            student_logits: (batch_size, block_size, vocab_size)
            teacher_logits: (batch_size, block_size, vocab_size)
        输出：
            D: (batch_size, block_size, block_size)
        """
        batch_size, block_size, vocab_size = student_logits.shape
        D = torch.zeros(batch_size, block_size, block_size, device=student_logits.device)

        # 批量计算 MSE
        for i in range(block_size):
            for j in range(block_size):
                D[:, i, j] = F.mse_loss(student_logits[:, i, :], teacher_logits[:, j, :], reduction='mean', dim=1)
        return D

    def compute_flow(self, D, w_T, w_S):
        """
        计算流量矩阵 F，支持批次维度
        输入：
            D: (batch_size, block_size, block_size)
            w_T, w_S: (batch_size, block_size) - 教师和学生的初始权重
        输出：
            F: (batch_size, block_size, block_size)
        """
        batch_size, block_size, _ = D.shape
        total_flow = torch.min(w_T.sum(dim=1), w_S.sum(dim=1))  # (batch_size,)
        F = torch.ones(batch_size, block_size, block_size, device=D.device) * (total_flow / (block_size ** 2)).view(-1,
                                                                                                                    1,
                                                                                                                    1)
        F = F * (w_T.unsqueeze(2) / F.sum(dim=2, keepdim=True).clamp(min=1e-6))  # 满足教师约束
        F = F * (w_S.unsqueeze(1) / F.sum(dim=1, keepdim=True).clamp(min=1e-6))  # 满足学生约束
        F = F.clamp(min=0)
        return F

    def update_weights(self, D, F, w_T, w_S, temperature=1.0):
        """
        动态调整权重，支持批次维度
        输入：
            D, F: (batch_size, block_size, block_size)
            w_T, w_S: (batch_size, block_size)
            temperature: 温度参数
        输出：
            w_T_new, w_S_new: (batch_size, block_size)
        """
        C_T = (D * F).sum(dim=2) / w_T.clamp(min=1e-6)  # (batch_size, block_size)
        w_T_bar = C_T.sum(dim=1, keepdim=True) / C_T.clamp(min=1e-6)  # (batch_size, 1)
        w_S_bar = C_T.sum(dim=1, keepdim=True) / C_T.clamp(min=1e-6)  # (batch_size, 1)
        w_T_new = F.softmax(w_T_bar / temperature * torch.ones_like(w_T), dim=1)
        w_S_new = F.softmax(w_S_bar / temperature * torch.ones_like(w_S), dim=1)
        return w_T_new, w_S_new

    def forward(self, student_logits, teacher_logits, temperature=1.0, reduction='mean', padding_id=0, block_size=64):
        """
        计算 EMD 损失，支持三维输入
        输入：
            student_logits: (batch_size, seq_length, student_vocab_size)
            teacher_logits: (batch_size, seq_length, teacher_vocab_size)
            temperature, reduction, padding_id, block_size
        输出：
            loss: 根据 reduction 返回标量或张量
        """
        batch_size, seq_length, _ = student_logits.shape

        # 对齐词表并投影
        student_logits, teacher_logits = self.align_vocab(student_logits, teacher_logits, padding_id)
        vocab_size = self.max_vocab_size

        # 创建掩码
        mask = (student_logits.max(dim=2)[1] != padding_id).float()  # (batch_size, seq_length)

        # 分块计算
        num_blocks = (seq_length + block_size - 1) // block_size
        emd_total = torch.zeros(batch_size, device=student_logits.device)

        for i in range(num_blocks):
            start = i * block_size
            end = min((i + 1) * block_size, seq_length)
            block_size_actual = end - start

            if block_size_actual <= 0:
                continue

            student_block = student_logits[:, start:end, :]  # (batch_size, block_size_actual, vocab_size)
            teacher_block = teacher_logits[:, start:end, :]  # (batch_size, block_size_actual, vocab_size)
            block_mask = mask[:, start:end]  # (batch_size, block_size_actual)

            if block_mask.sum() == 0:  # 全为 padding 跳过
                continue

            # 初始权重
            w_T = torch.ones(batch_size, block_size_actual, device=student_logits.device) / block_size_actual
            w_S = torch.ones(batch_size, block_size_actual, device=student_logits.device) / block_size_actual
            w_T = w_T * block_mask / block_mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
            w_S = w_S * block_mask / block_mask.sum(dim=1, keepdim=True).clamp(min=1e-6)

            # 计算地面距离矩阵
            D = self.compute_ground_distance(student_block, teacher_block)

            # 计算流量矩阵
            F = self.compute_flow(D, w_T, w_S)

            # 更新权重并重新计算 F
            w_T, w_S = self.update_weights(D, F, w_T, w_S, temperature)
            F = self.compute_flow(D, w_T, w_S)

            # 计算 EMD
            WORK = (F * D).sum(dim=[1, 2])  # (batch_size,)
            total_flow = F.sum(dim=[1, 2])  # (batch_size,)
            emd = WORK / total_flow.clamp(min=1e-6)
            emd_total += emd * block_mask.mean(dim=1)  # 加权考虑有效 token 比例

        # 归约
        if reduction == 'mean':
            return emd_total.mean()
        elif reduction == 'sum':
            return emd_total.sum()
        elif reduction == 'none':
            return emd_total
        else:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")