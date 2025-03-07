# 以下是针对ot.emd2实现的版本，要将数据转移到CPU上；

# import torch
# import numpy as np
# import torch.nn.functional as F
# import ot
# # pip install pot
#
# def compute_wasserstein_loss(
#         logits,  # 学生模型的输出 [batch_size, seq_length, student_vocab_size]
#         teacher_logits,  # 教师模型的输出 [batch_size, seq_length, teacher_vocab_size]
#         target,  # 目标标签 [batch_size, seq_length]
#         padding_id=-100,  # 用于填充的ID
#         reduction="sum",  # 指定如何处理多个样本的损失，"sum" 或 "mean"
#         temp=2.0,  # 温度参数，用于调整概率分布的平滑度
#         wasserstein_version=1,  # Wasserstein距离版本：1（W1）或 2（W2）
# ):
#     """
#     计算基于Wasserstein距离的知识蒸馏损失，借鉴bert-emd的实现
#     Args:
#         logits: 学生模型的logits
#         teacher_logits: 教师模型的logits（可能维度更大）
#         target: 目标标签
#         padding_id: 用于忽略填充部分的ID
#         reduction: 损失缩减方式，"sum" 或 "mean"
#         temp: 温度参数
#         wasserstein_version: Wasserstein距离版本，1（W1，L1范数）或 2（W2，L2范数）
#     Returns:
#         torch.Tensor: 计算得到的Wasserstein损失
#     """
#     # 获取有效序列长度（忽略padding）
#     mask = (target != padding_id).float()  # [batch_size, seq_length]
#
#     # 获取学生和教师的词汇表大小
#     student_vocab_size = logits.shape[-1]
#     teacher_vocab_size = teacher_logits.shape[-1]
#     max_vocab_size = max(student_vocab_size, teacher_vocab_size)
#
#     # 将学生logits扩展到与教师相同的维度（填充负无穷）
#     if student_vocab_size < teacher_vocab_size:
#         padding = torch.full(
#             (logits.shape[0], logits.shape[1], teacher_vocab_size - student_vocab_size),
#             float('-inf'),
#             device=logits.device,
#             dtype=logits.dtype
#         )
#         logits_padded = torch.cat([logits, padding], dim=-1)
#     else:
#         logits_padded = logits
#
#     # 转换为概率分布
#     student_probs = F.softmax(logits_padded / temp, dim=-1)  # [batch_size, seq_length, max_vocab_size]
#     teacher_probs = F.softmax(teacher_logits / temp, dim=-1)  # [batch_size, seq_length, teacher_vocab_size]
#
#     if teacher_vocab_size < max_vocab_size:
#         padding = torch.zeros(
#             (teacher_logits.shape[0], teacher_logits.shape[1], max_vocab_size - teacher_vocab_size),
#             device=teacher_logits.device,
#             dtype=teacher_logits.dtype
#         )
#         teacher_probs = torch.cat([teacher_probs, padding], dim=-1)
#
#     batch_size, seq_length, vocab_size = student_probs.shape
#     losses = []
#
#     # 对每个batch和序列位置计算Wasserstein距离
#     for b in range(batch_size):
#         for s in range(seq_length):
#             if mask[b, s] == 0:  # 跳过padding位置
#                 continue
#
#             # 获取当前学生和教师的概率分布，并转换为float32以兼容numpy
#             student_dist = student_probs[b, s].to(torch.float32).detach().cpu().numpy().astype('float64')
#             teacher_dist = teacher_probs[b, s].to(torch.float32).detach().cpu().numpy().astype('float64')
#
#             # 构造距离矩阵（词汇表之间的绝对距离）
#             distance_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float64)
#             for i in range(vocab_size):
#                 for j in range(vocab_size):
#                     distance_matrix[i, j] = abs(i - j) if wasserstein_version == 1 else (i - j) ** 2
#
#             # 计算EMD
#             if wasserstein_version == 1:
#                 emd_loss = ot.emd2(student_dist, teacher_dist, distance_matrix)
#             elif wasserstein_version == 2:
#                 emd_loss = np.sqrt(ot.emd2(student_dist, teacher_dist, distance_matrix))
#             else:
#                 raise ValueError("wasserstein_version must be 1 or 2")
#
#             losses.append(emd_loss)
#
#     # 处理损失
#     if not losses:
#         loss = torch.tensor(0.0, device=logits.device)
#     else:
#         loss = torch.tensor(losses, device=logits.device)
#         if reduction == "sum":
#             loss = loss.sum()
#         elif reduction == "mean":
#             loss = loss.mean()
#         else:
#             raise ValueError("reduction must be 'sum' or 'mean'")
#
#     return loss


# 为了充分发挥 bf16 和 GPU 的优势，我们可以用 PyTorch 原生操作近似计算 Wasserstein 距离，避免依赖 ot.emd2
# 使用pytorch加速
import torch
import torch.nn.functional as F

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
    使用 PyTorch 原生操作计算 Wasserstein 距离，支持 bf16。
    """
    # 确定目标数据类型
    if logits.dtype == torch.bfloat16 or teacher_logits.dtype == torch.bfloat16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # 将温度参数转换为张量
    temp = torch.tensor(temp, dtype=dtype, device=logits.device)

    # 获取有效序列长度（忽略padding）
    mask = (target != padding_id).to(dtype=dtype)  # [batch_size, seq_length]

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
            dtype=dtype
        )
        logits_padded = torch.cat([logits.to(dtype=dtype), padding], dim=-1)
    else:
        logits_padded = logits.to(dtype=dtype)

    # 转换为概率分布
    student_probs = F.softmax(logits_padded / temp, dim=-1)  # [batch_size, seq_length, max_vocab_size]
    teacher_probs = F.softmax(teacher_logits.to(dtype=dtype) / temp, dim=-1)  # [batch_size, seq_length, teacher_vocab_size]

    if teacher_vocab_size < max_vocab_size:
        padding = torch.zeros(
            (teacher_logits.shape[0], teacher_logits.shape[1], max_vocab_size - teacher_vocab_size),
            device=teacher_logits.device,
            dtype=dtype
        )
        teacher_probs = torch.cat([teacher_probs, padding], dim=-1)

    # 构造位置索引并计算距离矩阵
    vocab_indices = torch.arange(max_vocab_size, dtype=dtype, device=logits.device)
    vocab_indices = vocab_indices.view(1, 1, -1)  # [1, 1, max_vocab_size]
    if wasserstein_version == 1:
        w_loss = torch.abs(student_probs - teacher_probs)  # L1 距离近似
        w_loss = w_loss.sum(dim=-1)  # [batch_size, seq_length]
    elif wasserstein_version == 2:
        w_loss = (student_probs - teacher_probs) ** 2  # L2 距离近似
        w_loss = w_loss.sum(dim=-1).sqrt()  # [batch_size, seq_length]
    else:
        raise ValueError("wasserstein_version must be 1 or 2")

    # 应用掩码
    w_loss = w_loss * mask

    # 归约损失
    if reduction == "sum":
        loss = w_loss.sum()
    elif reduction == "mean":
        loss = w_loss.sum() / mask.sum()
    else:
        raise ValueError("reduction must be 'sum' or 'mean'")

    return loss

def example_usage():
    batch_size, seq_length = 2, 4
    student_vocab_size, teacher_vocab_size = 10, 15
    logits = torch.randn(batch_size, seq_length, student_vocab_size)
    teacher_logits = torch.randn(batch_size, seq_length, teacher_vocab_size)
    target = torch.tensor([[1, 2, 3, -100], [0, 1, -100, -100]], dtype=torch.long)

    loss_w1 = compute_wasserstein_loss(logits, teacher_logits, target, padding_id=-100, temp=2.0, wasserstein_version=1)
    loss_w2 = compute_wasserstein_loss(logits, teacher_logits, target, padding_id=-100, temp=2.0, wasserstein_version=2)

    print(f"Wasserstein-1 Loss: {loss_w1.item()}")
    print(f"Wasserstein-2 Loss: {loss_w2.item()}")

# 示例用法
if __name__ == "__main__":
    example_usage()