# import torch
# import torch.nn.functional as F

# def compute_class_center_distance_matrix(
#     teacher_logits_samples,  # 教师模型的logits样本 [num_samples, seq_length, vocab_size]
#     reduction="mean"  # 如何归约seq_length维度，"mean" 或 "sum"
# ):
#     """
#     计算基于类中心距离的欧几里得距离矩阵。
#     输入：教师模型在训练数据上的logits样本。
#     输出：距离矩阵 [vocab_size, vocab_size]。
#     """
#     # 检查输入维度
#     if teacher_logits_samples.dim() != 3:
#         raise ValueError("teacher_logits_samples must be of shape [num_samples, seq_length, vocab_size]")

#     num_samples, seq_length, vocab_size = teacher_logits_samples.shape

#     # 对seq_length维度归约，计算每个类别的logits均值
#     if reduction == "mean":
#         class_means = teacher_logits_samples.mean(dim=0)  # [seq_length, vocab_size]
#         class_means = class_means.mean(dim=0)  # [vocab_size]
#     elif reduction == "sum":
#         class_means = teacher_logits_samples.sum(dim=0).sum(dim=0)  # [vocab_size]
#     else:
#         raise ValueError("reduction must be 'mean' or 'sum'")

#     # 计算欧几里得距离矩阵
#     class_means = class_means.unsqueeze(0)  # [1, vocab_size]
#     distance_matrix = torch.cdist(class_means, class_means, p=2).squeeze(0)  # [vocab_size, vocab_size]

#     return distance_matrix

# def compute_wasserstein_loss(
#     logits,  # 学生模型的输出 [batch_size, seq_length, student_vocab_size]
#     teacher_logits,  # 教师模型的输出 [batch_size, seq_length, teacher_vocab_size]
#     target,  # 目标标签 [batch_size, seq_length]
#     padding_id=-100,  # 用于填充的ID
#     reduction="sum",  # "sum" 或 "mean"
#     temp=2.0,  # 温度参数，建议实验范围 [1.0, 5.0]
#     wasserstein_version=1,  # Wasserstein距离版本：1（W1）或 2（W2）
#     distance_matrix=None,  # 可选的类间距离矩阵 [vocab_size, vocab_size]，默认用类中心距离
# ):
#     """
#     使用 PyTorch 原生操作计算 Wasserstein 距离，支持 bf16。
#     改进：基于CDF计算，支持类中心距离矩阵。
#     """
#     # 确定目标数据类型
#     if logits.dtype == torch.bfloat16 or teacher_logits.dtype == torch.bfloat16:
#         dtype = torch.bfloat16
#     else:
#         dtype = torch.float32

#     # 将温度参数转换为张量
#     temp = torch.tensor(temp, dtype=dtype, device=logits.device)

#     # 获取有效序列长度（忽略padding）
#     mask = (target != padding_id).to(dtype=dtype)  # [batch_size, seq_length]

#     # 获取学生和教师的词汇表大小
#     student_vocab_size = logits.shape[-1]
#     teacher_vocab_size = teacher_logits.shape[-1]
#     max_vocab_size = max(student_vocab_size, teacher_vocab_size)

#     # 将学生logits扩展到与教师相同的维度（填充负无穷）
#     if student_vocab_size < teacher_vocab_size:
#         padding = torch.full(
#             (logits.shape[0], logits.shape[1], teacher_vocab_size - student_vocab_size),
#             float('-inf'),
#             device=logits.device,
#             dtype=dtype
#         )
#         logits_padded = torch.cat([logits.to(dtype=dtype), padding], dim=-1)
#     else:
#         logits_padded = logits.to(dtype=dtype)

#     # 转换为概率分布
#     student_probs = F.softmax(logits_padded / temp, dim=-1)  # [batch_size, seq_length, max_vocab_size]
#     teacher_probs = F.softmax(teacher_logits.to(dtype=dtype) / temp, dim=-1)  # [batch_size, seq_length, teacher_vocab_size]

#     if teacher_vocab_size < max_vocab_size:
#         padding = torch.zeros(
#             (teacher_logits.shape[0], teacher_logits.shape[1], max_vocab_size - teacher_vocab_size),
#             device=teacher_logits.device,
#             dtype=dtype
#         )
#         teacher_probs = torch.cat([teacher_probs, padding], dim=-1)

#     # 构造默认距离矩阵（若未提供）
#     if distance_matrix is None:
#         # 这里假设没有样本数据可用，默认回退到 |i - j|
#         vocab_indices = torch.arange(max_vocab_size, dtype=dtype, device=logits.device)
#         distance_matrix = torch.abs(vocab_indices.unsqueeze(1) - vocab_indices.unsqueeze(0))
#         print("Warning: No distance_matrix provided, falling back to |i - j|. Provide teacher_logits_samples for class center distance.")
#     else:
#         if distance_matrix.shape != (max_vocab_size, max_vocab_size):
#             raise ValueError(f"distance_matrix must be of shape [{max_vocab_size}, {max_vocab_size}]")
#         distance_matrix = distance_matrix.to(dtype=dtype, device=logits.device)

#     # 计算累积分布函数（CDF）
#     student_cdf = torch.cumsum(student_probs, dim=-1)  # [batch_size, seq_length, max_vocab_size]
#     teacher_cdf = torch.cumsum(teacher_probs, dim=-1)  # [batch_size, seq_length, max_vocab_size]

#     # 计算Wasserstein距离
#     if wasserstein_version == 1:
#         w_loss = torch.abs(student_cdf - teacher_cdf)  # L1距离
#         w_loss = (w_loss * distance_matrix.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # [batch_size, seq_length]
#     elif wasserstein_version == 2:
#         w_loss = (student_cdf - teacher_cdf) ** 2  # L2距离
#         w_loss = (w_loss * distance_matrix.unsqueeze(0).unsqueeze(0)).sum(dim=-1).sqrt()  # [batch_size, seq_length]
#     else:
#         raise ValueError("wasserstein_version must be 1 or 2")

#     # 应用掩码
#     w_loss = w_loss * mask

#     # 归约损失
#     if reduction == "sum":
#         loss = w_loss.sum()
#     elif reduction == "mean":
#         loss = w_loss.sum() / mask.sum()
#     else:
#         raise ValueError("reduction must be 'sum' or 'mean'")

#     return loss

# # 示例用法
# if __name__ == "__main__":
#     # 随机生成测试数据
#     logits = torch.randn(2, 3, 5)  # 学生模型输出
#     teacher_logits = torch.randn(2, 3, 7)  # 教师模型输出
#     target = torch.tensor([[1, 2, -100], [0, 1, 2]])  # 目标标签，-100为padding

#     # 生成教师logits样本并计算类中心距离矩阵
#     teacher_samples = torch.randn(100, 3, 7)  # 假设的训练数据样本
#     distance_matrix = compute_class_center_distance_matrix(teacher_samples, reduction="mean")

#     # 计算Wasserstein损失
#     loss_w1 = compute_wasserstein_loss(
#         logits, teacher_logits, target, wasserstein_version=1, distance_matrix=distance_matrix
#     )
#     print(f"W1 Loss with class center distance: {loss_w1.item()}")

#     loss_w2 = compute_wasserstein_loss(
#         logits, teacher_logits, target, wasserstein_version=2, distance_matrix=distance_matrix
#     )
#     print(f"W2 Loss with class center distance: {loss_w2.item()}")




#版本 2（基于 CDF，带 vocab_indices 加权）
import torch
import torch.nn.functional as F

def compute_wasserstein_loss(
    logits,
    teacher_logits,
    target,
    padding_id=-100,
    reduction="sum",
    temp=2.0,
    wasserstein_version=1,
):
    dtype = torch.bfloat16 if logits.dtype == torch.bfloat16 else torch.float32
    temp = torch.tensor(temp, dtype=dtype, device=logits.device)
    mask = (target != padding_id).to(dtype=dtype)

    student_vocab_size = logits.shape[-1]
    teacher_vocab_size = teacher_logits.shape[-1]
    max_vocab_size = max(student_vocab_size, teacher_vocab_size)

    if student_vocab_size < teacher_vocab_size:
        padding = torch.full(
            (logits.shape[0], logits.shape[1], teacher_vocab_size - student_vocab_size),
            float('-inf'), device=logits.device, dtype=dtype
        )
        logits_padded = torch.cat([logits.to(dtype=dtype), padding], dim=-1)
    else:
        logits_padded = logits.to(dtype=dtype)

    student_probs = F.softmax(logits_padded / temp, dim=-1)
    teacher_probs = F.softmax(teacher_logits.to(dtype=dtype) / temp, dim=-1)
    if teacher_vocab_size < max_vocab_size:
        padding = torch.zeros(
            (teacher_logits.shape[0], teacher_logits.shape[1], max_vocab_size - teacher_vocab_size),
            device=teacher_logits.device, dtype=dtype
        )
        teacher_probs = torch.cat([teacher_probs, padding], dim=-1)

    vocab_indices = torch.arange(max_vocab_size, dtype=dtype, device=logits.device)
    student_cdf = torch.cumsum(student_probs, dim=-1)
    teacher_cdf = torch.cumsum(teacher_probs, dim=-1)
    
    if wasserstein_version == 1:
        w_loss = torch.abs(student_cdf - teacher_cdf) * vocab_indices
        w_loss = w_loss.sum(dim=-1)
    elif wasserstein_version == 2:
        w_loss = (student_cdf - teacher_cdf) ** 2 * vocab_indices
        w_loss = w_loss.sum(dim=-1).sqrt()

    w_loss = w_loss * mask
    return w_loss.sum() if reduction == "sum" else w_loss.sum() / mask.sum()