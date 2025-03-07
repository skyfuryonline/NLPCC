import torch
import torch.nn.functional as F
import numpy as np

from pyemd import emd_with_flow

def combined_wasserstein_loss(
    student_logits, teacher_logits, target, student_reps, teacher_reps,
    padding_id=-100, reduction="sum", temp=2.0, wasserstein_version=1,
    student_layer_weight, teacher_layer_weight, device, loss_mse, alpha=0.5
):
    # 输出分布损失（代码B）
    dtype = torch.bfloat16 if student_logits.dtype == torch.bfloat16 else torch.float32
    temp = torch.tensor(temp, dtype=dtype, device=student_logits.device)
    mask = (target != padding_id).to(dtype)

    student_vocab_size = student_logits.shape[-1]
    teacher_vocab_size = teacher_logits.shape[-1]
    max_vocab_size = max(student_vocab_size, teacher_vocab_size)

    if student_vocab_size < teacher_vocab_size:
        padding = torch.full((student_logits.shape[0], student_logits.shape[1], teacher_vocab_size - student_vocab_size),
                             float('-inf'), device=student_logits.device, dtype=dtype)
        logits_padded = torch.cat([student_logits.to(dtype), padding], dim=-1)
    else:
        logits_padded = student_logits.to(dtype)

    student_probs = F.softmax(logits_padded / temp, dim=-1)
    teacher_probs = F.softmax(teacher_logits.to(dtype) / temp, dim=-1)
    if teacher_vocab_size < max_vocab_size:
        padding = torch.zeros((teacher_logits.shape[0], teacher_logits.shape[1], max_vocab_size - teacher_vocab_size),
                              device=teacher_logits.device, dtype=dtype)
        teacher_probs = torch.cat([teacher_probs, padding], dim=-1)

    vocab_indices = torch.arange(max_vocab_size, dtype=dtype, device=student_logits.device)
    student_cdf = torch.cumsum(student_probs, dim=-1)
    teacher_cdf = torch.cumsum(teacher_probs, dim=-1)
    if wasserstein_version == 1:
        w_loss_output = torch.abs(student_cdf - teacher_cdf) * vocab_indices
        w_loss_output = w_loss_output.sum(dim=-1)
    else:
        w_loss_output = (student_cdf - teacher_cdf) ** 2 * vocab_indices
        w_loss_output = w_loss_output.sum(dim=-1).sqrt()
    w_loss_output = w_loss_output * mask
    loss_output = w_loss_output.sum() if reduction == "sum" else w_loss_output.sum() / mask.sum()

    # 层间表示损失（代码D - emd_rep_loss）
    stu_layer_num = len(student_reps) - 1  # 假设从第1层开始
    tea_layer_num = len(teacher_reps) - 1
    student_layer_weight = np.concatenate((student_layer_weight, np.zeros(tea_layer_num)))
    teacher_layer_weight = np.concatenate((np.zeros(stu_layer_num), teacher_layer_weight))
    totol_num = stu_layer_num + tea_layer_num
    distance_matrix = torch.zeros([totol_num, totol_num], device=device, dtype=dtype)

    for i in range(stu_layer_num):
        student_rep = student_reps[i + 1]
        for j in range(tea_layer_num):
            teacher_rep = teacher_reps[j + 1]
            tmp_loss = loss_mse(student_rep, teacher_rep)
            distance_matrix[i][j + stu_layer_num] = distance_matrix[j + stu_layer_num][i] = tmp_loss

    _, trans_matrix = emd_with_flow(student_layer_weight, teacher_layer_weight,
                                    distance_matrix.detach().cpu().numpy().astype('float64'))
    loss_rep = torch.sum(torch.tensor(trans_matrix, device=device, dtype=dtype) * distance_matrix)

    # 总损失
    loss_total = alpha * loss_output + (1 - alpha) * loss_rep
    return loss_total

# 示例调用（假设已有模型输出）
# student_logits, teacher_logits = ...
# student_reps, teacher_reps = [model(inputs, return_layers=True)[i] for i in range(num_layers)]
# loss = combined_wasserstein_loss(student_logits, teacher_logits, target, student_reps, teacher_reps, ...)
# loss = combined_wasserstein_loss(student_logits, teacher_logits, target, student_reps, teacher_reps, ...)