import torch
import torch.nn.functional as F
import ot


def compute_wasserstein_loss(
        logits,  # 学生模型输出 [batch_size, seq_length, student_vocab_size]
        teacher_logits,  # 教师模型输出 [batch_size, seq_length, teacher_vocab_size]
        target,  # 目标标签 [batch_size, seq_length]
        teacher_emb,  # 教师词嵌入矩阵 [teacher_vocab_size, embed_dim]
        student_emb,  # 学生词嵌入矩阵 [student_vocab_size, embed_dim]
        padding_id=-100,
        reduction="sum",
        temp=2.0,
        wasserstein_version=1,
        sinkhorn_reg=0.1,
        eps=1e-9
):
    """
    计算基于Wasserstein距离的知识蒸馏损失。
    """
    # 输入验证
    assert wasserstein_version in [1, 2], "wasserstein_version must be 1 or 2"
    assert reduction in ["sum", "mean"], "reduction must be 'sum' or 'mean'"
    device = logits.device

    # batch_size, seq_length = target.shape
    # teacher_vocab_size, embed_dim = teacher_emb.shape


    student_vocab_size, _ = student_emb.shape

    # 计算成本矩阵（语义距离）
    with torch.no_grad():
        teacher_emb_norm = F.normalize(teacher_emb, p=2, dim=1)
        student_emb_norm = F.normalize(student_emb, p=2, dim=1)
        C = 1 - torch.mm(teacher_emb_norm, student_emb_norm.t())
        if wasserstein_version == 2:
            C = C ** 2
        C = C.to(device)

    # 向量化处理所有非padding位置
    mask = (target != padding_id)  # [batch_size, seq_length]
    valid_indices = mask.nonzero(as_tuple=False)  # [num_valid, 2]
    if valid_indices.shape[0] == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 提取有效的logits
    p_logits = teacher_logits[valid_indices[:, 0], valid_indices[:, 1]]  # [num_valid, teacher_vocab]
    q_logits = logits[valid_indices[:, 0], valid_indices[:, 1]]  # [num_valid, student_vocab]

    # 计算概率分布
    p = F.softmax(p_logits / temp, dim=-1) + eps  # [num_valid, teacher_vocab]
    q = F.softmax(q_logits / temp, dim=-1) + eps  # [num_valid, student_vocab]

    # 逐样本计算Sinkhorn计划
    total_loss = 0.0
    for i in range(valid_indices.shape[0]):
        p_i = p[i]  # [teacher_vocab]
        q_i = q[i]  # [student_vocab]
        # 计算Sinkhorn计划
        gamma_i = ot.sinkhorn(
            p_i, q_i, C, sinkhorn_reg, numItermax=50, epsilon=eps, verbose=False
        )  # [teacher_vocab, student_vocab]
        # 计算该时间步的损失
        loss_i = torch.sum(gamma_i * C)
        total_loss += loss_i

    # 归约处理
    if reduction == "mean":
        total_loss = total_loss / (valid_indices.shape[0] + eps)
    return total_loss


# 示例用法
if __name__ == "__main__":
    # 模拟数据
    batch_size, seq_length = 2, 3
    teacher_vocab_size, student_vocab_size, embed_dim = 300, 100, 32

    logits = torch.randn(batch_size, seq_length, student_vocab_size)
    teacher_logits = torch.randn(batch_size, seq_length, teacher_vocab_size)
    target = torch.tensor([[0, 1, -100], [2, -100, -100]])
    teacher_emb = torch.randn(teacher_vocab_size, embed_dim)
    student_emb = torch.randn(student_vocab_size, embed_dim)

    loss = compute_wasserstein_loss(
        logits, teacher_logits, target, teacher_emb, student_emb
    )
    print(f"Loss: {loss.item()}")