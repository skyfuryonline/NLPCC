import torch
import torch.nn.functional as F
import ot


def compute_wasserstein_loss(
        logits, teacher_logits, target, teacher_emb, student_emb,
        padding_id=-100, reduction="sum", temp=1.0,  # 降低 temp
        wasserstein_version=1, sinkhorn_reg=1.0,  # 增大正则化
        num_iter_max=500,  # 增加迭代次数
        eps=1e-9, convergence_threshold=1e-6  # 更严格的阈值
):
    assert wasserstein_version in [1, 2], "wasserstein_version must be 1 or 2"
    assert reduction in ["sum", "mean"], "reduction must be 'sum' or 'mean'"
    device = logits.device

    batch_size, seq_length = target.shape
    teacher_vocab_size, embed_dim = teacher_emb.shape
    student_vocab_size, _ = student_emb.shape

    with torch.no_grad():
        teacher_emb_norm = F.normalize(teacher_emb, p=2, dim=1)
        student_emb_norm = F.normalize(student_emb, p=2, dim=1)
        C = 1 - torch.mm(teacher_emb_norm, student_emb_norm.t())
        if wasserstein_version == 2:
            C = C ** 2
        C = C.to(device)

    mask = (target != padding_id)
    valid_indices = mask.nonzero(as_tuple=False)
    if valid_indices.shape[0] == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    p_logits = teacher_logits[valid_indices[:, 0], valid_indices[:, 1]]
    q_logits = logits[valid_indices[:, 0], valid_indices[:, 1]]

    p = F.softmax(p_logits / temp, dim=-1) + eps
    q = F.softmax(q_logits / temp, dim=-1) + eps

    total_loss = 0.0
    for i in range(valid_indices.shape[0]):
        p_i = p[i]
        q_i = q[i]
        gamma_i = ot.sinkhorn(
            p_i, q_i, C, sinkhorn_reg, numItermax=num_iter_max, epsilon=eps, verbose=True  # 显示详细信息
        )
        # 详细收敛检查
        p_recon = gamma_i.sum(dim=1)
        q_recon = gamma_i.sum(dim=0)
        p_error = torch.max(torch.abs(p_i - p_recon))
        q_error = torch.max(torch.abs(q_i - q_recon))
        print(f"Sample {i}: p_error={p_error.item():.6f}, q_error={q_error.item():.6f}")
        loss_i = torch.sum(gamma_i * C)
        total_loss += loss_i

    if reduction == "mean":
        total_loss = total_loss / (valid_indices.shape[0] + eps)
    return total_loss


# 示例用法
if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size, seq_length = 2, 3
    teacher_vocab_size, student_vocab_size, embed_dim = 300, 100, 32

    logits = torch.randn(batch_size, seq_length, student_vocab_size)
    teacher_logits = torch.randn(batch_size, seq_length, teacher_vocab_size)
    target = torch.tensor([[0, 1, -100], [2, -100, -100]])
    teacher_emb = torch.randn(teacher_vocab_size, embed_dim)
    student_emb = torch.randn(student_vocab_size, embed_dim)

    for run in range(3):
        loss = compute_wasserstein_loss(
            logits, teacher_logits, target, teacher_emb, student_emb
        )
        print(f"Run {run + 1} Loss: {loss.item()}")


