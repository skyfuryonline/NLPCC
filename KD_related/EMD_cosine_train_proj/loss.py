import torch
import torch.nn.functional as F
from torch import nn
from geomloss import SamplesLoss
from torch.cuda import amp


class ProjectionMatrix(nn.Module):
    def __init__(self, teacher_dim, student_dim):
        super().__init__()
        self.proj_matrix = nn.Parameter(torch.randn(teacher_dim, student_dim) * 0.01)  # 可训练投影矩阵

    def forward(self, x):
        return x @ self.proj_matrix


def compute_ot_loss_improved(student_logits, teacher_logits, student_embeddings, teacher_embeddings,
                             proj_layer=None, target=None, padding_id=None, topk=50, temp=2.0,
                             reduction='sum', chunk_size=4, reg=0.1):
    device = student_logits.device
    batch_size, seq_len, vocab_size = student_logits.shape

    # 训练时确保 `proj_layer` 存在
    if proj_layer is None:
        raise ValueError("Projection layer must be provided and should be a trainable module.")

    # 初始化 geomloss 的 Sinkhorn 损失
    ot_loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=reg, scaling=0.9)

    total_loss = 0.0
    for batch_start in range(0, batch_size, chunk_size):
        batch_end = min(batch_start + chunk_size, batch_size)
        chunk_student_logits = student_logits[batch_start:batch_end]
        chunk_teacher_logits = teacher_logits[batch_start:batch_end]

        with amp.autocast('cuda'):
            student_temp = temp * (chunk_student_logits.var(dim=-1).mean() + 1e-6).sqrt()
            teacher_temp = temp * (chunk_teacher_logits.var(dim=-1).mean() + 1e-6).sqrt()
            student_probs = F.softmax(chunk_student_logits / student_temp, dim=-1)
            teacher_probs = F.softmax(chunk_teacher_logits / teacher_temp, dim=-1)

            student_probs_topk, indices = student_probs.topk(topk, dim=-1)
            teacher_probs_topk = torch.gather(teacher_probs, -1, indices)

            student_topk_embeddings = student_embeddings[indices]
            teacher_topk_embeddings = teacher_embeddings[indices]

            # 使用可训练投影层
            teacher_topk_embeddings_proj = proj_layer(teacher_topk_embeddings.view(-1, teacher_embeddings.shape[-1]))
            teacher_topk_embeddings_proj = teacher_topk_embeddings_proj.view(chunk_size, seq_len, topk, -1)

            student_topk_embeddings_flat = student_topk_embeddings.view(-1, topk, student_embeddings.shape[-1])
            teacher_topk_embeddings_proj_flat = teacher_topk_embeddings_proj.view(-1, topk, student_embeddings.shape[-1])

            student_probs_flat = student_probs_topk.view(-1, topk)
            teacher_probs_flat = teacher_probs_topk.view(-1, topk)

            # **代价矩阵使用余弦相似度**
            cost_matrix = 1 - F.cosine_similarity(
                student_topk_embeddings_flat, teacher_topk_embeddings_proj_flat, dim=-1
            )

            # **Wasserstein-1 距离（已注释）**
            # cost_matrix = torch.cdist(student_topk_embeddings_flat, teacher_topk_embeddings_proj_flat, p=1)

            # 计算 OT 损失
            ot_loss = ot_loss_fn(student_probs_flat, teacher_probs_flat, cost_matrix)

            if target is not None and padding_id is not None:
                padding_mask = (target[batch_start:batch_end] != padding_id).float().view(-1)
                ot_loss = ot_loss * padding_mask

            total_loss += ot_loss.sum()

    if reduction == 'mean':
        return total_loss / (batch_size * seq_len)
    elif reduction == 'sum':
        return total_loss
    return total_loss
