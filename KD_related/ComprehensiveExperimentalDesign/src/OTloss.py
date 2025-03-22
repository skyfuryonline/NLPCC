import torch
import torch.nn.functional as F
from torch import amp
from geomloss import SamplesLoss

def OT_loss(student_logits, teacher_logits, student_embeddings, teacher_embeddings,
                            proj_matrix=None, target=None, padding_id=None, topk=50, temp=2.0,
                            reduction='sum', chunk_size=4, reg=0.1):
    device = student_logits.device
    batch_size, seq_len, vocab_size = student_logits.shape
    student_embed_dim = student_embeddings.shape[-1]
    teacher_embed_dim = teacher_embeddings.shape[-1]

    if proj_matrix is None:
        proj_matrix = torch.nn.init.orthogonal_(torch.empty(teacher_embed_dim, student_embed_dim, device=device))

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
            teacher_topk_embeddings_proj = torch.bmm(teacher_topk_embeddings.view(-1, topk, teacher_embed_dim),
                                                    proj_matrix.expand(chunk_size * seq_len, -1, -1))
            teacher_topk_embeddings_proj = teacher_topk_embeddings_proj.view(chunk_size, seq_len, topk, student_embed_dim)

            student_topk_embeddings_flat = student_topk_embeddings.view(-1, topk, student_embed_dim)
            teacher_topk_embeddings_proj_flat = teacher_topk_embeddings_proj.view(-1, topk, student_embed_dim)

            student_probs_flat = student_probs_topk.view(-1, topk)
            teacher_probs_flat = teacher_probs_topk.view(-1, topk)

            # 使用 geomloss 计算批量 Sinkhorn 损失
            ot_loss = ot_loss_fn(student_probs_flat, student_topk_embeddings_flat,
                               teacher_probs_flat, teacher_topk_embeddings_proj_flat)

            if target is not None and padding_id is not None:
                padding_mask = (target[batch_start:batch_end] != padding_id).float().view(-1)
                ot_loss = ot_loss * padding_mask

            total_loss += ot_loss.sum()

    if reduction == 'mean':
        return total_loss / (batch_size * seq_len)
    elif reduction == 'sum':
        return total_loss
    return total_loss