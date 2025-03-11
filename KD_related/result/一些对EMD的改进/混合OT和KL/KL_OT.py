import torch
import torch.nn.functional as F
from torch import amp
from geomloss import SamplesLoss


def compute_ot_kl_loss(student_logits, teacher_logits, student_embeddings, teacher_embeddings, 
                      projection, target=None, padding_id=None, topk=50, temp=2.0, reduction='sum', 
                      chunk_size=4, reg=0.1, alpha=0.5):
    """
    混合 OT 和 KL 的知识蒸馏损失函数。

    Args:
        student_logits (torch.Tensor): 学生模型的 logits，形状 (batch_size, seq_len, vocab_size)
        teacher_logits (torch.Tensor): 教师模型的 logits，形状 (batch_size, seq_len, vocab_size)
        student_embeddings (torch.Tensor): 学生模型的词嵌入权重，形状 (vocab_size, student_embed_dim)
        teacher_embeddings (torch.Tensor): 教师模型的词嵌入权重，形状 (vocab_size, teacher_embed_dim)
        projection (nn.Module): 可训练的投影层，将学生嵌入映射到教师嵌入维度
        target (torch.Tensor, optional): 目标标签，形状 (batch_size, seq_len)
        padding_id (int, optional): Padding 的 token ID
        topk (int): OT 计算使用的 Top-k 词汇数
        temp (float): 温度系数
        reduction (str): 损失缩减方式 ['mean', 'sum', 'none']
        chunk_size (int): 分块大小
        reg (float): Sinkhorn 正则化参数
        alpha (float): OT 和 KL 损失的加权系数 (0 <= alpha <= 1)

    Returns:
        torch.Tensor: 混合损失
    """
    device = student_logits.device
    batch_size, seq_len, vocab_size = student_logits.shape
    student_embed_dim = student_embeddings.shape[-1]  # 1536
    teacher_embed_dim = teacher_embeddings.shape[-1]  # 3584

    ot_loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=reg, scaling=0.9)
    total_loss = 0.0

    for batch_start in range(0, batch_size, chunk_size):
        batch_end = min(batch_start + chunk_size, batch_size)
        chunk_student_logits = student_logits[batch_start:batch_end]
        chunk_teacher_logits = teacher_logits[batch_start:batch_end]

        with torch.cuda.amp.autocast():
            student_temp = temp * (chunk_student_logits.var(dim=-1).mean() + 1e-6).sqrt()
            teacher_temp = temp * (chunk_teacher_logits.var(dim=-1).mean() + 1e-6).sqrt()
            student_probs = F.softmax(chunk_student_logits / student_temp, dim=-1)
            teacher_probs = F.softmax(chunk_teacher_logits / teacher_temp, dim=-1)

            student_probs_topk, indices = student_probs.topk(topk, dim=-1)
            teacher_probs_topk = torch.gather(teacher_probs, -1, indices)
            student_topk_emb = F.embedding(indices, student_embeddings)
            teacher_topk_emb = F.embedding(indices, teacher_embeddings)

            student_probs_flat = student_probs_topk.view(-1, topk)
            teacher_probs_flat = teacher_probs_topk.view(-1, topk)
            student_emb_flat = student_topk_emb.view(-1, topk, student_embed_dim)
            teacher_emb_flat = teacher_topk_emb.view(-1, topk, teacher_embed_dim)

            # 使用可训练的投影层
            student_emb_flat_proj = projection(student_emb_flat)  # (chunk_size * seq_len, topk, 3584)

            ot_loss = ot_loss_fn(student_probs_flat, student_emb_flat_proj, 
                               teacher_probs_flat, teacher_emb_flat)

            kl_loss = F.kl_div(F.log_softmax(chunk_student_logits / temp, dim=-1), 
                             teacher_probs, reduction='none').sum(dim=-1)

            if target is not None and padding_id is not None:
                padding_mask = (target[batch_start:batch_end] != padding_id).float()
                ot_loss = ot_loss.view(batch_end - batch_start, seq_len) * padding_mask
                kl_loss = kl_loss * padding_mask

            total_loss += alpha * ot_loss.sum() + (1 - alpha) * kl_loss.sum()

    if reduction == 'mean':
        return total_loss / (batch_size * seq_len)
    elif reduction == 'sum':
        return total_loss
    return total_loss