import torch
import torch.nn.functional as F
from torch import amp
from geomloss import SamplesLoss

def compute_ot_kl_loss(student_logits, teacher_logits, student_embeddings, teacher_embeddings,
                      target=None, padding_id=None, topk=50, temp=2.0, reduction='sum',
                      chunk_size=4, reg=0.1, alpha=0.5):
    """
    混合 OT 和 KL 的知识蒸馏损失函数。

    Args:
        student_logits (torch.Tensor): 学生模型的 logits，形状 (batch_size, seq_len, vocab_size)
        teacher_logits (torch.Tensor): 教师模型的 logits，形状 (batch_size, seq_len, vocab_size)
        student_embeddings (nn.Embedding): 学生模型的词嵌入层
        teacher_embeddings (nn.Embedding): 教师模型的词嵌入层
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
    student_embed_dim = student_embeddings.embedding_dim  # 1536
    teacher_embed_dim = teacher_embeddings.embedding_dim  # 3584

    # 初始化 geomloss 的 Sinkhorn 损失
    ot_loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=reg, scaling=0.9)
    total_loss = 0.0

    # 分块处理以控制显存
    for batch_start in range(0, batch_size, chunk_size):
        batch_end = min(batch_start + chunk_size, batch_size)
        chunk_student_logits = student_logits[batch_start:batch_end]
        chunk_teacher_logits = teacher_logits[batch_start:batch_end]

        with amp.autocast('cuda'):  # 使用混合精度加速
            # 自适应温度调整
            student_temp = temp * (chunk_student_logits.var(dim=-1).mean() + 1e-6).sqrt()
            teacher_temp = temp * (chunk_teacher_logits.var(dim=-1).mean() + 1e-6).sqrt()
            student_probs = F.softmax(chunk_student_logits / student_temp, dim=-1)
            teacher_probs = F.softmax(chunk_teacher_logits / teacher_temp, dim=-1)

            # OT 损失：基于 Top-k
            student_probs_topk, indices = student_probs.topk(topk, dim=-1)  # (chunk_size, seq_len, topk)
            teacher_probs_topk = torch.gather(teacher_probs, -1, indices)   # (chunk_size, seq_len, topk)
            student_topk_emb = student_embeddings(indices)                  # (chunk_size, seq_len, topk, 1536)
            teacher_topk_emb = teacher_embeddings(indices)                  # (chunk_size, seq_len, topk, 3584)

            # 展平用于 OT 计算
            student_probs_flat = student_probs_topk.view(-1, topk)          # (chunk_size * seq_len, topk)
            teacher_probs_flat = teacher_probs_topk.view(-1, topk)          # (chunk_size * seq_len, topk)
            student_emb_flat = student_topk_emb.view(-1, topk, student_embed_dim)
            teacher_emb_flat = teacher_topk_emb.view(-1, topk, teacher_embed_dim)

            # 计算 OT 损失
            ot_loss = ot_loss_fn(student_probs_flat, student_emb_flat,
                               teacher_probs_flat, teacher_emb_flat)      # (chunk_size * seq_len,)

            # KL 损失：基于全词汇表
            kl_loss = F.kl_div(F.log_softmax(chunk_student_logits / temp, dim=-1),
                             teacher_probs, reduction='none').sum(dim=-1)  # (chunk_size, seq_len)

            # 处理 padding
            if target is not None and padding_id is not None:
                padding_mask = (target[batch_start:batch_end] != padding_id).float()
                ot_loss = ot_loss.view(chunk_size, seq_len) * padding_mask  # 调整形状后应用 mask
                kl_loss = kl_loss * padding_mask

            # 混合损失
            total_loss += alpha * ot_loss.sum() + (1 - alpha) * kl_loss.sum()

    # 缩减方式
    if reduction == 'mean':
        return total_loss / (batch_size * seq_len)
    elif reduction == 'sum':
        return total_loss
    return total_loss

# 测试代码
if __name__ == "__main__":
    student_logits = torch.randn(16, 128, 151643).cuda()
    teacher_logits = torch.randn(16, 128, 151643).cuda()
    student_embeddings = nn.Embedding(151643, 1536).cuda()
    teacher_embeddings = nn.Embedding(151643, 3584).cuda()

    loss = compute_ot_kl_loss(student_logits, teacher_logits, student_embeddings, teacher_embeddings)
    print(f"Loss: {loss.item()}")