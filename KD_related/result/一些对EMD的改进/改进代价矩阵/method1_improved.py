#当教师模型和学生模型的嵌入维度不一致时（例如，学生模型嵌入维度为 1536，教师模型为 3584），直接基于词嵌入计算语义距离（如余弦距离）会面临维度不匹配的问题。
#为了解决这个问题，我们需要将两者的嵌入投影到同一空间中，或者使用其他方法对齐语义表示。


import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

def compute_ot_loss_topk_sinkhorn(student_logits, teacher_logits, student_embeddings, teacher_embeddings, proj_matrix=None, target=None, padding_id=None, topk=50, temp=2.0, epsilon=0.1, reduction='sum', max_iter=20):
    """
    计算基于 Top-k 词的 Optimal Transport (OT) 知识蒸馏损失，优化用于单张 4090 显卡。

    Args:
        student_logits (torch.Tensor): 学生模型 logits, (batch_size, seq_len, vocab_size)
        teacher_logits (torch.Tensor): 教师模型 logits, (batch_size, seq_len, vocab_size)
        student_embeddings (torch.Tensor): 学生模型词嵌入, (vocab_size, student_embed_dim)
        teacher_embeddings (torch.Tensor): 教师模型词嵌入, (vocab_size, teacher_embed_dim)
        proj_matrix (torch.Tensor, optional): 预计算的投影矩阵, (teacher_embed_dim, student_embed_dim)
        target (torch.Tensor, optional): 目标 token IDs, (batch_size, seq_len)
        padding_id (int, optional): PAD token ID
        topk (int): 仅计算 top-k 个词的 Wasserstein 距离，默认减小到 100
        temp (float): 温度系数
        epsilon (float): Sinkhorn 正则化参数
        reduction (str): ['mean', 'sum', 'none']
        max_iter (int): Sinkhorn 迭代次数，默认减小到 20

    Returns:
        torch.Tensor: OT 知识蒸馏损失
    """
    device = student_logits.device
    batch_size, seq_len, vocab_size = student_logits.shape
    student_embed_dim = student_embeddings.shape[-1]  # 1536
    teacher_embed_dim = teacher_embeddings.shape[-1]  # 3584

    # **混合精度计算**
    with autocast():
        # **计算 softmax 概率分布**
        student_probs = F.softmax(student_logits / temp, dim=-1)  # (batch_size, seq_len, vocab_size)
        teacher_probs = F.softmax(teacher_logits / temp, dim=-1)  # (batch_size, seq_len, vocab_size)

        # **取 top-k**
        student_probs_topk, indices = student_probs.topk(topk, dim=-1)  # (batch_size, seq_len, topk)
        teacher_probs_topk = torch.gather(teacher_probs, -1, indices)   # (batch_size, seq_len, topk)

        # **获取 top-k 的词嵌入**
        student_topk_embeddings = student_embeddings[indices]  # (batch_size, seq_len, topk, 1536)
        teacher_topk_embeddings = teacher_embeddings[indices]  # (batch_size, seq_len, topk, 3584)

        # **投影教师嵌入到学生维度**
        if proj_matrix is None:
            # 如果未提供预计算投影矩阵，随机初始化并固定
            proj_matrix = torch.randn(teacher_embed_dim, student_embed_dim, device=device)
            proj_matrix /= proj_matrix.norm(dim=0, keepdim=True)  # 归一化
        teacher_topk_embeddings_proj = teacher_topk_embeddings @ proj_matrix  # (batch_size, seq_len, topk, 1536)

        # **基于投影后的词嵌入构造成本矩阵**
        student_topk_embeddings_norm = F.normalize(student_topk_embeddings, p=2, dim=-1)
        teacher_topk_embeddings_norm = F.normalize(teacher_topk_embeddings_proj, p=2, dim=-1)
        similarity = torch.einsum('bski,bskj->bsij', student_topk_embeddings_norm, teacher_topk_embeddings_norm)
        cost_matrix = 1 - similarity  # (batch_size, seq_len, topk, topk)

        # **Sinkhorn 算法**
        K = torch.exp(-cost_matrix / epsilon)
        u = torch.ones_like(student_probs_topk)
        for _ in range(max_iter):
            v = teacher_probs_topk / (K.transpose(-1, -2) @ u + 1e-10)
            u = student_probs_topk / (K @ v + 1e-10)
        T = u.unsqueeze(-1) * K * v.unsqueeze(-2)
        ot_loss = (T * cost_matrix).sum(dim=(-1, -2))  # (batch_size, seq_len)

        # **处理 padding**
        if target is not None and padding_id is not None:
            padding_mask = (target != padding_id).float()
            ot_loss = ot_loss * padding_mask

        # **处理 reduction**
        if reduction == 'mean':
            return ot_loss.mean()
        elif reduction == 'sum':
            return ot_loss.sum()
        return ot_loss