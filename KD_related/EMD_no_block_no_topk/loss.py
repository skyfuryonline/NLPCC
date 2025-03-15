import torch
import torch.nn.functional as F
from geomloss import SamplesLoss

def compute_ot_loss_improved(student_logits, teacher_logits, student_embeddings, teacher_embeddings,
                            proj_matrix=None, target=None, padding_id=None, temp=2.0,
                            reduction='sum', reg=0.1):
    """
    计算 student 和 teacher 之间的 OT 损失，支持词表大小不一致。
    参数:
        student_logits: (batch_size, seq_len, vocab_size_s)
        teacher_logits: (batch_size, seq_len, vocab_size_t)
        student_embeddings: (vocab_size_s, student_embed_dim)
        teacher_embeddings: (vocab_size_t, teacher_embed_dim)
    """
    device = student_logits.device
    batch_size, seq_len, vocab_size_s = student_logits.shape
    _, _, vocab_size_t = teacher_logits.shape
    student_embed_dim = student_embeddings.shape[-1]
    teacher_embed_dim = teacher_embeddings.shape[-1]

    # 投影矩阵初始化
    if proj_matrix is None:
        proj_matrix = torch.nn.init.orthogonal_(torch.empty(teacher_embed_dim, student_embed_dim, device=device))

    # 初始化 geomloss 的 Sinkhorn 损失，支持批量计算
    ot_loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=reg, scaling=0.9, backend="tensorized")

    with torch.cuda.amp.autocast():
        # 计算温度调整后的概率分布
        student_temp = temp * (student_logits.var(dim=-1).mean() + 1e-6).sqrt()
        teacher_temp = temp * (teacher_logits.var(dim=-1).mean() + 1e-6).sqrt()
        student_probs = F.softmax(student_logits / student_temp, dim=-1)  # (batch_size, seq_len, vocab_size_s)
        teacher_probs = F.softmax(teacher_logits / teacher_temp, dim=-1)  # (batch_size, seq_len, vocab_size_t)

        # 展平概率分布为 (batch_size * seq_len, vocab_size)
        student_probs_flat = student_probs.view(-1, vocab_size_s)  # (batch_size * seq_len, vocab_size_s)
        teacher_probs_flat = teacher_probs.view(-1, vocab_size_t)  # (batch_size * seq_len, vocab_size_t)

        # 投影 teacher_embeddings 到 student 的嵌入空间
        teacher_embeddings_proj = torch.matmul(teacher_embeddings, proj_matrix)  # (vocab_size_t, student_embed_dim)

        # 扩展 embeddings 为批量形式，匹配 probs 的样本数
        student_embeddings_exp = student_embeddings.unsqueeze(0).expand(batch_size * seq_len, vocab_size_s, student_embed_dim)
        teacher_embeddings_exp = teacher_embeddings_proj.unsqueeze(0).expand(batch_size * seq_len, vocab_size_t, student_embed_dim)

        # 批量计算 OT 损失
        ot_loss = ot_loss_fn(student_probs_flat, student_embeddings_exp,
                           teacher_probs_flat, teacher_embeddings_exp)  # (batch_size * seq_len,)

        # 处理 padding（如果提供）
        if target is not None and padding_id is not None:
            padding_mask = (target != padding_id).float().view(-1)  # (batch_size * seq_len,)
            ot_loss = ot_loss * padding_mask
            total_loss = ot_loss.sum()
            if reduction == 'mean':
                total_loss = total_loss / padding_mask.sum()  # 只对非 padding 部分求均值
        else:
            total_loss = ot_loss.sum()

    if reduction == 'mean' and (target is None or padding_id is None):
        return total_loss / (batch_size * seq_len)
    elif reduction == 'sum':
        return total_loss
    return total_loss

# 测试代码
if __name__ == "__main__":
    # 测试词表大小不一致
    batch_size, seq_len = 2, 3
    vocab_size_s, vocab_size_t = 100, 150
    student_embed_dim, teacher_embed_dim = 64, 128

    student_logits = torch.randn(batch_size, seq_len, vocab_size_s, device='cuda')
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size_t, device='cuda')
    student_embeddings = torch.randn(vocab_size_s, student_embed_dim, device='cuda')
    teacher_embeddings = torch.randn(vocab_size_t, teacher_embed_dim, device='cuda')

    loss = compute_ot_loss_improved(student_logits, teacher_logits, student_embeddings, teacher_embeddings)
    print(f"Loss: {loss.item()}")