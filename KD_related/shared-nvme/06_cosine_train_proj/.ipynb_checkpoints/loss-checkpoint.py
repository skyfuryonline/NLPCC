# import torch
# import torch.nn.functional as F
# from torch import nn
# from geomloss import SamplesLoss
# from torch import amp

# class ProjectionMatrix(nn.Module):
#     def __init__(self, teacher_dim, student_dim):
#         super().__init__()
#         self.proj_matrix = nn.Parameter(torch.randn(teacher_dim, student_dim) * 0.01)  # 可训练投影矩阵

#     def forward(self, x):
#         return x @ self.proj_matrix


# def compute_ot_loss_improved(student_logits, teacher_logits, student_embeddings, teacher_embeddings,
#                              proj_layer=None, target=None, padding_id=None, topk=50, temp=2.0,
#                              reduction='sum', chunk_size=4, reg=0.1):
#     device = student_logits.device
#     batch_size, seq_len, vocab_size = student_logits.shape

#     # 训练时确保 `proj_layer` 存在
#     if proj_layer is None:
#         raise ValueError("Projection layer must be provided and should be a trainable module.")

#     # 初始化 geomloss 的 Sinkhorn 损失
#     ot_loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=reg, scaling=0.9)

#     total_loss = 0.0
#     for batch_start in range(0, batch_size, chunk_size):
#         batch_end = min(batch_start + chunk_size, batch_size)
#         chunk_student_logits = student_logits[batch_start:batch_end]
#         chunk_teacher_logits = teacher_logits[batch_start:batch_end]

#         with amp.autocast('cuda'):  # 修正此处
#             student_temp = temp * (chunk_student_logits.var(dim=-1).mean() + 1e-6).sqrt()
#             teacher_temp = temp * (chunk_teacher_logits.var(dim=-1).mean() + 1e-6).sqrt()
#             student_probs = F.softmax(chunk_student_logits / student_temp, dim=-1)
#             teacher_probs = F.softmax(chunk_teacher_logits / teacher_temp, dim=-1)

#             student_probs_topk, indices = student_probs.topk(topk, dim=-1)
#             teacher_probs_topk = torch.gather(teacher_probs, -1, indices)

#             student_topk_embeddings = student_embeddings[indices]
#             teacher_topk_embeddings = teacher_embeddings[indices]

#             # 使用可训练投影层
#             teacher_topk_embeddings_proj = proj_layer(teacher_topk_embeddings.view(-1, teacher_embeddings.shape[-1]))
#             teacher_topk_embeddings_proj = teacher_topk_embeddings_proj.view(chunk_size, seq_len, topk, -1)

#             student_topk_embeddings_flat = student_topk_embeddings.view(-1, topk, student_embeddings.shape[-1])
#             teacher_topk_embeddings_proj_flat = teacher_topk_embeddings_proj.view(-1, topk, student_embeddings.shape[-1])

#             student_probs_flat = student_probs_topk.view(-1, topk)
#             teacher_probs_flat = teacher_probs_topk.view(-1, topk)

#             # **代价矩阵使用余弦相似度**
#             cost_matrix = 1 - F.cosine_similarity(
#                 student_topk_embeddings_flat, teacher_topk_embeddings_proj_flat, dim=-1
#             )

#             # **Wasserstein-1 距离（已注释）**
#             # cost_matrix = torch.cdist(student_topk_embeddings_flat, teacher_topk_embeddings_proj_flat, p=1)

#             # 计算 OT 损失
#             ot_loss = ot_loss_fn(student_probs_flat, teacher_probs_flat, cost_matrix)

#             if target is not None and padding_id is not None:
#                 padding_mask = (target[batch_start:batch_end] != padding_id).float().view(-1)
#                 ot_loss = ot_loss * padding_mask

#             total_loss += ot_loss.sum()

#     if reduction == 'mean':
#         return total_loss / (batch_size * seq_len)
#     elif reduction == 'sum':
#         return total_loss
#     return total_loss



import torch
import torch.nn.functional as F
from torch import nn
from geomloss import SamplesLoss
from torch import amp

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

    # 定义一个自定义余弦距离 cost 函数，输入 x, y 均形状 (N, topk, embedding_dim)
    # 计算两组点之间的成批余弦距离，输出形状 (N, topk, topk)
    custom_cosine_cost = lambda x, y: 1 - torch.bmm(F.normalize(x, dim=-1), F.normalize(y, dim=-1).transpose(1, 2))
    
    # 初始化 geomloss 的 Sinkhorn 损失，传入自定义 cost 函数
    ot_loss_fn = SamplesLoss(loss="sinkhorn", cost=custom_cosine_cost, blur=reg, scaling=0.9)

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

            # 选取 topk 的概率和对应的索引
            student_probs_topk, indices = student_probs.topk(topk, dim=-1)
            teacher_probs_topk = torch.gather(teacher_probs, -1, indices)

            # 根据相同的索引从嵌入中选取对应的 topk 项
            student_topk_embeddings = student_embeddings[indices]
            teacher_topk_embeddings = teacher_embeddings[indices]

            # 使用可训练投影层对 teacher 的 topk 嵌入进行投影
            teacher_topk_embeddings_proj = proj_layer(teacher_topk_embeddings.view(-1, teacher_embeddings.shape[-1]))
            teacher_topk_embeddings_proj = teacher_topk_embeddings_proj.view(chunk_size, seq_len, topk, -1)

            # 将嵌入展平为 (N, topk, embedding_dim) 形式，其中 N = chunk_size * seq_len
            student_topk_embeddings_flat = student_topk_embeddings.view(-1, topk, student_embeddings.shape[-1])
            teacher_topk_embeddings_proj_flat = teacher_topk_embeddings_proj.view(-1, topk, student_embeddings.shape[-1])

            # 概率展平为 (N, topk)
            student_probs_flat = student_probs_topk.view(-1, topk)
            teacher_probs_flat = teacher_probs_topk.view(-1, topk)

            # 使用四参数形式调用 ot_loss_fn：
            # (α, x, β, y) 分别为权重和对应的点云，其中内部 cost 使用我们自定义的余弦距离函数
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
