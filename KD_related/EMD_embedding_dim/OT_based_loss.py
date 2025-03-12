import torch
import torch.nn.functional as F
import ot  # POT (Python Optimal Transport) library for EMD_diff_probability calculation


def align_embeddings(teacher_emb, student_emb, method="linear"):
    """
    对齐教师模型和学生模型的嵌入维度。
    teacher_emb: (V_teacher, D_teacher)
    student_emb: (V_student, D_student)
    method: 对齐方法，可选 "linear"（线性投影） 或 "pca"（主成分分析）
    词嵌入维度对齐（align_embeddings）：
    - 如果维度相同，直接返回；
    - 使用线性投影 (Linear) 或 PCA 进行降维。
    """
    # D_teacher, D_student = teacher_emb.shape[1], student_emb.shape[1]

    V_teacher, D_teacher = teacher_emb.shape
    V_student, D_student = student_emb.shape

    if D_teacher == D_student:
        return teacher_emb  # 维度相同，直接返回

    if method == "linear":
        projection = torch.nn.Linear(D_teacher, D_student, bias=False).to(teacher_emb.device)
        return projection(teacher_emb)
    elif method == "pca":
        # u, s, v = torch.pca_lowrank(teacher_emb, q=D_student)
        # return teacher_emb @ v  # 低维投影

        # 添加中心化和检查
        teacher_emb_centered = teacher_emb - teacher_emb.mean(dim=0, keepdim=True)
        try:
            u, s, v = torch.pca_lowrank(teacher_emb_centered, q=D_student)
            return teacher_emb @ v
        except RuntimeError:
            raise ValueError("PCA computation failed, possibly due to singular matrix")
    else:
        raise ValueError("Unsupported alignment method")


def compute_emd_loss(
        teacher_logits,
        student_logits,
        teacher_emb,
        student_emb,
        temperature=2.0, reduction='sum', mask=None):
    """
    计算基于 EMD_diff_probability (Wasserstein 距离) 的蒸馏损失
    teacher_logits: (batch, V_teacher)
    student_logits: (batch, V_student)
    teacher_emb: (V_teacher, D_teacher)
    student_emb: (V_student, D_student)

    计算 EMD_diff_probability（compute_emd_loss）：
    - Softmax 归一化 logits，转化为概率分布。
    - 嵌入维度对齐，让教师和学生嵌入可以比较。
    - 计算代价矩阵（Cost Matrix），使用余弦相似度衡量 token 之间的距离。
    - 求解最优传输矩阵，使用 ot.emd 计算 最优传输。
    - 计算 Wasserstein-1 距离 作为最终损失。
    """
    # 输入验证
    if teacher_logits.shape[0] != student_logits.shape[0]:
        raise ValueError("Batch sizes of teacher and student logits must match")

    batch_size = teacher_logits.shape[0]
    dtype = teacher_logits.dtype

    # Step 1: 归一化 logits 转换为概率分布
    teacher_probs = F.softmax(teacher_logits/temperature, dim=-1)  # (batch, V_teacher)
    student_probs = F.softmax(student_logits/temperature, dim=-1)  # (batch, V_student)

    # 检查概率分布是否有效
    if not torch.allclose(teacher_probs.sum(dim=-1), torch.ones(batch_size, device=teacher_logits.device), atol=1e-6):
        teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True)
    if not torch.allclose(student_probs.sum(dim=-1), torch.ones(batch_size, device=student_logits.device), atol=1e-6):
        student_probs = student_probs / student_probs.sum(dim=-1, keepdim=True)

    # Step 2: 对齐教师嵌入维度到学生维度
    teacher_emb_aligned = align_embeddings(teacher_emb, student_emb)

    # Step 3: 计算代价矩阵 (Cost Matrix) C(i, j) = 1 - cos(emb_i, emb_j)
    cost_matrix = 1 - F.cosine_similarity(
        teacher_emb_aligned.unsqueeze(1), student_emb.unsqueeze(0), dim=-1
    )  # (V_teacher, V_student)

    # # Step 4: 计算最优传输矩阵 T (Optimal Transport Plan)
    # emd_loss = 0
    # for i in range(batch_size):
    #     T = ot.emd(teacher_probs[i].detach().cpu().numpy(),
    #                student_probs[i].detach().cpu().numpy(),
    #                cost_matrix.detach().cpu().numpy())  # 计算最优传输矩阵
    #
    #     # Step 5: 计算 Wasserstein-1 距离
    #     emd_loss += torch.sum(torch.tensor(T, device=teacher_logits.device) * cost_matrix)
    #
    # return emd_loss / batch_size  # 归一化损失

    # 4. Compute optimal transport for all batch elements
    emd_loss = torch.zeros(batch_size, device=teacher_logits.device,dtype=dtype)

    for i in range(batch_size):
        if mask is not None and mask[i] == 0:
            emd_loss[i] = 0.0
            continue  # Skip masked elements (padding)




        T = ot.emd(
            teacher_probs[i].detach().cpu().numpy(),
            student_probs[i].detach().cpu().numpy(),
            cost_matrix.detach().cpu().numpy()
        )  # Compute optimal transport plan

        # Compute Wasserstein-1 distance
        emd_loss[i] = torch.sum(torch.tensor(T, device=teacher_logits.device) * cost_matrix)

        # Step 5: 应用 reduction
        if reduction == 'mean':
            valid_count = batch_size if mask is None else (mask != 0).sum()
            return emd_loss.sum() / max(valid_count, 1)  # 避免除以0
        elif reduction == 'sum':
            return emd_loss.sum()
        elif reduction == 'none':
            return emd_loss
        else:
            raise ValueError(f"Unsupported reduction type: {reduction}")
