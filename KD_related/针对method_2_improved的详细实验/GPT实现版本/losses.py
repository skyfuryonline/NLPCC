import math
import torch
import torch.nn.functional as F


def ot_distillation_loss_logits(student_logits, teacher_logits, projection,
                                temperature=1.0, lambda_reg=1.0,
                                reduction="mean", target=None, padding_id=None,
                                block_size=1024, num_sinkhorn_iters=50):
    """
    OT 蒸馏损失：直接对齐学生和教师的 logits

    参数说明：
      student_logits: 学生模型的 logits，形状 (N, d_stu)
      teacher_logits: 教师模型的 logits，形状 (M, d_tea)
      projection: 投影矩阵 P，形状 (d_tea, d_stu)，用于将教师 logits 映射到学生空间
      temperature: 温度参数，用于缩放相似性（确保与训练时一致）
      lambda_reg: 熵正则化超参数，控制正则化项对 OT 的影响
      reduction: 损失聚合方式，可选 "mean" 或 "sum"
      target: 学生对应的 token id 序列，用于过滤 padding token（形状 (N,)）
      padding_id: padding token 的 id
      block_size: 分块计算时每块的大小，以降低显存占用
      num_sinkhorn_iters: Sinkhorn 算法迭代次数

    返回：
      OT 损失（标量）
    """
    # 如果提供 target 和 padding_id，则过滤掉学生中 padding 的 logits
    if target is not None and padding_id is not None:
        student_mask = (target != padding_id)  # 布尔 mask, 形状 (N,)
        X = student_logits[student_mask]  # 形状 (N_valid, d_stu)
    else:
        X = student_logits
    # 假设教师 logits 全部有效
    Y = teacher_logits
    N_valid = X.shape[0]
    M_valid = Y.shape[0]

    # 将教师 logits 通过投影矩阵映射到学生空间，得到 Y_proj，形状 (M_valid, d_stu)
    Y_proj = torch.matmul(Y, projection)

    # 计算相似性矩阵 S = (X dot Y_proj^T) / (sqrt(d_stu) * temperature)
    d_stu = X.shape[1]
    scale = 1.0 / (math.sqrt(d_stu) * temperature)

    S_rows = []
    # 分块计算以降低显存压力
    for i in range(0, N_valid, block_size):
        X_block = X[i: i + block_size]  # 形状 (B, d_stu)
        # 计算当前块与整个教师 logits 的内积，形状 (B, M_valid)
        S_block = torch.matmul(X_block, Y_proj.t()) * scale
        # 对每一行进行 softmax 归一化，保证每一行形成概率分布
        S_norm_block = F.softmax(S_block, dim=1)
        S_rows.append(S_norm_block)
    S_norm = torch.cat(S_rows, dim=0)  # 形状 (N_valid, M_valid)

    # 定义成本矩阵：C = 1 - S_norm
    C = 1.0 - S_norm  # 数值越小表示相似度越高

    # 定义经验分布：学生和教师均为均匀分布
    a = torch.full((N_valid,), 1.0 / N_valid, device=X.device)
    b = torch.full((M_valid,), 1.0 / M_valid, device=Y_proj.device)

    # 求解带熵正则化的 OT 问题，使用 Sinkhorn 算法
    # 定义核矩阵：K = exp(-lambda_reg * C)
    K = torch.exp(-lambda_reg * C)  # 形状 (N_valid, M_valid)

    # 初始化对偶变量 u 和 v
    u = torch.ones_like(a)
    v = torch.ones_like(b)

    # Sinkhorn 迭代更新 u 和 v
    for _ in range(num_sinkhorn_iters):
        u = a / (torch.matmul(K, v) + 1e-8)
        v = b / (torch.matmul(K.t(), u) + 1e-8)

    # 计算最优传输计划：T = diag(u) * K * diag(v)
    T = torch.diag(u) @ K @ torch.diag(v)

    # OT 损失定义为内积 <T, C> = sum(T * C)
    loss_ot = torch.sum(T * C)

    # 根据 reduction 参数进行聚合
    if reduction == "mean":
        loss = loss_ot / (N_valid * M_valid)
    elif reduction == "sum":
        loss = loss_ot
    else:
        loss = loss_ot
    return loss
