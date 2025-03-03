import torch

# 前向kl散度
def compute_fkl(
        logits,# 学生模型的输出。
        teacher_logits,#教师模型的输出。
        target,#目标标签。
        padding_id,#用于填充的ID。
        reduction="sum",#指定如何处理多个样本的KL散度，默认为"sum"
        temp=1.0,#温度参数，用于调整概率分布的平滑度。
):
    # 将logits和teacher_logits除以温度参数temp，用于调整概率分布的平滑度
    logits = logits / temp
    teacher_logits = teacher_logits / temp

    # 计算logits的对数softmax值，结果是log概率
    log_probs = torch.log_softmax(logits, -1, dtype=torch.float32)
    # 计算teacher_logits的softmax值，结果是概率
    teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
    # 计算teacher_logits的对数softmax值，结果是对数概率
    teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
    # 计算前向KL散度：教师模型的概率分布与学生模型的概率分布之间的差异
    kl = (teacher_probs * (teacher_log_probs - log_probs))
    # 对每个样本的KL散度在最后一个维度上求和
    kl = kl.sum(-1)
    # 如果reduction为"sum"，则进行以下操作
    if reduction == "sum":
        # 创建一个掩码，标记目标中等于padding_id的位置
        pad_mask = target.eq(padding_id)
        # 将掩码位置的KL散度值设为0.0
        kl = kl.masked_fill_(pad_mask, 0.0)
        # 对所有样本的KL散度求和
        kl = kl.sum()
    # 返回计算得到的KL散度
    return kl