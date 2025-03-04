import torch

# 前向kl散度
def compute_fkl(
        logits,# 学生模型的输出。
        teacher_logits,#教师模型的输出。
        target,#目标标签。
        padding_id,#用于填充的ID。
        reduction="sum",#指定如何处理多个样本的KL散度，默认为"sum"
        temp=2.0,#温度参数，用于调整概率分布的平滑度。
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


'''
reduction参数：
选择依据：
`'none'`：用于自定义损失计算，比如需要对不同样本赋予不同权重时。
`'sum'`：适用于需要整体 KL 散度总量的情况，比如在某些优化目标中。
`'batchmean'`（推荐）：适用于批量训练，确保梯度稳定，不随 batch size 变化。
`'mean'`：适用于需要全局归一化的情况，但不如 `'batchmean'` 常见。

作用：
决定了如何聚合计算得到的 KL 散度值。这个参数在 torch.nn.KLDivLoss 中尤为重要：
一个例子：
P = torch.tensor([[0.4, 0.6], [0.3, 0.7]], dtype=torch.float)
Q = torch.tensor([[0.5, 0.5], [0.6, 0.4]], dtype=torch.float)

'none'（不聚合）：返回逐元素 KL 散度，不进行任何求和或均值操作。
kl_loss = F.kl_div(Q.log(), P, reduction='none')
结果为：
tensor([[ 0.0213, -0.0361],
        [ 0.0850, -0.1244]])

'sum'（求和）：对所有 KL 散度值求和，得到一个标量。
kl_loss = F.kl_div(Q.log(), P, reduction='sum')，对 reduction='none' 
计算过程：
计算结果的所有元素求和。
-0.0542

'batchmean'（批量均值）：先对每个样本的 KL 散度求和，然后取批量均值（PyTorch 默认）。
kl_loss = F.kl_div(Q.log(), P, reduction='batchmean')
计算过程：
第一行 KL 散度：0.0213+(−0.0361)=−0.01480.0213 + (-0.0361) = -0.01480.0213+(−0.0361)=−0.0148
第二行 KL 散度：0.0850+(−0.1244)=−0.03940.0850 + (-0.1244) = -0.03940.0850+(−0.1244)=−0.0394
批量均值：(−0.0148+(−0.0394))/2=−0.0271(-0.0148 + (-0.0394)) / 2 = -0.0271(−0.0148+(−0.0394))/2=−0.0271
-0.0271

'mean'（全局均值）：对所有 KL 散度值求均值。
kl_loss = F.kl_div(Q.log(), P, reduction='mean')
计算过程：
计算 `reduction='none'` 的所有元素之和：−0.0542
除以元素总数 4：−0.0542/4=−0.0135
-0.0135
'''