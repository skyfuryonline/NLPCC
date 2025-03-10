import torch

# 前向kl散度
def compute_fkl(
        logits,
        teacher_logits,
        target,
        padding_id,
        reduction="sum",
        temp=1.0,

):
    logits = logits / temp
    teacher_logits = teacher_logits / temp

    log_probs = torch.log_softmax(logits, -1, dtype=torch.float32)
    teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
    teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
    kl = (teacher_probs * (teacher_log_probs - log_probs))
    kl = kl.sum(-1)
    if reduction == "sum":
        pad_mask = target.eq(padding_id)
        kl = kl.masked_fill_(pad_mask, 0.0)
        kl = kl.sum()

    return kl