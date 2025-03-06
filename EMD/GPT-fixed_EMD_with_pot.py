import torch
import torch.nn.functional as F
import ot  # POT: Python Optimal Transport


def compute_wasserstein_loss(
        logits, teacher_logits, target,
        padding_id=-100, reduction="sum",
        temp=2.0, wasserstein_version=1,
        sinkhorn_reg=0.01
):
    mask = (target != padding_id).float()  # [batch_size, seq_length]

    student_vocab_size, teacher_vocab_size = logits.shape[-1], teacher_logits.shape[-1]
    max_vocab_size = max(student_vocab_size, teacher_vocab_size)

    # Pad logits to match max_vocab_size
    if student_vocab_size < max_vocab_size:
        logits = torch.cat([logits, torch.full_like(logits[..., :max_vocab_size - student_vocab_size], float('-inf'))],
                           dim=-1)
    if teacher_vocab_size < max_vocab_size:
        teacher_logits = torch.cat(
            [teacher_logits, torch.full_like(teacher_logits[..., :max_vocab_size - teacher_vocab_size], float('-inf'))],
            dim=-1)

    # Convert to probability distributions
    student_probs = (logits / temp).log_softmax(dim=-1).exp()
    teacher_probs = (teacher_logits / temp).log_softmax(dim=-1).exp()

    # Compute cost matrix
    vocab_range = torch.arange(max_vocab_size, dtype=torch.float32, device=logits.device)
    cost_matrix = torch.abs(vocab_range.unsqueeze(0) - vocab_range.unsqueeze(1)) if wasserstein_version == 1 else \
        (vocab_range.unsqueeze(0) - vocab_range.unsqueeze(1)) ** 2

    # Expand cost matrix for batch processing
    batch_size, seq_length, _ = student_probs.shape
    cost_matrix = cost_matrix.unsqueeze(0).expand(batch_size * seq_length, -1, -1)  # [B*S, V, V]

    # Flatten batch dimension for Sinkhorn computation
    student_probs_flat = student_probs.view(batch_size * seq_length, -1)
    teacher_probs_flat = teacher_probs.view(batch_size * seq_length, -1)

    # Compute Sinkhorn-Knopp distance
    try:
        emd_losses = ot.sinkhorn2(student_probs_flat, teacher_probs_flat, cost_matrix, sinkhorn_reg)
    except Exception as e:
        print(f"Sinkhorn computation failed: {e}")
        emd_losses = torch.zeros(batch_size * seq_length, device=logits.device)

    # Mask out padding positions
    emd_losses = emd_losses.view(batch_size, seq_length) * mask

    # Reduce loss
    if reduction == "sum":
        return emd_losses.sum()
    elif reduction == "mean":
        return emd_losses.sum() / mask.sum().clamp(min=1.0).detach()
    else:
        raise ValueError("reduction must be 'sum' or 'mean'")
