# system/flcore/criterion/distillation.py
import torch
import torch.nn.functional as F


def kl_divergence_with_temperature(student_logits, teacher_logits, T: float = 2.0, reduction="batchmean"):
    """KL(student || teacher) with temperature scaling."""
    s = F.log_softmax(student_logits / T, dim=1)
    t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(s, t, reduction=reduction) * (T * T)

def pairwise_cosine(x: torch.Tensor):
    """NxD -> NxN cosine similarity (row-wise)."""
    x = F.normalize(x, dim=1)
    return x @ x.t()

def relation_mse(student_logits, teacher_logits):
    """Relational KD (pairwise logit cosines), simple RKD-style loss."""
    with torch.no_grad():
        t_rel = pairwise_cosine(teacher_logits).detach()
    s_rel = pairwise_cosine(student_logits)
    return F.mse_loss(s_rel, t_rel)