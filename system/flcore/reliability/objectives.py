"""Loss primitives for reliability-weighted generator consolidation."""

import torch
import torch.nn.functional as F


def weighted_classification_loss(logits, targets, weights):
    """Return ``mean(w * CE(logits, targets))``."""

    per_sample = F.cross_entropy(logits, targets, reduction="none")
    return torch.mean(weights.to(per_sample) * per_sample)


def weighted_kl_distillation_loss(teacher_logits, student_logits, weights, temperature):
    """Return ``mean(w * KL(teacher/T || student/T))``."""

    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    # The client prediction is the fixed KD target. Gradients still flow from
    # the student branch into the synthetic image and generator.
    teacher_probabilities = F.softmax(
        teacher_logits / temperature, dim=-1
    ).detach()
    student_log_probabilities = F.log_softmax(student_logits / temperature, dim=-1)
    per_sample = F.kl_div(
        student_log_probabilities,
        teacher_probabilities,
        reduction="none",
    ).sum(dim=-1)
    return torch.mean(weights.to(per_sample) * per_sample)
