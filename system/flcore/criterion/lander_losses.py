# system/flcore/criterion/lander_losses.py
import torch
import torch.nn.functional as F


def bounding_loss(features: torch.Tensor,
                  anchors: torch.Tensor,
                  labels: torch.Tensor,
                  radius: float = 0.5,
                  reduction: str = "mean"):
    """
    LANDER's 'Bounding Loss':
    Encourage ||f - a_y||_2 <= radius; penalize only if outside the ball.
    L = max(0, ||f - a_y|| - R)^2
    """
    a = anchors[labels]  # [N, D]
    d = torch.norm(features - a, dim=1)
    penalty = torch.clamp(d - radius, min=0.0) ** 2
    return penalty.mean() if reduction == "mean" else penalty.sum()