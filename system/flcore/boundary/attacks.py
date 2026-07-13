"""Small, reusable attacks for synthetic replay and real-data evaluation."""

import torch
import torch.nn.functional as F


BOUNDARY_CONFIG_DEFAULTS = {
    "boundary_mode": "none",
    "adv_epsilon": 0.03,
    "pgd_steps": 3,
    "pgd_alpha": None,
    "lambda_adv": 1.0,
    "density_tau": None,
    "boundary_robust_eval": True,
    "robust_eval_max_batches": None,
}

BOUNDARY_MODES = {
    "none", "fgsm", "pgd_light", "margin_regularization", "prototype_margin"
}


def _clamp(inputs, clamp_min, clamp_max):
    if clamp_min is not None:
        inputs = torch.maximum(inputs, torch.as_tensor(
            clamp_min, device=inputs.device, dtype=inputs.dtype
        ))
    if clamp_max is not None:
        inputs = torch.minimum(inputs, torch.as_tensor(
            clamp_max, device=inputs.device, dtype=inputs.dtype
        ))
    return inputs


def fgsm_attack(model, inputs, targets, epsilon=0.03, clamp_min=0.0, clamp_max=1.0):
    """Return ``x + epsilon * sign(grad_x CE(model(x), y))``, detached."""

    attack_inputs = inputs.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(model(attack_inputs), targets)
    gradient = torch.autograd.grad(loss, attack_inputs, only_inputs=True)[0]
    adversarial = attack_inputs + float(epsilon) * gradient.sign()
    return _clamp(adversarial, clamp_min, clamp_max).detach()


def pgd_light_attack(
    model, inputs, targets, epsilon=0.03, steps=3, alpha=None,
    clamp_min=0.0, clamp_max=1.0,
):
    """Deterministic projected-gradient attack in an L-infinity epsilon ball."""

    epsilon = float(epsilon)
    steps = int(steps)
    if steps < 1:
        raise ValueError("pgd_steps must be at least 1")
    alpha = epsilon / 2.0 if alpha is None else float(alpha)
    original = inputs.detach()
    adversarial = original.clone()
    for _ in range(steps):
        attack_inputs = adversarial.detach().requires_grad_(True)
        loss = F.cross_entropy(model(attack_inputs), targets)
        gradient = torch.autograd.grad(loss, attack_inputs, only_inputs=True)[0]
        adversarial = attack_inputs + alpha * gradient.sign()
        adversarial = torch.maximum(adversarial, original - epsilon)
        adversarial = torch.minimum(adversarial, original + epsilon)
        adversarial = _clamp(adversarial, clamp_min, clamp_max).detach()
    return adversarial


def density_gate(bn_distance, tau=None):
    """Gate samples below tau; null tau uses the batch median distance."""

    if bn_distance.ndim != 1:
        raise ValueError("bn_distance must contain one value per sample")
    if bn_distance.numel() == 0:
        return torch.zeros_like(bn_distance, dtype=torch.bool), None
    threshold = torch.median(bn_distance.detach()) if tau is None else bn_distance.new_tensor(float(tau))
    return bn_distance < threshold, float(threshold)


def density_gated_adversarial_loss(logits, targets, weights, gate, lambda_adv=1.0):
    """Paper adversarial term: lambda * mean(gate * w * CE_adv)."""

    per_sample = F.cross_entropy(logits, targets, reduction="none")
    return float(lambda_adv) * torch.mean(
        gate.to(per_sample.dtype) * weights.detach().to(per_sample.dtype) * per_sample
    )
