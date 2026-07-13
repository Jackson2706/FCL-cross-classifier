"""Robust-accuracy evaluation helpers."""

import torch

from .attacks import fgsm_attack, pgd_light_attack


def evaluate_robust_accuracy(
    model, data_loader, device="cpu", attack="fgsm", epsilon=0.03,
    pgd_steps=3, pgd_alpha=None, clamp_min=0.0, clamp_max=1.0,
    max_batches=None,
):
    """Return ``(correct, total)`` under an FGSM or lightweight PGD attack."""

    correct = total = 0
    for batch_index, (inputs, targets) in enumerate(data_loader):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        if isinstance(inputs, list):
            inputs = inputs[0]
        inputs, targets = inputs.to(device), targets.to(device)
        if attack == "fgsm":
            adversarial = fgsm_attack(
                model, inputs, targets, epsilon, clamp_min, clamp_max
            )
        elif attack == "pgd_light":
            adversarial = pgd_light_attack(
                model, inputs, targets, epsilon, pgd_steps, pgd_alpha,
                clamp_min, clamp_max,
            )
        else:
            raise ValueError(f"Unknown robust-evaluation attack: {attack}")
        with torch.no_grad():
            predictions = model(adversarial).argmax(dim=1)
        correct += int(predictions.eq(targets).sum())
        total += int(targets.numel())
    return correct, total
