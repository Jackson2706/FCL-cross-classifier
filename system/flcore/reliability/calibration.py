"""Calibration metrics for classifier probabilities."""

import torch


def expected_calibration_error(logits, targets, bins=15, temperature=1.0):
    """Compute standard equal-width binned expected calibration error."""

    bins = int(bins)
    temperature = float(temperature)
    if bins <= 0:
        raise ValueError("bins must be positive")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if logits.ndim != 2:
        raise ValueError("logits must have shape [N,C]")
    targets = torch.as_tensor(targets, device=logits.device).reshape(-1)
    if targets.numel() != logits.shape[0]:
        raise ValueError("targets must match the logits batch length")
    if targets.numel() == 0:
        return logits.new_tensor(0.0)
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")

    probabilities = torch.softmax(logits / temperature, dim=-1)
    confidences, predictions = probabilities.max(dim=-1)
    correctness = predictions.eq(targets).to(confidences.dtype)
    edges = torch.linspace(0.0, 1.0, bins + 1, device=logits.device)
    ece = logits.new_tensor(0.0)
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        if index == 0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences > lower) & (confidences <= upper)
        if mask.any():
            fraction = mask.to(confidences.dtype).mean()
            ece = ece + fraction * (
                correctness[mask].mean() - confidences[mask].mean()
            ).abs()
    return ece
