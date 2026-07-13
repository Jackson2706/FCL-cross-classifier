"""Read-only heuristic privacy-risk proxy measurements.

None of these metrics establishes differential privacy or any formal privacy
property.  They are bounded empirical warning signals only.
"""

import torch


def nearest_neighbor_distances(generated, real, feature_fn=None, batch_size=64):
    """Return each generated sample's nearest real-sample L2 distance."""

    if generated.numel() == 0 or real.numel() == 0:
        return torch.empty(0)
    with torch.no_grad():
        generated = generated.detach()
        real = real.detach()
        if feature_fn is not None:
            generated = feature_fn(generated)
            real = feature_fn(real)
        generated = generated.reshape(generated.shape[0], -1).float().cpu()
        real = real.reshape(real.shape[0], -1).float().cpu()
        chunks = []
        for start in range(0, generated.shape[0], max(1, int(batch_size))):
            chunks.append(torch.cdist(generated[start:start + batch_size], real).min(dim=1).values)
        return torch.cat(chunks)


def memorization_summary(distances, tau=None):
    """Summarize NN risk; score is fraction with distance <= configured tau.

    If tau is null, the fraction is deliberately left null rather than selecting
    a data-dependent threshold that would be difficult to compare across runs.
    """

    values = distances.detach().float().cpu()
    if not values.numel():
        return {"count": 0, "min": None, "mean": None, "median": None,
                "p05": None, "p95": None, "memorization_tau": tau,
                "generator_memorization_score": None}
    score = None if tau is None else float((values <= float(tau)).float().mean())
    return {
        "count": int(values.numel()),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p05": float(torch.quantile(values, 0.05)),
        "p95": float(torch.quantile(values, 0.95)),
        "memorization_tau": None if tau is None else float(tau),
        "generator_memorization_score": score,
    }


def membership_inference_proxy(member_logits, member_targets, heldout_logits, heldout_targets):
    """Heuristic train-vs-held-out confidence/loss gaps (positive implies risk)."""

    def statistics(logits, targets):
        probabilities = logits.detach().float().softmax(dim=-1)
        targets = targets.detach().long().view(-1, 1)
        confidence = probabilities.gather(1, targets).squeeze(1).clamp_min(1e-12)
        return confidence.mean(), (-confidence.log()).mean()

    member_confidence, member_loss = statistics(member_logits, member_targets)
    heldout_confidence, heldout_loss = statistics(heldout_logits, heldout_targets)
    return {
        "definition": "member minus held-out true-class confidence; also held-out minus member CE loss",
        "member_confidence": float(member_confidence),
        "heldout_confidence": float(heldout_confidence),
        "confidence_gap": float(member_confidence - heldout_confidence),
        "member_loss": float(member_loss),
        "heldout_loss": float(heldout_loss),
        "loss_gap": float(heldout_loss - member_loss),
        "heuristic_only": True,
    }
