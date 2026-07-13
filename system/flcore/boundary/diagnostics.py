"""Read-only boundary diagnostics and margin-based replay objectives."""

from contextlib import contextmanager
import random

import numpy as np
import torch
import torch.nn.functional as F


def classification_margins(logits, targets):
    """Return true-class logit minus the largest other-class logit."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    if logits.shape[1] < 2:
        raise ValueError("margin diagnostics require at least two classes")
    targets = targets.to(device=logits.device, dtype=torch.long)
    true_logits = logits.gather(1, targets.view(-1, 1)).squeeze(1)
    other_logits = logits.clone()
    other_logits.scatter_(1, targets.view(-1, 1), -torch.inf)
    return true_logits - other_logits.max(dim=1).values


def top_two_margins(logits):
    """Return the prediction-confidence margin (largest minus second largest)."""

    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("logits must have shape [batch, at least two classes]")
    top_two = logits.topk(k=2, dim=1).values
    return top_two[:, 0] - top_two[:, 1]


def summarize_margins(margins, threshold=0.0, band=0.1, bins=20, value_range=(-5.0, 5.0)):
    """Create a compact JSON-safe distribution and boundary-ratio summary."""

    values = torch.as_tensor(margins, dtype=torch.float64).detach().cpu().reshape(-1)
    if values.numel() == 0:
        return {
            "count": 0, "mean": None, "std": None,
            "boundary_sample_ratio": None, "near_boundary_ratio": None,
            "threshold": float(threshold), "absolute_margin_band": float(band),
            "histogram": {"edges": [], "counts": [], "underflow": 0, "overflow": 0},
        }
    low, high = map(float, value_range)
    if not low < high:
        raise ValueError("histogram range must be increasing")
    edges = torch.linspace(low, high, int(bins) + 1, dtype=torch.float64)
    in_range = values[(values >= low) & (values <= high)]
    counts = torch.histc(in_range.float(), bins=int(bins), min=low, max=high).to(torch.long)
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "boundary_sample_ratio": float((values <= float(threshold)).double().mean()),
        "near_boundary_ratio": float((values.abs() <= float(band)).double().mean()),
        "threshold": float(threshold),
        "absolute_margin_band": float(band),
        "histogram": {
            "edges": [float(value) for value in edges],
            "counts": [int(value) for value in counts],
            "underflow": int((values < low).sum()),
            "overflow": int((values > high).sum()),
        },
    }


def class_prototypes(features, targets, class_labels=None):
    """Compute detached per-class feature centroids."""

    features = features.detach()
    targets = targets.detach().to(features.device)
    labels = torch.unique(targets).tolist() if class_labels is None else class_labels
    return {
        int(label): features[targets == int(label)].mean(dim=0).detach().cpu()
        for label in labels if bool((targets == int(label)).any())
    }


def prototype_drift(previous, current):
    """Return L2 drift for classes present in two prototype snapshots."""

    shared = sorted(set(previous).intersection(current))
    return {
        int(label): float(torch.linalg.vector_norm(
            current[label].detach().cpu() - previous[label].detach().cpu()
        ))
        for label in shared
    }


def class_center_distance_matrix(prototypes, class_labels=None):
    """Return ordered labels and the symmetric pairwise-L2 center matrix."""

    labels = sorted(prototypes) if class_labels is None else [
        int(label) for label in class_labels if int(label) in prototypes
    ]
    if not labels:
        return labels, torch.empty((0, 0), dtype=torch.float32)
    centers = torch.stack([prototypes[label].detach().cpu() for label in labels])
    return labels, torch.cdist(centers, centers, p=2)


def model_outputs_and_features(model, inputs):
    """Obtain BaseHeadSplit penultimate features and classifier logits.

    FEDRUA wraps its classifier in ``BaseHeadSplit``. Its ``get_proto`` method
    evaluates ``base(inputs)`` for the penultimate representation and applies
    ``head`` to that representation for logits.
    """

    if hasattr(model, "get_proto"):
        return model.get_proto(inputs)
    if hasattr(model, "base") and hasattr(model, "head"):
        features = model.base(inputs)
        return features, model.head(features)
    raise TypeError("prototype_feature_layer=penultimate requires BaseHeadSplit/get_proto")


def margin_regularization_loss(logits, targets, weights=None, margin_target=1.0, margin_lambda=0.1):
    """Weighted hinge: lambda * mean(w * max(0, target - true margin))."""

    margins = classification_margins(logits, targets)
    if weights is None:
        weights = torch.ones_like(margins)
    penalties = F.relu(float(margin_target) - margins)
    return float(margin_lambda) * torch.mean(weights.detach().to(penalties) * penalties)


def prototype_margin_loss(
    features, targets, prototypes, weights=None, margin_target=1.0, margin_lambda=0.1,
):
    """Weighted prototype triplet hinge over samples with known class centers.

    The per-sample term is ``max(0, d(f,p_y) - min_(c!=y)d(f,p_c) + target)``.
    """

    if float(margin_lambda) == 0.0 or not prototypes:
        return features.sum() * 0.0
    labels = sorted(int(label) for label in prototypes)
    if len(labels) < 2:
        return features.sum() * 0.0
    centers = torch.stack([
        torch.as_tensor(prototypes[label], device=features.device, dtype=features.dtype)
        for label in labels
    ])
    distances = torch.cdist(features.reshape(features.shape[0], -1), centers.reshape(len(labels), -1))
    label_to_column = {label: column for column, label in enumerate(labels)}
    target_values = targets.detach().cpu().tolist()
    valid_indices = [index for index, label in enumerate(target_values) if label in label_to_column]
    if not valid_indices:
        return features.sum() * 0.0
    rows = torch.as_tensor(valid_indices, device=features.device, dtype=torch.long)
    own_columns = torch.as_tensor(
        [label_to_column[int(target_values[index])] for index in valid_indices],
        device=features.device, dtype=torch.long,
    )
    selected = distances[rows]
    own = selected.gather(1, own_columns.view(-1, 1)).squeeze(1)
    other = selected.clone()
    other.scatter_(1, own_columns.view(-1, 1), torch.inf)
    penalties = F.relu(own - other.min(dim=1).values + float(margin_target))
    if weights is None:
        selected_weights = torch.ones_like(penalties)
    else:
        selected_weights = weights.detach().to(penalties)[rows]
    return float(margin_lambda) * torch.mean(selected_weights * penalties)


@contextmanager
def preserve_diagnostic_state(*models):
    """Restore RNG streams and every module train/eval flag after observation."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    modes = [{module: module.training for module in model.modules()} for model in models]
    try:
        yield
    finally:
        for model_modes in modes:
            for module, training in model_modes.items():
                module.training = training
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
