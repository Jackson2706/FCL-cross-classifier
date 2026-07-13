"""Architecture-agnostic classifier aggregation by knowledge distillation."""

from contextlib import contextmanager

import torch

from flcore.reliability.objectives import weighted_kl_distillation_loss


def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@contextmanager
def _preserve_training_mode(model):
    was_training = model.training
    try:
        yield
    finally:
        model.train(was_training)


def _normalized_teacher_weights(weights, teacher_count, batch_size, reference):
    if weights is None:
        resolved = reference.new_ones((teacher_count, batch_size))
    else:
        resolved = torch.as_tensor(weights, device=reference.device, dtype=reference.dtype)
        if resolved.ndim == 1:
            if resolved.numel() != teacher_count:
                raise ValueError("weights must contain one value per teacher")
            resolved = resolved[:, None].expand(-1, batch_size)
        elif resolved.shape != (teacher_count, batch_size):
            raise ValueError(
                "weights must have shape [teachers] or [teachers, transfer_samples]"
            )
    resolved = resolved.clamp_min(0.0)
    totals = resolved.sum(dim=0, keepdim=True)
    uniform = torch.full_like(resolved, 1.0 / teacher_count)
    normalized = torch.where(
        totals > 0.0, resolved / totals.clamp_min(1e-12), uniform
    )
    return normalized, totals.squeeze(0)


def server_distill(
    server_model,
    client_teachers,
    transfer_inputs,
    weights=None,
    T=2.0,
    optimizer=None,
):
    """Take one reliability-weighted KD step from clients into ``server_model``.

    ``weights`` may be one reliability/sample-mass value per teacher or a
    teacher-by-sample matrix.  Teacher probabilities, rather than parameters,
    are combined, so the teachers may have unrelated architectures.
    """

    teachers = list(client_teachers)
    if not teachers:
        raise ValueError("client_teachers must not be empty")
    temperature = float(T)
    if temperature <= 0.0:
        raise ValueError("T must be positive")

    device = _model_device(server_model)
    inputs = transfer_inputs.to(device)
    teacher_logits = []
    with torch.no_grad():
        for teacher in teachers:
            teacher_device = _model_device(teacher)
            with _preserve_training_mode(teacher):
                teacher.eval()
                teacher_logits.append(teacher(inputs.to(teacher_device)).to(device))
    ensemble = torch.stack(teacher_logits, dim=0)
    normalized, sample_weights = _normalized_teacher_weights(
        weights, len(teachers), inputs.shape[0], ensemble
    )
    target_probabilities = (
        normalized.unsqueeze(-1) * torch.softmax(ensemble / temperature, dim=-1)
    ).sum(dim=0)
    # Reconstruct logits whose temperature-softmax is the weighted target.
    target_logits = temperature * target_probabilities.clamp_min(1e-12).log()

    if optimizer is None:
        optimizer = torch.optim.SGD(server_model.parameters(), lr=0.01)
    with _preserve_training_mode(server_model):
        server_model.train()
        optimizer.zero_grad()
        student_logits = server_model(inputs)
        # Preserve per-sample reliability while keeping its mean scale stable.
        sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-12)
        loss = weighted_kl_distillation_loss(
            target_logits, student_logits, sample_weights, temperature
        ) * (temperature ** 2)
        loss.backward()
        optimizer.step()
    return float(loss.detach())


def client_distill(
    local_model,
    server_logits_or_model,
    transfer_inputs,
    T=2.0,
    steps=1,
    optimizer=None,
):
    """Distill fixed server soft targets into a client's own architecture."""

    steps = int(steps)
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if steps == 0:
        return 0.0
    temperature = float(T)
    if temperature <= 0.0:
        raise ValueError("T must be positive")

    device = _model_device(local_model)
    inputs = transfer_inputs.to(device)
    with torch.no_grad():
        if isinstance(server_logits_or_model, torch.nn.Module):
            server = server_logits_or_model
            with _preserve_training_mode(server):
                server.eval()
                targets = server(
                    transfer_inputs.to(_model_device(server))
                ).detach().to(device)
        else:
            targets = torch.as_tensor(server_logits_or_model).detach().to(device)
    if targets.shape[0] != inputs.shape[0]:
        raise ValueError("server targets and transfer_inputs must have the same batch size")

    if optimizer is None:
        optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01)
    last_loss = None
    with _preserve_training_mode(local_model):
        local_model.train()
        sample_weights = inputs.new_ones(inputs.shape[0])
        for _ in range(steps):
            optimizer.zero_grad()
            logits = local_model(inputs)
            loss = weighted_kl_distillation_loss(
                targets, logits, sample_weights, temperature
            ) * (temperature ** 2)
            loss.backward()
            optimizer.step()
            last_loss = loss.detach()
    return float(last_loss)
