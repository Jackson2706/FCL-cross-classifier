"""Framework-light uncertainty-to-weight transformations.

``oracle_debug`` intentionally uses labels and must never be used for reported
results.  It exists only as a wiring sanity check.
"""

import torch


RELIABILITY_CONFIG_DEFAULTS = {
    "reliability_mode": "none",
    "reliability_beta": 1.0,
    "reliability_beta_entropy": 1.0,
    "reliability_beta_bn": 1.0,
    "reliability_trust_floor": 0.0,
    "reliability_accept_threshold": 0.0,
}


class ReliabilityScorer:
    """Convert predictive/realism/trust signals into per-sample weights."""

    MODES = {
        "none", "entropy", "mutual_information", "bn_realism",
        "trust_only", "multi_signal", "oracle_debug",
    }

    def __init__(self, mode="none", **cfg):
        self.mode = str(mode).lower()
        if self.mode not in self.MODES:
            raise ValueError(
                f"Unknown reliability mode {mode!r}; expected one of {sorted(self.MODES)}"
            )
        self.beta = float(cfg.get("reliability_beta", 1.0))
        self.beta_entropy = float(cfg.get("reliability_beta_entropy", 1.0))
        self.beta_bn = float(cfg.get("reliability_beta_bn", 1.0))
        self.trust_floor = float(cfg.get("reliability_trust_floor", 0.0))
        self.oracle_error_weight = float(cfg.get("reliability_oracle_error_weight", 0.05))
        self.last_missing_signals = []

    @staticmethod
    def _stack_logits(logits_or_ensemble):
        if isinstance(logits_or_ensemble, (list, tuple)):
            if not logits_or_ensemble:
                raise ValueError("logits ensemble must not be empty")
            return torch.stack(list(logits_or_ensemble), dim=0)
        if not torch.is_tensor(logits_or_ensemble):
            raise TypeError("logits_or_ensemble must be a tensor or sequence of tensors")
        if logits_or_ensemble.ndim not in (2, 3):
            raise ValueError("logits must have shape [B,C] or ensemble shape [K,B,C]")
        return logits_or_ensemble

    @staticmethod
    def _predictive_distribution(logits):
        if logits.ndim == 2:
            return torch.log_softmax(logits, dim=-1).exp()
        return torch.log_softmax(logits, dim=-1).exp().mean(dim=0)

    @staticmethod
    def _entropy(probabilities):
        tiny = torch.finfo(probabilities.dtype).tiny
        probabilities = probabilities.clamp_min(tiny)
        return -(probabilities * probabilities.log()).sum(dim=-1)

    @staticmethod
    def _batch_size_and_reference(logits_or_ensemble, bn_distance, trust, targets):
        if logits_or_ensemble is not None:
            logits = ReliabilityScorer._stack_logits(logits_or_ensemble)
            return logits.shape[-2], logits
        for value in (bn_distance, trust, targets):
            if torch.is_tensor(value) and value.ndim > 0:
                return value.numel(), value
        raise ValueError("at least one batched reliability signal is required")

    @staticmethod
    def _as_batch(value, batch_size, reference, name):
        tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
        if tensor.ndim == 0:
            tensor = tensor.expand(batch_size)
        else:
            tensor = tensor.reshape(-1)
        if tensor.numel() != batch_size:
            raise ValueError(f"{name} must be scalar or have batch length {batch_size}")
        return tensor

    def score(self, logits_or_ensemble, targets=None, bn_distance=None, trust=None):
        """Return a detached weight in ``[0,1]`` for each sample."""

        self.last_missing_signals = []
        batch_size, reference = self._batch_size_and_reference(
            logits_or_ensemble, bn_distance, trust, targets
        )
        dtype = reference.dtype if reference.is_floating_point() else torch.float32
        reference = torch.empty((), device=reference.device, dtype=dtype)

        if self.mode == "none":
            return torch.ones(batch_size, device=reference.device, dtype=reference.dtype)

        logits = None
        if logits_or_ensemble is not None:
            logits = self._stack_logits(logits_or_ensemble).to(dtype=reference.dtype)

        if self.mode in {"entropy", "multi_signal"}:
            if logits is None:
                entropy_factor = torch.ones(batch_size, device=reference.device)
                self.last_missing_signals.append("entropy")
            else:
                entropy = self._entropy(self._predictive_distribution(logits))
                beta = self.beta if self.mode == "entropy" else self.beta_entropy
                entropy_factor = torch.exp(-beta * entropy)
            if self.mode == "entropy":
                return entropy_factor.clamp(0.0, 1.0).detach()

        if self.mode == "mutual_information":
            if logits is None or logits.ndim != 3:
                raise ValueError("mutual_information requires teacher logits shaped [K,B,C]")
            teacher_probabilities = torch.log_softmax(logits, dim=-1).exp()
            predictive_entropy = self._entropy(teacher_probabilities.mean(dim=0))
            expected_entropy = self._entropy(teacher_probabilities).mean(dim=0)
            uncertainty = (predictive_entropy - expected_entropy).clamp_min(0.0)
            return torch.exp(-self.beta * uncertainty).clamp(0.0, 1.0).detach()

        if self.mode in {"bn_realism", "multi_signal"}:
            if bn_distance is None:
                bn_factor = torch.ones(batch_size, device=reference.device)
                self.last_missing_signals.append("bn_distance")
            else:
                distance = self._as_batch(
                    bn_distance, batch_size, reference, "bn_distance"
                )
                distance = torch.nan_to_num(
                    distance, nan=float("inf"), posinf=float("inf"), neginf=0.0
                ).clamp_min(0.0)
                bn_factor = torch.exp(-self.beta_bn * distance)
            if self.mode == "bn_realism":
                return bn_factor.clamp(0.0, 1.0).detach()

        if self.mode in {"trust_only", "multi_signal"}:
            if trust is None:
                trust_factor = torch.ones(batch_size, device=reference.device)
                self.last_missing_signals.append("trust")
            else:
                trust_factor = self._as_batch(trust, batch_size, reference, "trust")
                trust_factor = torch.nan_to_num(
                    trust_factor, nan=self.trust_floor,
                    posinf=1.0, neginf=self.trust_floor,
                ).clamp(self.trust_floor, 1.0)
            if self.mode == "trust_only":
                return trust_factor.detach()

        if self.mode == "multi_signal":
            return (entropy_factor * bn_factor * trust_factor).clamp(0.0, 1.0).detach()

        if targets is None or logits is None:
            raise ValueError("oracle_debug requires logits and targets")
        predictions = self._predictive_distribution(logits).argmax(dim=-1)
        targets = torch.as_tensor(targets, device=predictions.device).reshape(-1)
        if targets.numel() != batch_size:
            raise ValueError(f"targets must have batch length {batch_size}")
        correct = predictions.eq(targets)
        weights = torch.full(
            (batch_size,), self.oracle_error_weight,
            device=reference.device, dtype=reference.dtype,
        )
        return torch.where(correct, torch.ones_like(weights), weights).clamp(0.0, 1.0)
