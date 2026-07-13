"""Boundary-preserving adversarial replay and robust evaluation."""

from .attacks import (
    BOUNDARY_CONFIG_DEFAULTS,
    density_gated_adversarial_loss,
    density_gate,
    fgsm_attack,
    pgd_light_attack,
)
from .evaluation import evaluate_robust_accuracy
from .diagnostics import (
    class_center_distance_matrix,
    classification_margins,
    margin_regularization_loss,
    prototype_drift,
    prototype_margin_loss,
)

__all__ = [
    "BOUNDARY_CONFIG_DEFAULTS",
    "density_gated_adversarial_loss",
    "density_gate",
    "evaluate_robust_accuracy",
    "fgsm_attack",
    "pgd_light_attack",
    "class_center_distance_matrix",
    "classification_margins",
    "margin_regularization_loss",
    "prototype_drift",
    "prototype_margin_loss",
]
