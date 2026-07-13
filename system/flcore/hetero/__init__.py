"""Model-heterogeneity helpers for the journal simulation framework."""

from .aggregator import client_distill, server_distill
from .metrics import (
    summarize_capacity_gap,
    summarize_distillation_communication,
    summarize_model_type_accuracy,
)
from .model_factory import (
    HETERO_CONFIG_DEFAULTS,
    build_client_model,
    resolve_heterogeneity_config,
)

__all__ = [
    "HETERO_CONFIG_DEFAULTS",
    "client_distill",
    "build_client_model",
    "resolve_heterogeneity_config",
    "server_distill",
    "summarize_capacity_gap",
    "summarize_distillation_communication",
    "summarize_model_type_accuracy",
]
