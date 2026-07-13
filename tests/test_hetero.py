"""Fast CPU-only checks for Phase-6a model heterogeneity scaffolding."""

import math
import os
import sys
from types import SimpleNamespace

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "system"))

from flcore.hetero.metrics import summarize_model_type_accuracy
from flcore.hetero.model_factory import (
    build_client_model,
    resolve_heterogeneity_config,
)


def test_factory_models_forward_and_split():
    args = SimpleNamespace(num_classes=10, device="cpu")
    inputs = torch.randn(2, 3, 32, 32)
    for name in ("small_cnn", "resnet18", "mobilenetv2", "lightweight_resnet"):
        model = build_client_model(name, args)
        model.eval()
        with torch.no_grad():
            logits = model(inputs)
            features, proto_logits = model.get_proto(inputs)
        assert hasattr(model, "base") and hasattr(model, "head")
        assert logits.shape == (2, 10)
        assert proto_logits.shape == (2, 10)
        assert features.shape[0] == 2


def test_heterogeneous_fedavg_is_refused():
    args = SimpleNamespace(
        model_heterogeneity=True,
        client_model_pool=["resnet18", "mobilenetv2"],
        server_model="resnet18",
        aggregation_mode="fedavg",
    )
    try:
        resolve_heterogeneity_config(args)
    except ValueError as exc:
        message = str(exc)
        assert "aggregation_mode=logit_distillation" in message
        assert "synthetic_distillation" in message
        assert "Phase 6b" in message
    else:
        raise AssertionError("heterogeneous FedAvg pool should have been refused")


def test_model_type_accuracy_and_fairness_gap():
    results = [
        {"model_type": "resnet18", "correct": 8, "samples": 10},
        {"model_type": "resnet18", "correct": 6, "samples": 10},
        {"model_type": "small_cnn", "correct": 4, "samples": 10},
    ]
    summary = summarize_model_type_accuracy(
        results, {"resnet18": 1000, "small_cnn": 400}
    )
    assert math.isclose(
        summary["per_model_type"]["resnet18"]["mean_test_accuracy"], 0.7
    )
    assert math.isclose(
        summary["per_model_type"]["small_cnn"]["mean_test_accuracy"], 0.4
    )
    assert math.isclose(summary["fairness_gap"], 0.3)
    assert summary["per_model_type"]["resnet18"]["parameter_bytes"] == 1000


if __name__ == "__main__":
    test_factory_models_forward_and_split()
    test_heterogeneous_fedavg_is_refused()
    test_model_type_accuracy_and_fairness_gap()
    print("heterogeneity tests passed")
