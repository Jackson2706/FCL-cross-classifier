"""Fast CPU-only checks for heterogeneous model distillation."""

import math
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "system"))

from flcore.hetero.aggregator import client_distill, server_distill
from flcore.hetero.metrics import (
    summarize_distillation_communication,
    summarize_model_type_accuracy,
)
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


def test_distillation_modes_accept_distinct_architectures():
    for mode in ("logit_distillation", "synthetic_distillation"):
        config = resolve_heterogeneity_config(SimpleNamespace(
            model_heterogeneity=True,
            client_model_pool=["small_cnn", "lightweight_resnet"],
            server_model="small_cnn",
            aggregation_mode=mode,
            client_distill_steps=2,
            distill_transfer_size=7,
        ))
        assert config.aggregation_mode == mode
        assert config.client_distill_steps == 2
        assert config.distill_transfer_size == 7


def _kl(teacher_logits, student_logits, temperature=2.0):
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    )


def test_server_distill_reduces_teacher_student_kl():
    torch.manual_seed(4)
    inputs = torch.randn(16, 5)
    teacher = torch.nn.Linear(5, 3)
    student = torch.nn.Linear(5, 3)
    with torch.no_grad():
        before = _kl(teacher(inputs), student(inputs)).item()
    optimizer = torch.optim.SGD(student.parameters(), lr=0.5)
    for _ in range(20):
        server_distill(student, [teacher], inputs, [1.0], 2.0, optimizer)
    with torch.no_grad():
        after = _kl(teacher(inputs), student(inputs)).item()
    assert after < before


def test_client_distill_moves_local_model_toward_server_targets():
    torch.manual_seed(5)
    inputs = torch.randn(16, 4)
    server = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3))
    local = torch.nn.Linear(4, 3)
    with torch.no_grad():
        server_logits = server(inputs)
        before = _kl(server_logits, local(inputs)).item()
    client_distill(
        local, server_logits, inputs, T=2.0, steps=20,
        optimizer=torch.optim.SGD(local.parameters(), lr=0.5),
    )
    with torch.no_grad():
        after = _kl(server_logits, local(inputs)).item()
    assert after < before


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
    capacity = summary["capacity_gap"]
    assert capacity["smallest_model_type"] == "small_cnn"
    assert capacity["largest_model_type"] == "resnet18"
    assert math.isclose(capacity["largest_minus_smallest_accuracy"], 0.3)


def test_communication_by_architecture_math():
    summary = summarize_distillation_communication({
        "small_cnn": {
            "uplink_bytes": 2 * 8 * 10 * 4,
            "downlink_bytes": 3 * 8 * 10 * 4,
            "uplink_events": 2,
            "downlink_events": 3,
        }
    })
    record = summary["small_cnn"]
    assert record["uplink_bytes"] == 640
    assert record["downlink_bytes"] == 960
    assert record["total_bytes"] == 1600
    assert record["payload"] == "transfer_logits_float32"


def test_distillation_round_does_not_call_parameter_averaging():
    from flcore.servers.server_ourv2 import OursV2

    class GuardServer:
        uploaded_models = []

        def _receive_distillation_models(self):
            self.uploaded_models = [object()]

        def _server_distillation_update(self, timer_key):
            self.timer_key = timer_key

        def send_models(self):
            self.sent = True

        def aggregate_parameters(self):
            raise AssertionError("parameter averaging must not be called")

    server = GuardServer()
    OursV2._distillation_round(server, 9)
    assert server.timer_key == 9
    assert server.sent


if __name__ == "__main__":
    test_factory_models_forward_and_split()
    test_heterogeneous_fedavg_is_refused()
    test_distillation_modes_accept_distinct_architectures()
    test_server_distill_reduces_teacher_student_kl()
    test_client_distill_moves_local_model_toward_server_targets()
    test_model_type_accuracy_and_fairness_gap()
    test_communication_by_architecture_math()
    test_distillation_round_does_not_call_parameter_averaging()
    print("heterogeneity tests passed")
