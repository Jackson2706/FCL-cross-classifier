"""Fast plain-script checks for safe W&B tracking."""

import builtins
import os
import sys
import tempfile
from argparse import Namespace


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "system"))

from flcore.tracking.config import normalize_wandb_config
from flcore.tracking.wandb_logger import (
    WandbTracker,
    filter_private_metrics,
    sanitize_resolved_config,
)
from flcore.tracking.metric_forwarding import (
    build_async_round_metrics, build_boundary_metrics, build_final_metrics,
)


def test_normalization():
    assert normalize_wandb_config(Namespace(wandb=False))["mode"] == "disabled"
    legacy = normalize_wandb_config(Namespace(wandb=True))
    assert legacy["enabled"] and legacy["mode"] == "online"
    structured = normalize_wandb_config(Namespace(wandb={"enabled": True, "mode": "offline"}))
    assert structured["mode"] == "offline"
    forced_off = normalize_wandb_config(Namespace(wandb={"enabled": False, "mode": "online"}))
    assert forced_off["mode"] == "disabled"


def test_disabled_is_pure_noop_without_wandb():
    real_import = builtins.__import__
    def reject_wandb(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("simulated missing dependency")
        return real_import(name, *args, **kwargs)
    builtins.__import__ = reject_wandb
    try:
        with tempfile.TemporaryDirectory() as folder:
            tracker = WandbTracker({"enabled": False, "mode": "disabled"}, {}, folder)
            tracker.log({"metric": 1.0}, step=1)
            tracker.log_summary({"metric": 1.0})
            tracker.finish()
            assert not os.path.exists(os.path.join(folder, "wandb_run.json"))
    finally:
        builtins.__import__ = real_import


def test_privacy_filters():
    assert filter_private_metrics({
        "loss": 1.0,
        "per_client_accuracy_list": [1.0],
        "unrecognized_raw_array": [1.0],
        "Client_Global/Client_0/Averaged Test Accurancy": 0.5,
        "eval/avg_accuracy": 0.75,
    }) == {"loss": 1.0, "eval/avg_accuracy": 0.75}
    sanitized = sanitize_resolved_config({"dataset": "CIFAR100", "out_folder": "/secret/path"})
    assert sanitized == {"dataset": "CIFAR100"}


def test_metric_forwarding_builders():
    summary = {
        "global": {
            "average_accuracy": 0.8, "forgetting": None,
            "backward_transfer": -0.1, "average_anytime_accuracy": 0.7,
        },
        "latest_task_accuracy": {"0": 0.9, "1": None},
        "communication": {"total_mb": 2048.0},
        "generator": {"total": 1.2, "cls": 0.3, "kd": 0.4, "bn": 0.5},
        "reliability": {
            "mean_weight": 0.6, "std_weight": 0.1, "min_weight": 0.2,
            "max_weight": 0.9, "accepted_ratio": 0.75,
        },
        "robustness": {
            "boundary": {
                "latest_diagnostic": {"real_test": {"true_class_margin": {"mean": 1.0, "std": 0.2}}},
                "latest_robust_accuracy": {"fgsm": {"average_accuracy": 0.4}},
            },
            "attacks": {"corrupted_client_fraction": 0.5, "detected_corrupted_clients": 1},
        },
        "heterogeneity": {
            "per_model_type": {"small": {"mean_test_accuracy": 0.6}},
            "fairness_gap": 0.2,
            "capacity_gap": {"smallest_mean_test_accuracy": 0.6, "largest_mean_test_accuracy": 0.8},
        },
    }
    flat = build_boundary_metrics(summary, task_id=1, glob_iter=4)
    assert flat["eval/avg_accuracy"] == 0.8
    assert flat["eval/task_accuracy/task_0"] == 0.9
    assert flat["generator/loss_kd"] == 0.4
    assert flat["reliability/accepted_ratio"] == 0.75
    assert flat["boundary/robust_acc_fgsm"] == 0.4
    assert flat["heterogeneity/accuracy_by_model_type/small"] == 0.6
    assert "eval/forgetting" not in flat
    assert "eval/task_accuracy/task_1" not in flat
    final = build_final_metrics(summary, runtime_seconds=12.0)
    assert final["communication/total_gb"] == 2.0
    assert final["runtime/total_seconds"] == 12.0
    state = Namespace(active=True, dropped=False)
    dropped = Namespace(active=False, dropped=True)
    async_flat = build_async_round_metrics(
        {0: state, 1: dropped}, {"temporal_lag_mean": 0.5, "temporal_lag_std": 0.5},
    )
    assert async_flat == {
        "async/active_client_count": 1,
        "async/dropout_count": 1,
        "async/partial_participation_rate": 0.5,
        "async/temporal_lag_mean": 0.5,
        "async/temporal_lag_std": 0.5,
    }


if __name__ == "__main__":
    test_normalization()
    test_disabled_is_pure_noop_without_wandb()
    test_privacy_filters()
    test_metric_forwarding_builders()
    print("wandb tracker tests passed")
