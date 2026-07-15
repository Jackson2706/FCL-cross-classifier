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
    }) == {"loss": 1.0}
    sanitized = sanitize_resolved_config({"dataset": "CIFAR100", "out_folder": "/secret/path"})
    assert sanitized == {"dataset": "CIFAR100"}


if __name__ == "__main__":
    test_normalization()
    test_disabled_is_pure_noop_without_wandb()
    test_privacy_filters()
    print("wandb tracker tests passed")
