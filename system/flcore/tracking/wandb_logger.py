"""Failure-tolerant, privacy-aware Weights & Biases integration."""

import os
import re
from collections.abc import Mapping

from flcore.utils.json_io import atomic_write_json


_CONFIG_ALLOWLIST = {
    "algorithm", "dataset", "num_classes", "model", "model_str", "seed",
    "optimizer", "batch_size", "local_learning_rate", "learning_rate_decay",
    "learning_rate_decay_gamma", "join_ratio", "random_join_ratio",
    "client_drop_rate", "global_rounds", "local_epochs", "eval_gap",
    "num_tasks", "cpt", "g_lr", "g_steps", "replay_weight", "nz",
    "latent_dim", "num_clients", "generated_samples_per_class", "gen_type",
    "generator_distillation", "kd", "kd_weight", "generator_grad_clip",
    "oh", "bn", "adv", "async_mode", "reliability_mode", "boundary_mode",
    "robustness_mode", "model_heterogeneity", "aggregation_mode",
}
_SECRET_PARTS = {"api_key", "apikey", "secret", "password", "token", "credential"}
_PATH_PARTS = {"path", "folder", "directory", "dir", "root", "device_id", "host", "user"}
_PRIVATE_LOG_PARTS = {
    "image", "sample", "logit", "gradient", "schedule", "client_id",
    "client_ids", "corrupted_client", "per_client", "accuracy_list",
}


def sanitize_resolved_config(resolved_config):
    """Return a scalar/list-only curated subset safe for run metadata."""
    if not isinstance(resolved_config, Mapping):
        try:
            resolved_config = vars(resolved_config)
        except TypeError:
            return {}
    clean = {}
    for key, value in resolved_config.items():
        lowered = str(key).lower()
        if key not in _CONFIG_ALLOWLIST:
            continue
        if any(part in lowered for part in _SECRET_PARTS | _PATH_PARTS):
            continue
        if value is None or isinstance(value, (str, bool, int, float)):
            clean[key] = value
        elif isinstance(value, (list, tuple)) and all(
            item is None or isinstance(item, (str, bool, int, float)) for item in value
        ):
            clean[key] = list(value)
    return clean


def filter_private_metrics(data):
    """Drop metrics whose keys indicate raw client data or high-risk payloads."""
    return {
        key: value for key, value in data.items()
        if not any(part in str(key).lower() for part in _PRIVATE_LOG_PARTS)
        and re.search(r"(?:^|[/_])client_?\d+(?:$|[/_])", str(key).lower()) is None
        and (value is None or isinstance(value, (str, bool, int, float)))
    }


class WandbTracker:
    """Own all optional W&B interaction and degrade safely on failure."""

    def __init__(self, cfg_dict, resolved_config, save_folder):
        self.cfg = dict(cfg_dict)
        self.mode = self.cfg.get("mode", "disabled")
        self.save_folder = os.fspath(save_folder)
        self._wandb = None
        self._run = None
        self._warning_printed = False
        if self.mode == "disabled":
            return

        try:
            import wandb
            self._wandb = wandb
        except Exception as error:
            self._disable(f"import failed: {error}")
            return

        init_args = {
            key: self.cfg.get(key)
            for key in ("project", "entity", "name", "group", "job_type", "tags", "notes", "resume")
        }
        init_args["config"] = sanitize_resolved_config(resolved_config)
        try:
            self._run = self._wandb.init(mode=self.mode, **init_args)
        except Exception as online_error:
            if self.mode == "online":
                try:
                    self.mode = "offline"
                    self._run = self._wandb.init(mode="offline", **init_args)
                    self._warn(f"online init failed; using offline mode: {online_error}")
                except Exception as offline_error:
                    self._disable(f"init failed: {offline_error}")
            else:
                self._disable(f"init failed: {online_error}")

        if self._run is not None:
            self._write_run_metadata()

    @property
    def enabled(self):
        return self._run is not None and self.mode != "disabled"

    def _warn(self, message):
        if not self._warning_printed:
            print(f"[W&B] Warning: {message}")
            self._warning_printed = True

    def _disable(self, message):
        self._warn(message)
        self.mode = "disabled"
        self._run = None

    def _write_run_metadata(self):
        try:
            atomic_write_json(os.path.join(self.save_folder, "wandb_run.json"), {
                "run_id": getattr(self._run, "id", None),
                "mode": self.mode,
                "project": self.cfg.get("project"),
                "entity": getattr(self._run, "entity", self.cfg.get("entity")),
                "name": getattr(self._run, "name", self.cfg.get("name")),
                "url": getattr(self._run, "url", None),
            }, "wandb_run.json")
        except Exception as error:
            self._warn(f"could not write run metadata: {error}")

    def log(self, data, step=None):
        if not self.enabled:
            return
        payload = filter_private_metrics(data) if self.cfg.get("privacy_safe_mode", True) else data
        if not payload:
            return
        try:
            self._wandb.log(payload, step=step)
        except Exception as error:
            self._disable(f"logging failed; tracking disabled: {error}")

    def log_summary(self, data):
        if not self.enabled:
            return
        payload = filter_private_metrics(data) if self.cfg.get("privacy_safe_mode", True) else data
        try:
            self._run.summary.update(payload)
        except Exception as error:
            self._disable(f"summary failed; tracking disabled: {error}")

    def finish(self):
        if self._run is None or self._wandb is None:
            return
        try:
            self._wandb.finish()
        except Exception as error:
            self._warn(f"finish failed: {error}")
        finally:
            self._run = None
