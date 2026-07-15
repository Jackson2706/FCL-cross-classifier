"""Normalization for legacy and structured Weights & Biases configuration."""

from collections.abc import Mapping


WANDB_DEFAULTS = {
    "enabled": False,
    "mode": "online",
    "project": "FEDRUA-Journal",
    "entity": None,
    "group": None,
    "job_type": "train",
    "name": None,
    "tags": [],
    "notes": None,
    "log_model": False,
    "watch_model": False,
    "log_gradients": False,
    "log_freq": 1,
    "log_generated_samples": False,
    "max_logged_images": 32,
    "privacy_safe_mode": True,
    "resume": "allow",
}


def normalize_wandb_config(args):
    """Return a complete structured config from ``args.wandb``.

    Booleans retain the legacy CLI/JSON meaning. A disabled configuration always
    has the effective mode ``disabled``, irrespective of its requested mode.
    """
    raw = getattr(args, "wandb", False)
    config = dict(WANDB_DEFAULTS)
    config["tags"] = []

    if isinstance(raw, Mapping):
        config.update(raw)
    elif isinstance(raw, bool):
        config["enabled"] = raw
        config["mode"] = "online" if raw else "disabled"
    else:
        config["enabled"] = False
        config["mode"] = "disabled"

    config["enabled"] = bool(config.get("enabled", False))
    requested_mode = str(config.get("mode", "online")).lower()
    if requested_mode not in {"online", "offline", "disabled"}:
        requested_mode = "online"
    config["mode"] = requested_mode if config["enabled"] else "disabled"
    config["tags"] = list(config.get("tags") or [])
    return config
