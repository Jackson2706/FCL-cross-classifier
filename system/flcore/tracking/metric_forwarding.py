"""Pure builders for the journal metric namespace forwarded to W&B."""

from collections.abc import Mapping
import math


def _put(result, key, value):
    if value is None or isinstance(value, (Mapping, list, tuple)):
        return
    try:
        if isinstance(value, float) and not math.isfinite(value):
            return
    except TypeError:
        return
    if isinstance(value, (bool, int, float)):
        result[key] = value


def build_boundary_metrics(summary, task_id=None, glob_iter=None):
    """Flatten one metrics-summary snapshot into task-boundary W&B scalars."""

    summary = summary if isinstance(summary, Mapping) else {}
    result = {}
    global_metrics = summary.get("global", {}) or {}
    for source, target in (
        ("average_accuracy", "avg_accuracy"),
        ("forgetting", "forgetting"),
        ("backward_transfer", "bwt"),
        ("average_anytime_accuracy", "aaa"),
    ):
        _put(result, f"eval/{target}", global_metrics.get(source))
    _put(result, "eval/round", glob_iter)
    _put(result, "eval/task_id", task_id)
    latest_row = summary.get("latest_task_accuracy", {}) or {}
    for key, value in latest_row.items():
        _put(result, f"eval/task_accuracy/task_{key}", value)

    generator = summary.get("generator", {}) or {}
    for source, target in (("total", "loss_total"), ("cls", "loss_cls"),
                           ("kd", "loss_kd"), ("bn", "loss_bn")):
        _put(result, f"generator/{target}", generator.get(source))
    classifier = summary.get("classifier", {}) or {}
    _put(result, "classifier/loss_replay", classifier.get("loss_replay"))
    _put(result, "classifier/loss_adv", classifier.get("loss_adv"))

    reliability = summary.get("reliability", {}) or {}
    for source, target in (("mean_weight", "mean"), ("std_weight", "std"),
                           ("min_weight", "min"), ("max_weight", "max"),
                           ("accepted_ratio", "accepted_ratio")):
        _put(result, f"reliability/{target}", reliability.get(source))

    boundary = (summary.get("robustness", {}) or {}).get("boundary", {}) or {}
    diagnostic = boundary.get("latest_diagnostic", {}) or {}
    margin = diagnostic.get("real_test", {}).get("true_class_margin", {}) or {}
    _put(result, "boundary/margin_mean", margin.get("mean"))
    _put(result, "boundary/margin_std", margin.get("std"))
    robust = boundary.get("latest_robust_accuracy", {}) or {}
    _put(result, "boundary/robust_acc_fgsm", (robust.get("fgsm", {}) or {}).get("average_accuracy"))
    _put(result, "boundary/robust_acc_pgd_light", (robust.get("pgd_light", {}) or {}).get("average_accuracy"))

    runtime = summary.get("runtime", {}) or {}
    _put(result, "runtime/task_seconds", runtime.get("task_seconds"))
    _put(result, "runtime/server_cotrain_seconds", runtime.get("server_cotrain_seconds"))

    attacks = (summary.get("robustness", {}) or {}).get("attacks", {}) or {}
    for source in ("corrupted_client_fraction", "detected_corrupted_clients",
                   "filter_drop_rate", "label_noise_rate", "bn_noise_std"):
        _put(result, f"robustness/{source}", attacks.get(source))

    hetero = summary.get("heterogeneity", {}) or {}
    for model_type, values in (hetero.get("per_model_type", {}) or {}).items():
        _put(result, f"heterogeneity/accuracy_by_model_type/{model_type}",
             (values or {}).get("mean_test_accuracy"))
    _put(result, "heterogeneity/fairness_gap", hetero.get("fairness_gap"))
    capacity = hetero.get("capacity_gap", {}) or {}
    _put(result, "heterogeneity/small_client_accuracy", capacity.get("smallest_mean_test_accuracy"))
    _put(result, "heterogeneity/large_client_accuracy", capacity.get("largest_mean_test_accuracy"))
    _put(result, "heterogeneity/distillation_loss", hetero.get("distillation_loss"))
    return result


def build_final_metrics(summary, runtime_seconds=None):
    """Build end-of-run summary keys, reusing the boundary namespace."""

    result = build_boundary_metrics(summary)
    result = {key: value for key, value in result.items() if key not in {"eval/round", "eval/task_id"}}
    communication = (summary or {}).get("communication", {}) or {}
    total_mb = communication.get("total_mb")
    if total_mb is not None:
        _put(result, "communication/total_gb", float(total_mb) / 1024.0)
    _put(result, "runtime/total_seconds", runtime_seconds)
    return result


def build_async_round_metrics(states, scheduler_metrics):
    """Build privacy-safe aggregate async scalars for one materialized round."""

    values = list(states.values()) if isinstance(states, Mapping) else list(states or [])
    active = sum(bool(state.active and not state.dropped) for state in values)
    result = {
        "async/active_client_count": active,
        "async/dropout_count": sum(bool(state.dropped) for state in values),
        "async/partial_participation_rate": active / max(1, len(values)),
    }
    scheduler_metrics = scheduler_metrics or {}
    _put(result, "async/temporal_lag_mean", scheduler_metrics.get("temporal_lag_mean"))
    _put(result, "async/temporal_lag_std", scheduler_metrics.get("temporal_lag_std"))
    return result
