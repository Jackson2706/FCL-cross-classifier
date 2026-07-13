"""Model-type accuracy and fairness summaries."""

from collections import defaultdict


def summarize_model_type_accuracy(results, parameter_bytes=None):
    """Summarize client accuracies by architecture.

    Each result is a mapping with ``model_type``, ``correct``, and ``samples``.
    Correct/sample counts may already be accumulated across multiple tasks.  A
    model-type mean is the arithmetic mean of its clients' accuracies, ensuring
    one large client does not dominate the fairness comparison.
    """

    grouped = defaultdict(list)
    for result in results:
        samples = int(result.get("samples", 0))
        if samples <= 0:
            continue
        grouped[str(result["model_type"])].append(
            float(result.get("correct", 0.0)) / samples
        )

    per_model_type = {}
    parameter_bytes = parameter_bytes or {}
    for model_type in sorted(grouped):
        accuracies = grouped[model_type]
        per_model_type[model_type] = {
            "mean_test_accuracy": float(sum(accuracies) / len(accuracies)),
            "client_count": len(accuracies),
            "parameter_bytes": int(parameter_bytes.get(model_type, 0)),
        }

    means = [entry["mean_test_accuracy"] for entry in per_model_type.values()]
    fairness_gap = float(max(means) - min(means)) if means else None
    capacity_gap = summarize_capacity_gap(per_model_type)
    return {
        "per_model_type": per_model_type,
        "fairness_gap": fairness_gap,
        "capacity_gap": capacity_gap,
    }


def summarize_capacity_gap(per_model_type):
    """Compare accuracy of the smallest and largest parameter-count groups."""

    eligible = [
        (model_type, values)
        for model_type, values in per_model_type.items()
        if int(values.get("parameter_bytes", 0)) > 0
        and values.get("mean_test_accuracy") is not None
    ]
    if not eligible:
        return None
    smallest_type, smallest = min(
        eligible, key=lambda item: (int(item[1]["parameter_bytes"]), item[0])
    )
    largest_type, largest = max(
        eligible, key=lambda item: (int(item[1]["parameter_bytes"]), item[0])
    )
    return {
        "smallest_model_type": smallest_type,
        "smallest_parameter_bytes": int(smallest["parameter_bytes"]),
        "smallest_mean_test_accuracy": float(smallest["mean_test_accuracy"]),
        "largest_model_type": largest_type,
        "largest_parameter_bytes": int(largest["parameter_bytes"]),
        "largest_mean_test_accuracy": float(largest["mean_test_accuracy"]),
        "largest_minus_smallest_accuracy": float(
            largest["mean_test_accuracy"] - smallest["mean_test_accuracy"]
        ),
    }


def summarize_distillation_communication(records):
    """Summarize classifier-logit traffic accumulated by architecture."""

    summary = {}
    for model_type, record in sorted(records.items()):
        uplink = int(record.get("uplink_bytes", 0))
        downlink = int(record.get("downlink_bytes", 0))
        summary[str(model_type)] = {
            "uplink_bytes": uplink,
            "downlink_bytes": downlink,
            "total_bytes": uplink + downlink,
            "uplink_events": int(record.get("uplink_events", 0)),
            "downlink_events": int(record.get("downlink_events", 0)),
            "payload": "transfer_logits_float32",
        }
    return summary
