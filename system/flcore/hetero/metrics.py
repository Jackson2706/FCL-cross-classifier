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
    return {
        "per_model_type": per_model_type,
        "fairness_gap": fairness_gap,
    }
