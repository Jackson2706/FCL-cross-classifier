"""Average anytime accuracy for the repository's accuracy matrices.

Rows are evaluations after successive training tasks and columns are evaluated
task IDs.  At stage ``r`` only columns ``0..r`` are included, even though the
current evaluator also records accuracies for not-yet-trained future tasks.
"""

import numpy as np


def metric_average_anytime_accuracy(accuracy_matrix):
    """Mean of per-stage average accuracies over tasks seen by each stage.

    Non-finite or absent cells in partially filled matrices are ignored.  Empty
    matrices return ``0.0`` and a single-task matrix returns its sole accuracy.
    """
    if accuracy_matrix is None or len(accuracy_matrix) == 0:
        return 0.0

    stage_averages = []
    for stage, row in enumerate(accuracy_matrix):
        seen_values = [
            float(value)
            for value in row[: stage + 1]
            if np.isfinite(value)
        ]
        if seen_values:
            stage_averages.append(float(np.mean(seen_values)))

    return float(np.mean(stage_averages)) if stage_averages else 0.0
