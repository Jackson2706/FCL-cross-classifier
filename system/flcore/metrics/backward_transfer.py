"""Backward transfer for the repository's task-accuracy matrix convention.

Rows are evaluations after successive training tasks and columns are evaluated
task IDs.  Thus ``accuracy_matrix[r][c]`` is the accuracy on task ``c`` after
training through task ``r``.  Only columns ``c <= r`` represent tasks seen at
that stage; future-task entries are deliberately ignored.
"""

import numpy as np


def metric_backward_transfer(accuracy_matrix):
    """Return standard BWT at the latest available stage.

    BWT is the mean of ``A[T, i] - A[i, i]`` for prior tasks ``i < T``.
    Missing/non-finite entries in a partially filled matrix are ignored.  A
    single-stage matrix has no prior task and therefore returns ``0.0``.
    """
    if accuracy_matrix is None or len(accuracy_matrix) < 2:
        return 0.0

    final_stage = len(accuracy_matrix) - 1
    final_row = accuracy_matrix[final_stage]
    differences = []

    for task_id in range(final_stage):
        if task_id >= len(final_row) or task_id >= len(accuracy_matrix[task_id]):
            continue
        initial = accuracy_matrix[task_id][task_id]
        current = final_row[task_id]
        if np.isfinite(initial) and np.isfinite(current):
            differences.append(float(current) - float(initial))

    return float(np.mean(differences)) if differences else 0.0
