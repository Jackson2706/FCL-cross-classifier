"""Fast CPU-only checks for continual-learning metrics."""

import math
import os
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "system"))

from flcore.metrics.average_anytime_accuracy import metric_average_anytime_accuracy
from flcore.metrics.average_forgetting import metric_average_forgetting
from flcore.metrics.backward_transfer import metric_backward_transfer


ACCURACY_MATRIX = [
    [0.80, 0.00, 0.00],
    [0.70, 0.75, 0.00],
    [0.60, 0.70, 0.90],
]


def test_hand_computed_metrics():
    expected_forgetting = ((0.80 - 0.60) + (0.75 - 0.70)) / 2
    expected_bwt = ((0.60 - 0.80) + (0.70 - 0.75)) / 2
    expected_aaa = (0.80 + (0.70 + 0.75) / 2 + (0.60 + 0.70 + 0.90) / 3) / 3

    assert math.isclose(metric_average_forgetting(2, ACCURACY_MATRIX), expected_forgetting)
    assert math.isclose(metric_backward_transfer(ACCURACY_MATRIX), expected_bwt)
    assert math.isclose(metric_average_anytime_accuracy(ACCURACY_MATRIX), expected_aaa)


def test_partial_and_single_task_matrices():
    assert metric_backward_transfer([[0.80]]) == 0.0
    assert metric_average_anytime_accuracy([[0.80]]) == 0.80

    partial = [[0.80], [0.70, 0.75], [0.60]]
    assert math.isclose(metric_backward_transfer(partial), -0.20)
    assert math.isclose(metric_average_anytime_accuracy(partial), (0.80 + 0.725 + 0.60) / 3)


if __name__ == "__main__":
    test_hand_computed_metrics()
    test_partial_and_single_task_matrices()
    print("metric tests passed")
