"""Plain-script tests for scripts/aggregate_seeds.py (no pytest dependency)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import aggregate_seeds as agg
import plots


def test_flatten():
    flat = agg.flatten({
        "global": {"average_accuracy": 0.5, "forgetting": 0.1},
        "async": {"active_client_count_per_round": [4, 4]},
        "flag": True,
        "note": "ignored_string",
    })
    assert flat["global.average_accuracy"] == 0.5
    assert flat["global.forgetting"] == 0.1
    assert flat["async.active_client_count_per_round[0]"] == 4.0
    assert flat["flag"] == 1.0
    assert "note" not in flat  # strings dropped


def test_aggregate_metrics_mean_std_n():
    dicts = [
        {"acc": 0.4, "bwt": -0.1},
        {"acc": 0.6, "bwt": -0.3},
        {"acc": 0.5},  # missing bwt
    ]
    out = agg.aggregate_metrics(dicts)
    assert out["acc"]["n"] == 3
    assert abs(out["acc"]["mean"] - 0.5) < 1e-9
    # population std of [0.4,0.6,0.5]
    expected_std = math.sqrt(sum((v - 0.5) ** 2 for v in [0.4, 0.6, 0.5]) / 3)
    assert abs(out["acc"]["std"] - expected_std) < 1e-9
    assert out["bwt"]["n"] == 2  # only two runs had bwt
    assert abs(out["bwt"]["mean"] - (-0.2)) < 1e-9


def test_single_value_std_zero():
    out = agg.aggregate_metrics([{"x": 3.0}])
    assert out["x"]["n"] == 1
    assert out["x"]["std"] == 0.0


def test_strip_seed_grouping():
    assert agg._strip_seed("baseline_seed0") == "baseline"
    assert agg._strip_seed("rel_multi_signal_seed12") == "rel_multi_signal"
    assert agg._strip_seed("async_fixed_groups") == "async_fixed_groups"
    runs = [
        {"note": "baseline_seed0", "config": {}, "metrics": {"acc": 0.4}},
        {"note": "baseline_seed1", "config": {}, "metrics": {"acc": 0.6}},
        {"note": "paper_seed0", "config": {}, "metrics": {"acc": 0.9}},
    ]
    groups = agg.group_runs(runs)
    assert set(groups) == {"baseline", "paper"}
    assert len(groups["baseline"]) == 2


def test_boundary_margin_extractor_uses_tasks():
    boundary_log = {
        "tasks": [{
            "task": 0,
            "real_test": {
                "true_class_margin": {
                    "histogram": {"bin_edges": [-1.0, 0.0, 1.0], "counts": [2, 3]}
                }
            },
        }]
    }
    histogram = plots.extract_margin_histogram(boundary_log)
    assert histogram["counts"] == [2, 3]


if __name__ == "__main__":
    test_flatten()
    test_aggregate_metrics_mean_std_n()
    test_single_value_std_zero()
    test_strip_seed_grouping()
    test_boundary_margin_extractor_uses_tasks()
    print("aggregate tests passed")
