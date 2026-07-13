"""Fast CPU-only checks for the Phase 2a synchronous scheduler."""

import os
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "system"))

from flcore.scheduler.task_scheduler import (
    TaskScheduler,
    accuracy_by_client_stage,
    active_client_count_per_round,
    old_task_retention_under_async,
    per_client_task_divergence,
    per_client_task_id,
    temporal_lag_mean,
    temporal_lag_std,
)


def test_synchronous_task_sequence():
    scheduler = TaskScheduler(
        num_clients=3,
        num_tasks=4,
        rounds_per_task=2,
        mode="synchronous",
    )

    assert list(scheduler.task_sequence()) == list(range(4))
    observed = []
    for global_round in range(8):
        states = scheduler.state_for_round(global_round)
        task_ids = {state.task_id for state in states.values()}
        assert len(task_ids) == 1
        assert all(state.active and not state.dropped for state in states.values())
        observed.append(task_ids.pop())
    assert observed == [0, 0, 1, 1, 2, 2, 3, 3]


def test_synchronous_async_metrics():
    scheduler = TaskScheduler(num_clients=3, num_tasks=2, rounds_per_task=1)
    states = scheduler.state_for_round(1)

    assert temporal_lag_mean(states) == 0.0
    assert temporal_lag_std(states) == 0.0
    assert per_client_task_id(states) == {0: 1, 1: 1, 2: 1}
    assert per_client_task_divergence(states) == {0: 0.0, 1: 0.0, 2: 0.0}
    assert active_client_count_per_round(states) == 3
    assert old_task_retention_under_async() is None
    assert accuracy_by_client_stage() == {}

    summary = scheduler.synchronous_metrics()
    assert summary["temporal_lag_mean"] == 0.0
    assert summary["temporal_lag_std"] == 0.0
    assert summary["active_client_count_per_round"] == [3]


def test_async_modes_are_not_accidentally_activated():
    try:
        TaskScheduler(num_clients=2, num_tasks=2, rounds_per_task=1, mode="fixed_groups")
    except NotImplementedError:
        pass
    else:
        raise AssertionError("Phase 2a must not execute an asynchronous scheduler mode")


if __name__ == "__main__":
    test_synchronous_task_sequence()
    test_synchronous_async_metrics()
    test_async_modes_are_not_accidentally_activated()
    print("scheduler tests passed")
