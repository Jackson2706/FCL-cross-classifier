"""Fast CPU-only checks for deterministic synchronous and async scheduling."""

import json
import math
import os
import random
import sys
import tempfile


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "system"))

from flcore.scheduler.task_scheduler import (
    TaskScheduler,
    accuracy_by_client_stage,
    active_client_count_per_round,
    async_continual_metrics,
    old_task_retention_under_async,
    per_client_task_divergence,
    per_client_task_id,
    temporal_lag_mean,
    temporal_lag_std,
)


def _materialize(mode, rounds=18, **kwargs):
    scheduler = TaskScheduler(
        num_clients=8,
        num_tasks=5,
        rounds_per_task=2,
        mode=mode,
        max_task_lag=4,
        seed=73,
        **kwargs,
    )
    schedules = [scheduler.state_for_round(round_id) for round_id in range(rounds)]
    return scheduler, schedules


def _assert_monotone_nonjumping(schedules):
    for client_id in schedules[0]:
        tasks = [states[client_id].task_id for states in schedules]
        assert tasks == sorted(tasks)
        assert all(after - before <= 1 for before, after in zip(tasks, tasks[1:]))


def test_synchronous_task_sequence_and_metrics():
    scheduler = TaskScheduler(
        num_clients=3,
        num_tasks=4,
        rounds_per_task=2,
        client_dropout_rate=1.0,
        permanent_dropout_rate=1.0,
        partial_participation_rate=0.0,
    )
    observed = []
    for global_round in range(8):
        states = scheduler.state_for_round(global_round)
        task_ids = {state.task_id for state in states.values()}
        assert len(task_ids) == 1
        assert all(state.active and not state.dropped for state in states.values())
        observed.append(task_ids.pop())
    assert list(scheduler.task_sequence()) == list(range(4))
    assert observed == [0, 0, 1, 1, 2, 2, 3, 3]
    assert temporal_lag_mean(states) == temporal_lag_std(states) == 0.0
    assert per_client_task_id(states) == {0: 3, 1: 3, 2: 3}
    assert per_client_task_divergence(states) == {0: 0.0, 1: 0.0, 2: 0.0}
    assert active_client_count_per_round(states) == 3


def test_each_async_mode_is_deterministic_and_nonjumping():
    for mode in ("fixed_groups", "uniform_speed", "poisson_clock"):
        first, first_states = _materialize(mode)
        second, second_states = _materialize(mode)
        assert first.materialized_schedule() == second.materialized_schedule()
        assert first_states == second_states
        _assert_monotone_nonjumping(first_states)


def test_scheduler_rng_is_separate_from_training_rng():
    random.seed(991)
    before = random.getstate()
    scheduler, _ = _materialize("uniform_speed", rounds=3)
    assert scheduler.latest_state
    assert random.getstate() == before


def test_max_task_lag_and_zero_barrier_hold():
    scheduler = TaskScheduler(
        num_clients=8,
        num_tasks=5,
        rounds_per_task=1,
        mode="fixed_groups",
        num_speed_groups=3,
        max_task_lag=1,
        seed=8,
    )
    for round_id in range(20):
        states = scheduler.state_for_round(round_id)
        tasks = [state.task_id for state in states.values() if not state.dropped]
        assert max(tasks) - min(tasks) <= 1

    barrier = TaskScheduler(
        num_clients=8,
        num_tasks=5,
        rounds_per_task=1,
        mode="uniform_speed",
        max_task_lag=0,
        seed=8,
    )
    for round_id in range(20):
        states = barrier.state_for_round(round_id)
        assert len({state.task_id for state in states.values()}) == 1


def test_dropout_and_partial_participation_are_plausible():
    scheduler = TaskScheduler(
        num_clients=20,
        num_tasks=3,
        rounds_per_task=2,
        mode="fixed_groups",
        max_task_lag=2,
        client_dropout_rate=0.25,
        partial_participation_rate=0.5,
        seed=19,
    )
    counts = [
        active_client_count_per_round(scheduler.state_for_round(round_id))
        for round_id in range(100)
    ]
    assert 500 < sum(counts) < 1000
    assert min(counts) < max(counts)

    permanent = TaskScheduler(
        num_clients=100,
        num_tasks=2,
        rounds_per_task=1,
        mode="fixed_groups",
        permanent_dropout_rate=0.25,
        max_task_lag=1,
        seed=4,
    )
    permanent.state_for_round(0)
    dropped = permanent.materialized_schedule()["permanent_dropped_clients"]
    assert 10 < len(dropped) < 40


def test_custom_schedule_validation_and_replay():
    payload = {
        "clients": {
            str(client_id): [
                {"round": 0, "task_id": 0, "active": client_id != 1},
                {"round": 1, "task_id": 1, "active": True},
                {"round": 2, "task_id": 2, "dropped": client_id == 2},
            ]
            for client_id in range(3)
        }
    }
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "schedule.json")
        with open(path, "w") as handle:
            json.dump(payload, handle)
        first = TaskScheduler(3, 3, 1, "custom_schedule", max_task_lag=2, seed=2, custom_schedule_path=path)
        second = TaskScheduler(3, 3, 1, "custom_schedule", max_task_lag=2, seed=2, custom_schedule_path=path)
        for round_id in range(3):
            assert first.state_for_round(round_id) == second.state_for_round(round_id)
        assert not first.materialized_schedule()["rounds"][0]["clients"]["1"]["active"]
        assert first.materialized_schedule()["rounds"][2]["clients"]["2"]["dropped"]

        replay_path = os.path.join(directory, "materialized.json")
        first.save_schedule(replay_path)
        replay = TaskScheduler(
            3, 3, 1, "custom_schedule", max_task_lag=2,
            custom_schedule_path=replay_path,
        )
        replay_states = [replay.state_for_round(round_id) for round_id in range(3)]
        assert replay_states == [first.state_for_round(round_id) for round_id in range(3)]

        payload["clients"]["0"][1]["task_id"] = 2
        with open(path, "w") as handle:
            json.dump(payload, handle)
        try:
            TaskScheduler(3, 3, 1, "custom_schedule", custom_schedule_path=path)
        except ValueError as exc:
            assert "jump" in str(exc)
        else:
            raise AssertionError("custom jumps must require allow_task_jumps=true")


def test_async_metrics_on_hand_built_state():
    records = {
        0: {
            0: {"accuracies": [0.8], "sample_counts": [10]},
            1: {"accuracies": [0.6, 0.9], "sample_counts": [10, 30]},
            2: {"accuracies": [0.5, 0.8, 0.95], "sample_counts": [10, 30, 20]},
        },
        1: {
            0: {"accuracies": [0.5], "sample_counts": [60]},
            1: {"accuracies": [0.4, 0.7], "sample_counts": [60, 20]},
        },
    }
    materialized = accuracy_by_client_stage(records)
    assert materialized[0][1]["accuracies"] == [0.6, 0.9]

    retention = old_task_retention_under_async(records)
    assert math.isclose(retention["macro_average"], ((0.5 * 10 + 0.8 * 30) / 40 + 0.4) / 2)
    assert math.isclose(retention["sample_weighted"], (0.5 * 10 + 0.8 * 30 + 0.4 * 60) / 100)

    metrics = async_continual_metrics(records)
    # Client 0 triangular forgetting is mean((.8-.5), (.9-.8)) = .2.
    assert math.isclose(metrics["per_client"][0]["forgetting"], 0.2)
    assert math.isclose(metrics["per_client"][0]["backward_transfer"], -0.2)
    assert math.isclose(metrics["per_client"][0]["average_anytime_accuracy"],
                        (0.8 + 0.75 + 0.75) / 3)
    assert math.isclose(metrics["macro_average"]["forgetting"], 0.15)
    assert math.isclose(metrics["sample_weighted"]["forgetting"], (0.2 * 60 + 0.1 * 80) / 140)


if __name__ == "__main__":
    test_synchronous_task_sequence_and_metrics()
    test_each_async_mode_is_deterministic_and_nonjumping()
    test_scheduler_rng_is_separate_from_training_rng()
    test_max_task_lag_and_zero_barrier_hold()
    test_dropout_and_partial_participation_are_plausible()
    test_custom_schedule_validation_and_replay()
    test_async_metrics_on_hand_built_state()
    print("scheduler tests passed")
