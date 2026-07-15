"""Deterministic client task clocks, participation, and async metrics."""

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, Mapping, Optional

from flcore.metrics.average_anytime_accuracy import metric_average_anytime_accuracy
from flcore.metrics.backward_transfer import metric_backward_transfer
from flcore.utils.json_io import atomic_write_json


ASYNC_CONFIG_DEFAULTS = {
    "async_mode": False,
    "client_task_speed_distribution": "fixed_groups",
    "max_task_lag": 0,
    "client_dropout_rate": 0.0,
    "permanent_dropout_rate": 0.0,
    "partial_participation_rate": 1.0,
    "task_schedule_seed": 0,
    "num_speed_groups": 2,
    "speed_interval": [0.5, 1.0],
    "custom_schedule_path": None,
    "allow_task_jumps": False,
}

SUPPORTED_MODES = {
    "synchronous",
    "fixed_groups",
    "uniform_speed",
    "poisson_clock",
    "custom_schedule",
}


@dataclass(frozen=True)
class ClientRoundState:
    """A client's task-clock and upload eligibility for one server round."""

    task_id: int
    active: bool
    dropped: bool


class TaskScheduler:
    """Materialize deterministic per-client task clocks and participation.

    The scheduler owns a private Python RNG, so constructing or querying schedules
    never consumes Python/NumPy/Torch training RNG state. ``max_task_lag=0`` is a
    task barrier: an advance is admitted only when every non-permanently-dropped
    client can cross that boundary together.
    """

    def __init__(
        self,
        num_clients: int,
        num_tasks: int,
        rounds_per_task: int,
        mode: str = "synchronous",
        max_task_lag: int = 0,
        client_dropout_rate: float = 0.0,
        permanent_dropout_rate: float = 0.0,
        partial_participation_rate: float = 1.0,
        seed: int = 0,
        num_speed_groups: int = 2,
        speed_interval=(0.5, 1.0),
        custom_schedule_path: Optional[str] = None,
        allow_task_jumps: bool = False,
    ):
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unknown task scheduler mode {mode!r}; expected one of {sorted(SUPPORTED_MODES)}")
        if num_clients < 0 or num_tasks <= 0 or rounds_per_task < 0:
            raise ValueError("num_clients >= 0, num_tasks > 0, and rounds_per_task >= 0 are required")
        if max_task_lag < 0:
            raise ValueError("max_task_lag must be non-negative")
        for name, value in (
            ("client_dropout_rate", client_dropout_rate),
            ("permanent_dropout_rate", permanent_dropout_rate),
            ("partial_participation_rate", partial_participation_rate),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if num_speed_groups <= 0:
            raise ValueError("num_speed_groups must be positive")
        if len(speed_interval) != 2 or float(speed_interval[0]) <= 0 or float(speed_interval[1]) < float(speed_interval[0]):
            raise ValueError("speed_interval must be [positive_min, max] with max >= min")
        if mode == "custom_schedule" and not custom_schedule_path:
            raise ValueError("custom_schedule_path is required for custom_schedule mode")

        self.num_clients = int(num_clients)
        self.num_tasks = int(num_tasks)
        self.rounds_per_task = int(rounds_per_task)
        self.mode = mode
        self.max_task_lag = int(max_task_lag)
        self.client_dropout_rate = float(client_dropout_rate)
        self.permanent_dropout_rate = float(permanent_dropout_rate)
        self.partial_participation_rate = float(partial_participation_rate)
        self.seed = int(seed)
        self.num_speed_groups = int(num_speed_groups)
        self.speed_interval = (float(speed_interval[0]), float(speed_interval[1]))
        self.allow_task_jumps = bool(allow_task_jumps)
        self._rng = random.Random(self.seed)

        self._states_by_round: Dict[int, Dict[int, ClientRoundState]] = {}
        self._latest_state: Dict[int, ClientRoundState] = {}
        self._task_ids = {client_id: 0 for client_id in range(self.num_clients)}
        self._permanent_dropped = {
            client_id
            for client_id in range(self.num_clients)
            if self.mode != "synchronous" and self._rng.random() < self.permanent_dropout_rate
        }
        self._phase = {client_id: 0.0 for client_id in range(self.num_clients)}
        self._speeds = {
            client_id: self._rng.uniform(*self.speed_interval)
            for client_id in range(self.num_clients)
        }

        shuffled = list(range(self.num_clients))
        self._rng.shuffle(shuffled)
        self._speed_group = {
            client_id: position % self.num_speed_groups
            for position, client_id in enumerate(shuffled)
        }
        poisson_scale = float(max(1, self.rounds_per_task))
        self._next_arrival = {
            client_id: self._rng.expovariate(1.0 / poisson_scale)
            for client_id in range(self.num_clients)
        }
        self._custom = self._load_custom_schedule(custom_schedule_path) if mode == "custom_schedule" else None
        self._accuracy_records: Dict[int, Dict[int, dict]] = {}

    def task_sequence(self) -> Iterable[int]:
        """Yield shared stages for the unchanged synchronous training branch."""

        return range(self.num_tasks)

    def _load_custom_schedule(self, path):
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"custom schedule does not exist: {path}")
        if path.suffix.lower() == ".csv":
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
        else:
            with path.open() as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                rows = payload
            elif isinstance(payload, dict) and "clients" in payload:
                rows = []
                for client_id, entries in payload["clients"].items():
                    for entry in entries:
                        rows.append(dict(entry, client_id=int(client_id)))
            elif isinstance(payload, dict) and ("schedule" in payload or "rounds" in payload):
                rows = []
                for round_entry in payload.get("schedule", payload.get("rounds")):
                    round_id = round_entry["round"]
                    for client_id, state in round_entry["clients"].items():
                        rows.append(dict(state, round=round_id, client_id=int(client_id)))
            else:
                raise ValueError("custom schedule JSON must be rows, {'clients': ...}, or {'schedule': ...}")

        custom = {client_id: {} for client_id in range(self.num_clients)}
        for row in rows:
            try:
                client_id = int(row["client_id"])
                round_id = int(row["round"])
                task_id = int(row["task_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("every custom row needs integer client_id, round, and task_id") from exc
            if client_id not in custom:
                raise ValueError(f"custom schedule client_id {client_id} is out of bounds")
            if round_id < 0 or not 0 <= task_id < self.num_tasks:
                raise ValueError("custom schedule round must be non-negative and task_id must be in bounds")
            if round_id in custom[client_id]:
                raise ValueError(f"duplicate custom schedule row for client {client_id}, round {round_id}")
            custom[client_id][round_id] = {
                "task_id": task_id,
                "active": _as_bool(row.get("active", True)),
                "dropped": _as_bool(row.get("dropped", False)),
            }

        for client_id, entries in custom.items():
            if not entries or 0 not in entries:
                raise ValueError(f"custom schedule must cover client {client_id} starting at round 0")
            previous = None
            for round_id in sorted(entries):
                task_id = entries[round_id]["task_id"]
                if previous is not None:
                    if task_id < previous:
                        raise ValueError(f"custom task_ids must be monotone for client {client_id}")
                    if not self.allow_task_jumps and task_id - previous > 1:
                        raise ValueError(f"custom task jump for client {client_id} requires allow_task_jumps=true")
                previous = task_id
        return custom

    def _custom_entry(self, client_id, global_round):
        entries = self._custom[client_id]
        eligible = [round_id for round_id in entries if round_id <= global_round]
        return entries[max(eligible)]

    def _proposed_tasks(self, global_round):
        proposed = dict(self._task_ids)
        if self.mode == "synchronous":
            if self.rounds_per_task == 0:
                shared = min(global_round, self.num_tasks - 1)
            else:
                shared = min(global_round // self.rounds_per_task, self.num_tasks - 1)
            return {client_id: shared for client_id in proposed}
        if self.mode == "custom_schedule":
            return {
                client_id: self._custom_entry(client_id, global_round)["task_id"]
                for client_id in proposed
            }

        for client_id, current in self._task_ids.items():
            if client_id in self._permanent_dropped or current >= self.num_tasks - 1:
                continue
            advance = False
            if self.mode == "fixed_groups":
                interval = max(1, self.rounds_per_task) * (self._speed_group[client_id] + 1)
                advance = global_round > 0 and global_round % interval == 0
            elif self.mode == "uniform_speed":
                self._phase[client_id] += self._speeds[client_id]
                if self._phase[client_id] >= 1.0:
                    self._phase[client_id] -= 1.0
                    advance = True
            elif self.mode == "poisson_clock":
                if global_round + 1 >= self._next_arrival[client_id]:
                    advance = True
                    scale = float(max(1, self.rounds_per_task))
                    self._next_arrival[client_id] += self._rng.expovariate(1.0 / scale)
            if advance:
                proposed[client_id] = current + 1
        return proposed

    def _apply_lag_bound(self, proposed):
        eligible = [client_id for client_id in proposed if client_id not in self._permanent_dropped]
        if not eligible:
            return dict(self._task_ids)
        slowest_proposed = min(proposed[client_id] for client_id in eligible)
        bounded = dict(self._task_ids)
        for client_id in eligible:
            candidate = proposed[client_id]
            if candidate <= slowest_proposed + self.max_task_lag:
                bounded[client_id] = candidate
        return bounded

    def state_for_round(self, global_round: int) -> Dict[int, ClientRoundState]:
        """Return the materialized state for a zero-based server round."""

        global_round = int(global_round)
        if global_round < 0:
            raise ValueError("global_round must be non-negative")
        if global_round in self._states_by_round:
            return dict(self._states_by_round[global_round])
        expected = len(self._states_by_round)
        if global_round != expected:
            raise ValueError(f"schedule rounds must be queried sequentially; expected {expected}, got {global_round}")

        proposed = self._proposed_tasks(global_round)
        self._task_ids = self._apply_lag_bound(proposed)
        states = {}
        for client_id in range(self.num_clients):
            if self.mode == "synchronous":
                states[client_id] = ClientRoundState(
                    task_id=self._task_ids[client_id], active=True, dropped=False
                )
                continue
            permanent = client_id in self._permanent_dropped
            if self.mode == "custom_schedule":
                explicit = self._custom_entry(client_id, global_round)
                temporary_drop = explicit["dropped"]
                participates = explicit["active"]
            else:
                temporary_drop = self._rng.random() < self.client_dropout_rate
                participates = self._rng.random() < self.partial_participation_rate
            dropped = permanent or temporary_drop
            states[client_id] = ClientRoundState(
                task_id=self._task_ids[client_id],
                active=bool(not dropped and participates),
                dropped=bool(dropped),
            )
        self._states_by_round[global_round] = states
        self._latest_state = states
        return dict(states)

    @property
    def latest_state(self) -> Mapping[int, ClientRoundState]:
        return dict(self._latest_state)

    def record_client_stage_accuracy(self, client_id, stage, accuracies, sample_counts):
        """Record evaluation of one client immediately after it reaches ``stage``."""

        client_id, stage = int(client_id), int(stage)
        values = [float(value) for value in accuracies]
        counts = [int(value) for value in sample_counts]
        if len(values) != stage + 1 or len(counts) != len(values):
            raise ValueError("stage evaluation must contain tasks 0..stage and matching sample counts")
        self._accuracy_records.setdefault(client_id, {})[stage] = {
            "accuracies": values,
            "sample_counts": counts,
        }

    def metrics(self):
        """Return JSON-ready schedule and per-client-stage async metrics."""

        states = self._latest_state
        continual = async_continual_metrics(self._accuracy_records)
        return {
            "temporal_lag_mean": temporal_lag_mean(states),
            "temporal_lag_std": temporal_lag_std(states),
            "per_client_task_id": per_client_task_id(states),
            "per_client_task_divergence": per_client_task_divergence(states),
            "active_client_count_per_round": [
                active_client_count_per_round(self._states_by_round[round_id])
                for round_id in sorted(self._states_by_round)
            ],
            "accuracy_by_client_stage": accuracy_by_client_stage(self._accuracy_records),
            "old_task_retention_under_async": old_task_retention_under_async(self._accuracy_records),
            "per_client_stage_continual_metrics": continual,
        }

    # Backward-compatible Phase 2a method name.
    synchronous_metrics = metrics

    def materialized_schedule(self, run_seed=None):
        return {
            "seed": self.seed,
            "run_seed": None if run_seed is None else int(run_seed),
            "mode": self.mode,
            "permanent_dropped_clients": sorted(self._permanent_dropped),
            "rounds": [
                {
                    "round": round_id,
                    "clients": {
                        str(client_id): asdict(state)
                        for client_id, state in sorted(self._states_by_round[round_id].items())
                    },
                }
                for round_id in sorted(self._states_by_round)
            ],
        }

    def save_schedule(self, path, run_seed=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, self.materialized_schedule(run_seed), "async_schedule.json")


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _eligible_states(states):
    return {client_id: state for client_id, state in states.items() if not state.dropped}


def _task_lags(states: Mapping[int, ClientRoundState]) -> Dict[int, int]:
    eligible = _eligible_states(states)
    if not eligible:
        return {}
    leading_task = max(state.task_id for state in eligible.values())
    return {client_id: leading_task - state.task_id for client_id, state in eligible.items()}


def temporal_lag_mean(states: Mapping[int, ClientRoundState]) -> float:
    lags = list(_task_lags(states).values())
    return float(mean(lags)) if lags else 0.0


def temporal_lag_std(states: Mapping[int, ClientRoundState]) -> float:
    lags = list(_task_lags(states).values())
    return float(pstdev(lags)) if lags else 0.0


def per_client_task_id(states: Mapping[int, ClientRoundState]) -> Dict[int, int]:
    return {client_id: state.task_id for client_id, state in states.items()}


def per_client_task_divergence(states: Mapping[int, ClientRoundState]) -> Dict[int, float]:
    eligible = _eligible_states(states)
    if not eligible:
        return {}
    center = mean(state.task_id for state in eligible.values())
    return {client_id: float(abs(state.task_id - center)) for client_id, state in eligible.items()}


def active_client_count_per_round(states: Mapping[int, ClientRoundState]) -> int:
    return sum(state.active and not state.dropped for state in states.values())


def accuracy_by_client_stage(accuracy_records: Optional[Mapping[int, object]] = None) -> Dict[int, object]:
    """Return fully materialized task accuracies and sample counts by client stage."""

    if not accuracy_records:
        return {}
    return {
        int(client_id): {
            int(stage): {
                "accuracies": [float(value) for value in record["accuracies"]],
                "sample_counts": [int(value) for value in record["sample_counts"]],
            }
            for stage, record in sorted(stages.items())
        }
        for client_id, stages in sorted(accuracy_records.items())
    }


def old_task_retention_under_async(accuracy_records: Optional[Mapping[int, object]] = None):
    """Accuracy on tasks below each client's latest stage (macro and sample weighted)."""

    if not accuracy_records:
        return {"macro_average": None, "sample_weighted": None, "per_client": {}}
    per_client, weighted_sum, total_samples = {}, 0.0, 0
    for client_id, stages in accuracy_records.items():
        if not stages:
            continue
        stage = max(stages)
        record = stages[stage]
        pairs = list(zip(record["accuracies"][:stage], record["sample_counts"][:stage]))
        if not pairs:
            per_client[int(client_id)] = None
            continue
        client_samples = sum(count for _, count in pairs)
        client_value = sum(value * count for value, count in pairs) / client_samples if client_samples else mean(value for value, _ in pairs)
        per_client[int(client_id)] = float(client_value)
        weighted_sum += sum(value * count for value, count in pairs)
        total_samples += client_samples
    valid = [value for value in per_client.values() if value is not None]
    return {
        "macro_average": float(mean(valid)) if valid else None,
        "sample_weighted": float(weighted_sum / total_samples) if total_samples else None,
        "per_client": per_client,
    }


def async_continual_metrics(accuracy_records: Optional[Mapping[int, object]] = None):
    """Per-client forgetting/BWT/AAA with macro and sample-weighted aggregates."""

    if not accuracy_records:
        return {"per_client": {}, "macro_average": _empty_continual(), "sample_weighted": _empty_continual()}
    per_client = {}
    for client_id, stages in accuracy_records.items():
        if not stages:
            continue
        ordered = [stages[stage] for stage in sorted(stages)]
        matrix = [record["accuracies"] for record in ordered]
        per_client[int(client_id)] = {
            "forgetting": float(_triangular_forgetting(matrix)),
            "backward_transfer": float(metric_backward_transfer(matrix)),
            "average_anytime_accuracy": float(metric_average_anytime_accuracy(matrix)),
            "sample_count": int(sum(ordered[-1]["sample_counts"])),
        }
    macro = _aggregate_continual(per_client, weighted=False)
    weighted = _aggregate_continual(per_client, weighted=True)
    return {"per_client": per_client, "macro_average": macro, "sample_weighted": weighted}


def _empty_continual():
    return {"forgetting": None, "backward_transfer": None, "average_anytime_accuracy": None}


def _triangular_forgetting(matrix):
    """Forgetting for stage-indexed rows where future-task cells do not exist."""

    if len(matrix) < 2:
        return 0.0
    final_stage = len(matrix) - 1
    values = []
    for task_id in range(final_stage):
        history = [
            float(matrix[stage][task_id])
            for stage in range(task_id, final_stage)
            if task_id < len(matrix[stage]) and math.isfinite(matrix[stage][task_id])
        ]
        current = matrix[final_stage][task_id]
        if history and math.isfinite(current):
            values.append(max(history) - float(current))
    return float(mean(values)) if values else 0.0


def _aggregate_continual(per_client, weighted):
    if not per_client:
        return _empty_continual()
    result = {}
    weights = [record["sample_count"] if weighted else 1 for record in per_client.values()]
    denominator = sum(weights)
    for metric in ("forgetting", "backward_transfer", "average_anytime_accuracy"):
        result[metric] = float(sum(record[metric] * weight for record, weight in zip(per_client.values(), weights)) / denominator) if denominator else None
    return result
