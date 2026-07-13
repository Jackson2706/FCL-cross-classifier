"""Task-clock scheduling and Phase 2a asynchronous metric interfaces.

Only the synchronous scheduler is executable in Phase 2a.  The metric helpers are
kept here because they consume scheduler state directly; richer accuracy-based
implementations will move behind the same interfaces in Phase 2b.
"""

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Dict, Iterable, Mapping, Optional


ASYNC_CONFIG_DEFAULTS = {
    "async_mode": False,
    "client_task_speed_distribution": "fixed_groups",
    "max_task_lag": 0,
    "client_dropout_rate": 0.0,
    "partial_participation_rate": 1.0,
    "task_schedule_seed": 0,
}


@dataclass(frozen=True)
class ClientRoundState:
    """A client's task-clock and participation state for one global round."""

    task_id: int
    active: bool
    dropped: bool


class TaskScheduler:
    """Map global rounds to client task state.

    Phase 2a intentionally supports only ``mode="synchronous"``.  In that mode,
    all clients are active on ``global_round // rounds_per_task`` and no client
    drops.  This exactly matches the former task-outer/round-inner training loop.
    Other constructor fields are retained, but ignored, until Phase 2b.
    """

    def __init__(
        self,
        num_clients: int,
        num_tasks: int,
        rounds_per_task: int,
        mode: str = "synchronous",
        max_task_lag: int = 0,
        client_dropout_rate: float = 0.0,
        partial_participation_rate: float = 1.0,
        seed: int = 0,
    ):
        if mode != "synchronous":
            raise NotImplementedError(
                f"Task scheduler mode {mode!r} is planned for Phase 2b; "
                "Phase 2a supports only 'synchronous'."
            )
        if num_clients < 0 or num_tasks <= 0 or rounds_per_task < 0:
            raise ValueError("num_clients >= 0, num_tasks > 0, and rounds_per_task >= 0 are required")

        self.num_clients = int(num_clients)
        self.num_tasks = int(num_tasks)
        self.rounds_per_task = int(rounds_per_task)
        self.mode = mode

        # Plumbed for the Phase 2b implementations. Synchronous mode ignores them.
        self.max_task_lag = int(max_task_lag)
        self.client_dropout_rate = float(client_dropout_rate)
        self.partial_participation_rate = float(partial_participation_rate)
        self.seed = int(seed)

        self._states_by_round: Dict[int, Dict[int, ClientRoundState]] = {}
        self._latest_state: Dict[int, ClientRoundState] = {}

    def task_sequence(self) -> Iterable[int]:
        """Yield the shared task stages in the same order as ``range(num_tasks)``."""

        return range(self.num_tasks)

    def state_for_round(self, global_round: int) -> Dict[int, ClientRoundState]:
        """Return a fresh client-state mapping for a zero-based global round."""

        if global_round < 0:
            raise ValueError("global_round must be non-negative")
        if self.rounds_per_task == 0:
            task_id = min(global_round, self.num_tasks - 1)
        else:
            task_id = min(global_round // self.rounds_per_task, self.num_tasks - 1)
        states = {
            client_id: ClientRoundState(task_id=task_id, active=True, dropped=False)
            for client_id in range(self.num_clients)
        }
        self._states_by_round[int(global_round)] = states
        self._latest_state = states
        return dict(states)

    @property
    def latest_state(self) -> Mapping[int, ClientRoundState]:
        return dict(self._latest_state)

    def synchronous_metrics(self) -> dict:
        """Return JSON-ready scheduler metrics collected so far."""

        states = self._latest_state
        return {
            "temporal_lag_mean": temporal_lag_mean(states),
            "temporal_lag_std": temporal_lag_std(states),
            "per_client_task_id": per_client_task_id(states),
            "per_client_task_divergence": per_client_task_divergence(states),
            "active_client_count_per_round": [
                active_client_count_per_round(self._states_by_round[round_id])
                for round_id in sorted(self._states_by_round)
            ],
        }


def _task_lags(states: Mapping[int, ClientRoundState]) -> Dict[int, int]:
    if not states:
        return {}
    leading_task = max(state.task_id for state in states.values())
    return {client_id: leading_task - state.task_id for client_id, state in states.items()}


def temporal_lag_mean(states: Mapping[int, ClientRoundState]) -> float:
    """Mean number of tasks by which clients trail the leading client."""

    lags = list(_task_lags(states).values())
    return float(mean(lags)) if lags else 0.0


def temporal_lag_std(states: Mapping[int, ClientRoundState]) -> float:
    """Population standard deviation of client task lag."""

    lags = list(_task_lags(states).values())
    return float(pstdev(lags)) if lags else 0.0


def per_client_task_id(states: Mapping[int, ClientRoundState]) -> Dict[int, int]:
    """Expose each client's current task clock."""

    return {client_id: state.task_id for client_id, state in states.items()}


def per_client_task_divergence(states: Mapping[int, ClientRoundState]) -> Dict[int, float]:
    """Absolute distance between each task clock and the client mean."""

    if not states:
        return {}
    center = mean(state.task_id for state in states.values())
    return {client_id: float(abs(state.task_id - center)) for client_id, state in states.items()}


def active_client_count_per_round(states: Mapping[int, ClientRoundState]) -> int:
    """Count clients eligible to participate in a round."""

    return sum(state.active and not state.dropped for state in states.values())


def old_task_retention_under_async(
    accuracy_by_stage: Optional[Mapping[int, object]] = None,
) -> Optional[float]:
    """Phase 2b interface; no distinct async retention exists synchronously."""

    return None


def accuracy_by_client_stage(
    accuracy_records: Optional[Mapping[int, object]] = None,
) -> Dict[int, object]:
    """Phase 2b interface; synchronous aggregate evaluation has no client-stage split."""

    return {}
