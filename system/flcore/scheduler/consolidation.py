"""Completion-watermark consolidation for asynchronous continual learning."""

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch
from torch import nn
from flcore.utils.json_io import atomic_write_json


CONSOLIDATION_CONFIG_DEFAULTS = {
    "consolidation_trigger": "watermark",
    "consolidation_timeout_rounds": None,
    "consolidation_quorum": 1.0,
}

SUPPORTED_TRIGGERS = {"watermark", "quorum", "periodic"}


@dataclass(frozen=True)
class TeacherSnapshot:
    """A task-versioned teacher captured before later-task training can alter it."""

    boundary_k: int
    client_id: int
    model: nn.Module
    class_labels: Tuple[int, ...]
    sample_count: int
    bn_statistics: Tuple[Tuple[str, torch.Tensor, torch.Tensor], ...]


@dataclass(frozen=True)
class ConsolidationEvent:
    """One boundary that became ready and can be consolidated exactly once."""

    boundary_k: int
    trigger: str
    global_round: int
    participating_client_ids: Tuple[int, ...]
    missing_client_ids: Tuple[int, ...]
    newly_consolidated_classes: Tuple[int, ...]
    globally_consolidated_classes: Tuple[int, ...]
    teacher_snapshots: Tuple[TeacherSnapshot, ...]

    def as_log_record(self):
        return {
            "boundary_k": self.boundary_k,
            "trigger": self.trigger,
            "participating_client_ids": list(self.participating_client_ids),
            "missing_client_ids": list(self.missing_client_ids),
            "newly_consolidated_classes": list(self.newly_consolidated_classes),
            "global_round": self.global_round,
        }


class ConsolidationManager:
    """Track immutable boundary teachers and emit ordered, one-shot readiness.

    ``eligible_clients_by_boundary`` is derived from the materialized scheduler:
    permanently dropped clients are excluded, as are clients that are not expected
    to reach a particular task during the bounded run.
    """

    def __init__(
        self,
        eligible_clients_by_boundary: Mapping[int, Iterable[int]],
        trigger: str = "watermark",
        timeout_rounds: Optional[int] = None,
        quorum: float = 1.0,
    ):
        trigger = str(trigger).lower()
        if trigger not in SUPPORTED_TRIGGERS:
            raise ValueError(
                f"Unknown consolidation trigger {trigger!r}; "
                f"expected one of {sorted(SUPPORTED_TRIGGERS)}"
            )
        if trigger == "periodic":
            raise NotImplementedError(
                "consolidation_trigger='periodic' is an ablation reserved for a later phase"
            )
        if timeout_rounds is not None and int(timeout_rounds) < 0:
            raise ValueError("consolidation_timeout_rounds must be non-negative or None")
        if not 0.0 < float(quorum) <= 1.0:
            raise ValueError("consolidation_quorum must be in (0, 1]")

        self.trigger = trigger
        self.timeout_rounds = None if timeout_rounds is None else int(timeout_rounds)
        self.quorum = float(quorum)
        self._eligible = {
            int(boundary): frozenset(int(client_id) for client_id in client_ids)
            for boundary, client_ids in eligible_clients_by_boundary.items()
        }
        self._snapshots: Dict[Tuple[int, int], TeacherSnapshot] = {}
        self._opened_round: Dict[int, int] = {}
        self._fired = set()
        self._globally_consolidated_classes = set()
        self._events = []
        self._late_arrivals = []

    @staticmethod
    def _capture_bn_statistics(model):
        statistics = []
        for name, module in model.named_modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                mean = None if module.running_mean is None else module.running_mean.detach().cpu().clone()
                var = None if module.running_var is None else module.running_var.detach().cpu().clone()
                statistics.append((name, mean, var))
        return tuple(statistics)

    @staticmethod
    def _freeze_model(model):
        frozen = copy.deepcopy(model)
        frozen.eval()
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)
        return frozen

    def mark_reached(self, boundary_k: int, client_id: int, global_round: int):
        """Start a boundary's timeout when its first eligible client reaches it."""

        boundary_k, client_id = int(boundary_k), int(client_id)
        if client_id not in self._eligible.get(boundary_k, frozenset()):
            return
        self._opened_round.setdefault(boundary_k, int(global_round))

    def record_completion(
        self,
        boundary_k: int,
        client_id: int,
        model: nn.Module,
        class_labels: Iterable[int],
        sample_count: int,
        global_round: int,
    ) -> bool:
        """Publish a versioned teacher once; duplicates and late arrivals cannot replace it."""

        boundary_k, client_id = int(boundary_k), int(client_id)
        global_round = int(global_round)
        self.mark_reached(boundary_k, client_id, global_round)
        if client_id not in self._eligible.get(boundary_k, frozenset()):
            return False
        key = (boundary_k, client_id)
        if key in self._snapshots:
            return False
        if boundary_k in self._fired:
            self._late_arrivals.append({
                "boundary_k": boundary_k,
                "client_id": client_id,
                "global_round": global_round,
            })
            return False
        frozen_model = self._freeze_model(model)
        self._snapshots[key] = TeacherSnapshot(
            boundary_k=boundary_k,
            client_id=client_id,
            model=frozen_model,
            class_labels=tuple(sorted(set(int(label) for label in class_labels))),
            sample_count=int(sample_count),
            bn_statistics=self._capture_bn_statistics(frozen_model),
        )
        return True

    def snapshot_for(self, boundary_k: int, client_id: int):
        """Return a defensive copy so callers cannot mutate the stored teacher."""

        snapshot = self._snapshots.get((int(boundary_k), int(client_id)))
        return None if snapshot is None else copy.deepcopy(snapshot)

    def _readiness_reason(self, boundary_k, global_round):
        eligible = self._eligible.get(boundary_k, frozenset())
        participants = {
            client_id for candidate_boundary, client_id in self._snapshots
            if candidate_boundary == boundary_k
        }
        if not eligible or not participants:
            return None
        if self.trigger == "watermark" and eligible.issubset(participants):
            return "watermark"
        if self.trigger == "quorum":
            required = max(1, math.ceil(self.quorum * len(eligible)))
            if len(participants) >= required:
                return "quorum"
        opened = self._opened_round.get(boundary_k)
        if (
            self.timeout_rounds is not None
            and opened is not None
            and int(global_round) - opened >= self.timeout_rounds
        ):
            return "timeout"
        return None

    def pop_ready(self, global_round: int):
        """Emit all consecutively ready boundaries, each at most once."""

        ready = []
        for boundary_k in sorted(self._eligible):
            if boundary_k in self._fired:
                continue
            # Preserve replay semantics: later tasks cannot consolidate first.
            if any(previous not in self._fired for previous in self._eligible if previous < boundary_k):
                break
            reason = self._readiness_reason(boundary_k, global_round)
            if reason is None:
                break

            eligible = self._eligible[boundary_k]
            participant_ids = tuple(sorted(
                client_id for candidate_boundary, client_id in self._snapshots
                if candidate_boundary == boundary_k
            ))
            snapshots = tuple(
                copy.deepcopy(self._snapshots[(boundary_k, client_id)])
                for client_id in participant_ids
            )
            new_classes = tuple(sorted({
                label for snapshot in snapshots for label in snapshot.class_labels
            }))
            self._globally_consolidated_classes.update(new_classes)
            event = ConsolidationEvent(
                boundary_k=boundary_k,
                trigger=reason,
                global_round=int(global_round),
                participating_client_ids=participant_ids,
                missing_client_ids=tuple(sorted(eligible.difference(participant_ids))),
                newly_consolidated_classes=new_classes,
                globally_consolidated_classes=tuple(sorted(self._globally_consolidated_classes)),
                teacher_snapshots=snapshots,
            )
            self._fired.add(boundary_k)
            self._events.append(event.as_log_record())
            ready.append(event)
        return ready

    @property
    def late_arrivals(self):
        return copy.deepcopy(self._late_arrivals)

    @property
    def event_log(self):
        return copy.deepcopy(self._events)

    def summary(self):
        return {
            "trigger": self.trigger,
            "boundaries_fired": len(self._events),
            "boundary_ids": [event["boundary_k"] for event in self._events],
            "timeout_boundaries": [
                event["boundary_k"] for event in self._events if event["trigger"] == "timeout"
            ],
            "late_arrival_count": len(self._late_arrivals),
        }

    def save_log(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, self._events, "consolidation_log.json")
