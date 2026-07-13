"""Fast CPU checks for asynchronous completion-watermark consolidation."""

import os
import sys

import torch
from torch import nn


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "system"))

from flcore.scheduler.consolidation import ConsolidationManager


class TinyTeacher(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.bn = nn.BatchNorm1d(1)
        self.weight = nn.Parameter(torch.tensor([float(value)]))

    def forward(self, inputs):
        return inputs * self.weight


def _complete(manager, boundary, client_id, value, labels, round_id):
    return manager.record_completion(
        boundary_k=boundary,
        client_id=client_id,
        model=TinyTeacher(value),
        class_labels=labels,
        sample_count=10 + client_id,
        global_round=round_id,
    )


def test_watermark_fires_once_and_unions_participant_labels():
    manager = ConsolidationManager({0: {0, 1}}, trigger="watermark")
    assert _complete(manager, 0, 0, 1.0, [1, 2], 2)
    assert manager.pop_ready(2) == []
    assert _complete(manager, 0, 1, 2.0, [2, 3], 3)

    events = manager.pop_ready(3)
    assert len(events) == 1
    event = events[0]
    assert event.boundary_k == 0
    assert event.trigger == "watermark"
    assert event.participating_client_ids == (0, 1)
    assert event.missing_client_ids == ()
    assert event.newly_consolidated_classes == (1, 2, 3)
    assert event.globally_consolidated_classes == (1, 2, 3)
    assert manager.pop_ready(4) == []


def test_boundary_snapshot_is_frozen_and_never_replaced():
    manager = ConsolidationManager({0: {0, 1}, 1: {0}})
    original = TinyTeacher(1.0)
    assert manager.record_completion(0, 0, original, [4], 7, 0)
    original.weight.data.fill_(9.0)

    # A duplicate/newer model for the same boundary cannot overwrite the first.
    assert not manager.record_completion(0, 0, original, [99], 99, 1)
    assert manager.record_completion(1, 0, original, [5], 8, 1)
    old_snapshot = manager.snapshot_for(0, 0)
    new_snapshot = manager.snapshot_for(1, 0)
    assert old_snapshot.model.training is False
    assert all(not parameter.requires_grad for parameter in old_snapshot.model.parameters())
    assert old_snapshot.model.weight.item() == 1.0
    assert old_snapshot.class_labels == (4,)
    assert old_snapshot.sample_count == 7
    assert old_snapshot.bn_statistics
    assert new_snapshot.model.weight.item() == 9.0
    old_snapshot.model.weight.data.fill_(12.0)
    assert manager.snapshot_for(0, 0).model.weight.item() == 1.0


def test_timeout_records_missing_and_late_arrival_does_not_rerun():
    manager = ConsolidationManager(
        {0: {0, 1}}, trigger="watermark", timeout_rounds=2
    )
    assert _complete(manager, 0, 0, 1.0, [7], 4)
    assert manager.pop_ready(5) == []
    events = manager.pop_ready(6)
    assert len(events) == 1
    assert events[0].trigger == "timeout"
    assert events[0].participating_client_ids == (0,)
    assert events[0].missing_client_ids == (1,)
    assert events[0].newly_consolidated_classes == (7,)

    assert not _complete(manager, 0, 1, 2.0, [8], 7)
    assert manager.pop_ready(7) == []
    assert manager.late_arrivals == [
        {"boundary_k": 0, "client_id": 1, "global_round": 7}
    ]
    assert len(manager.event_log) == 1


if __name__ == "__main__":
    test_watermark_fires_once_and_unions_participant_labels()
    test_boundary_snapshot_is_frozen_and_never_replaced()
    test_timeout_records_missing_and_late_arrival_does_not_rerun()
    print("consolidation tests passed")
