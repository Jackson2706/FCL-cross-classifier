"""Fail-safe instrumentation helpers for communication and server compute."""

import time
from contextlib import contextmanager


def estimate_model_bytes(model, bytes_per_param=4):
    """Estimate model payload bytes from trainable and frozen parameters."""
    try:
        return int(sum(parameter.numel() for parameter in model.parameters())) * int(bytes_per_param)
    except Exception:
        return 0


class CommunicationAccountant:
    """Accumulate estimated model upload/download payload sizes."""

    def __init__(self, bytes_per_param=4):
        self.bytes_per_param = bytes_per_param
        self.uplink_bytes = 0
        self.downlink_bytes = 0
        self.uplink_events = 0
        self.downlink_events = 0

    def record_uplink(self, model, senders=1):
        try:
            self.uplink_bytes += estimate_model_bytes(model, self.bytes_per_param) * max(0, int(senders))
            self.uplink_events += 1
        except Exception:
            pass

    def record_downlink(self, model, recipients=1):
        try:
            self.downlink_bytes += estimate_model_bytes(model, self.bytes_per_param) * max(0, int(recipients))
            self.downlink_events += 1
        except Exception:
            pass

    def record_uplink_bytes(self, byte_count, events=1):
        try:
            self.uplink_bytes += max(0, int(byte_count))
            self.uplink_events += max(0, int(events))
        except Exception:
            pass

    def record_downlink_bytes(self, byte_count, events=1):
        try:
            self.downlink_bytes += max(0, int(byte_count))
            self.downlink_events += max(0, int(events))
        except Exception:
            pass

    def as_dict(self):
        try:
            divisor = 1024.0 ** 2
            uplink_mb = self.uplink_bytes / divisor
            downlink_mb = self.downlink_bytes / divisor
            return {
                "uplink_mb": uplink_mb,
                "downlink_mb": downlink_mb,
                "total_mb": uplink_mb + downlink_mb,
                "uplink_events": self.uplink_events,
                "downlink_events": self.downlink_events,
            }
        except Exception:
            return {
                "uplink_mb": 0.0,
                "downlink_mb": 0.0,
                "total_mb": 0.0,
                "uplink_events": 0,
                "downlink_events": 0,
            }


class ServerComputeTimer:
    """Accumulate task-boundary server computation wall time by task."""

    def __init__(self):
        self.per_task_seconds = {}

    @contextmanager
    def measure(self, task_id):
        start = time.perf_counter()
        try:
            yield
        finally:
            try:
                elapsed = max(0.0, time.perf_counter() - start)
                key = str(task_id)
                self.per_task_seconds[key] = self.per_task_seconds.get(key, 0.0) + elapsed
            except Exception:
                pass

    def as_dict(self):
        try:
            per_task = {
                key: float(value)
                for key, value in sorted(self.per_task_seconds.items(), key=lambda item: int(item[0]))
            }
            return {"per_task": per_task, "total": float(sum(per_task.values()))}
        except Exception:
            return {"per_task": {}, "total": 0.0}
