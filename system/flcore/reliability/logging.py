"""Best-effort structured reliability logging."""

import json
import os

import torch


class ReliabilityLogger:
    """Accumulate classifier batches and append one JSON record per consolidation."""

    def __init__(self, run_dir, accept_threshold=0.0, bins=20):
        self.path = os.path.join(run_dir, "reliability_log.json")
        self.accept_threshold = float(accept_threshold)
        self.bins = int(bins)
        self.records = []
        self.calibration_records = []
        self._batches = []

    def add_batch(self, weights, labels, source_clients, correct, missing_signals=()):
        try:
            self._batches.append({
                "weights": weights.detach().cpu().float(),
                "labels": labels.detach().cpu().long(),
                "source_clients": source_clients.detach().cpu().long(),
                "correct": correct.detach().cpu().bool(),
                "missing_signals": list(missing_signals),
            })
        except Exception:
            pass

    @staticmethod
    def _means_by_key(weights, keys):
        result = {}
        for key in torch.unique(keys).tolist():
            mask = keys == key
            result[str(int(key))] = float(weights[mask].mean())
        return result

    def finish_consolidation(self, mode):
        try:
            if not self._batches:
                return
            weights = torch.cat([batch["weights"] for batch in self._batches])
            labels = torch.cat([batch["labels"] for batch in self._batches])
            sources = torch.cat([batch["source_clients"] for batch in self._batches])
            correct = torch.cat([batch["correct"] for batch in self._batches])
            finite_weights = weights[torch.isfinite(weights)]
            histogram_min = float(finite_weights.min()) if finite_weights.numel() else 0.0
            histogram_max = float(finite_weights.max()) if finite_weights.numel() else 1.0
            if histogram_max <= histogram_min:
                histogram_min, histogram_max = 0.0, 1.0
            edges = torch.linspace(histogram_min, histogram_max, self.bins + 1)
            counts = torch.histc(
                weights, bins=self.bins, min=histogram_min, max=histogram_max
            ).long()
            missing = sorted({
                signal for batch in self._batches for signal in batch["missing_signals"]
            })
            accepted = weights >= self.accept_threshold
            record = {
                "consolidation": len(self.records),
                "mode": str(mode),
                "weight_histogram": {
                    "bin_edges": [float(value) for value in edges],
                    "counts": [int(value) for value in counts],
                },
                "mean_weight": float(weights.mean()),
                "mean_weight_by_class": self._means_by_key(weights, labels),
                "mean_weight_by_source_client": self._means_by_key(weights, sources),
                "accepted": int(accepted.sum()),
                "rejected": int((~accepted).sum()),
                "reliability_vs_correctness": {
                    "correct_count": int(correct.sum()),
                    "incorrect_count": int((~correct).sum()),
                    "mean_weight_correct": (
                        float(weights[correct].mean()) if correct.any() else None
                    ),
                    "mean_weight_incorrect": (
                        float(weights[~correct].mean()) if (~correct).any() else None
                    ),
                },
                "missing_signals": missing,
            }
            self.records.append(record)
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self._write()
        except Exception:
            pass
        finally:
            self._batches = []

    def add_calibration(self, ece, task, global_round, num_samples, bins, temperature):
        """Append a guarded real-test-set ECE record."""

        try:
            self.calibration_records.append({
                "task": int(task),
                "global_round": int(global_round),
                "ece": float(ece),
                "num_samples": int(num_samples),
                "bins": int(bins),
                "temperature": float(temperature),
            })
            self._write()
        except Exception:
            pass

    def _write(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as handle:
            json.dump({
                "calibration": self.calibration_records,
                "consolidations": self.records,
            }, handle, indent=2, sort_keys=True)
            handle.write("\n")
