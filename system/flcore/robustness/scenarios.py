"""Deterministic, config-gated robustness attack adapters.

These attacks are simulation mechanisms, not security guarantees.  The adapter
uses private RNG streams so merely configuring or measuring a scenario does not
advance the training RNG streams.
"""

import copy
import math
import random

import torch
from torch import nn
from torch.utils.data import Dataset, Subset


ROBUSTNESS_CONFIG_DEFAULTS = {
    "robustness_mode": "none",
    "corrupted_client_fraction": 0.0,
    "label_noise_rate": 0.0,
    "upload_noise_std": 5.0,
    "bn_noise_std": 0.0,
    "malicious_confidence_attack": False,
    "malicious_logit_value": 20.0,
    "extreme_imbalance_majority_fraction": 0.95,
    "single_class_expert_label": None,
    "robustness_seed": None,
    "privacy_proxies": False,
    "memorization_tau": None,
    "privacy_max_samples": 256,
}


class LabelNoiseDataset(Dataset):
    """Dataset view with a fixed, precomputed set of wrong labels."""

    def __init__(self, dataset, rate, num_classes, seed):
        self.dataset = dataset
        self.rate = max(0.0, min(1.0, float(rate)))
        self.num_classes = int(num_classes)
        generator = torch.Generator().manual_seed(int(seed))
        count = int(round(len(dataset) * self.rate))
        chosen = torch.randperm(len(dataset), generator=generator)[:count].tolist()
        self.replacements = {}
        for index in chosen:
            label = int(dataset[index][1])
            offset = int(torch.randint(
                1, self.num_classes, (1,), generator=generator
            ).item())
            self.replacements[index] = (label + offset) % self.num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        if index not in self.replacements:
            return item
        values = list(item)
        original = values[1]
        replacement = self.replacements[index]
        if torch.is_tensor(original):
            values[1] = original.new_tensor(replacement)
        else:
            try:
                values[1] = type(original)(replacement)
            except TypeError:
                values[1] = replacement
        return tuple(values)


def _labels(dataset):
    return [int(dataset[index][1]) for index in range(len(dataset))]


class RobustnessScenario:
    """Compose attacks for a deterministic subset of clients."""

    MODES = {
        "none", "label_noise", "corrupted_uploads", "corrupted_bn",
        "malicious_confidence", "extreme_imbalance", "single_class_expert",
        "client_dropout", "composed", "all",
    }

    def __init__(self, num_clients, num_classes, seed=0, **config):
        self.config = {
            key: config.get(key, default)
            for key, default in ROBUSTNESS_CONFIG_DEFAULTS.items()
        }
        raw_mode = str(self.config["robustness_mode"]).strip().lower()
        self.modes = {part.strip() for part in raw_mode.split(",") if part.strip()}
        if not self.modes:
            self.modes = {"none"}
        unknown = self.modes - self.MODES
        if unknown:
            raise ValueError(f"Unknown robustness_mode(s): {sorted(unknown)}")
        self.num_clients = int(num_clients)
        self.num_classes = int(num_classes)
        configured_seed = self.config["robustness_seed"]
        self.seed = int(seed if configured_seed is None else configured_seed)
        fraction = max(0.0, min(1.0, float(
            self.config["corrupted_client_fraction"]
        )))
        count = int(math.floor(self.num_clients * fraction))
        if fraction > 0.0 and count == 0:
            count = 1
        chooser = random.Random(self.seed)
        self.corrupted_client_ids = tuple(sorted(
            chooser.sample(range(self.num_clients), count)
        ))

    @property
    def enabled(self):
        return self.modes != {"none"} or any((
            float(self.config["label_noise_rate"]) > 0.0,
            float(self.config["bn_noise_std"]) > 0.0,
            bool(self.config["malicious_confidence_attack"]),
        )) or "corrupted_uploads" in self.modes

    def is_corrupted(self, client_id):
        return int(client_id) in self.corrupted_client_ids

    def has(self, mode):
        return mode in self.modes or "all" in self.modes

    def adapt_dataset(self, client_id, dataset, task=0):
        """Return an attacked view of local real training data."""

        if not self.is_corrupted(client_id):
            return dataset
        result = dataset
        labels = _labels(result)
        if self.has("single_class_expert") and labels:
            requested = self.config["single_class_expert_label"]
            expert_label = min(labels) if requested is None else int(requested)
            indices = [i for i, label in enumerate(labels) if label == expert_label]
            if indices:
                result = Subset(result, indices)
                labels = [expert_label] * len(indices)
        elif self.has("extreme_imbalance") and labels:
            majority = min(labels)
            majority_indices = [i for i, label in enumerate(labels) if label == majority]
            minority_indices = [i for i, label in enumerate(labels) if label != majority]
            target = float(self.config["extreme_imbalance_majority_fraction"])
            target = max(0.0, min(0.999999, target))
            keep_minority = int(len(majority_indices) * (1.0 - target) / max(target, 1e-12))
            generator = torch.Generator().manual_seed(
                self.seed + 1009 * int(client_id) + 9176 * int(task)
            )
            order = torch.randperm(len(minority_indices), generator=generator).tolist()
            indices = majority_indices + [minority_indices[i] for i in order[:keep_minority]]
            result = Subset(result, sorted(indices))
        rate = float(self.config["label_noise_rate"])
        if rate > 0.0 and len(result):
            result = LabelNoiseDataset(
                result, rate, self.num_classes,
                self.seed + 7919 * int(client_id) + 104729 * int(task),
            )
        return result

    def corrupt_upload(self, client_id, model):
        """Copy and corrupt a reported model; never mutate the client model."""

        if not self.is_corrupted(client_id):
            return model
        attacked = copy.deepcopy(model)
        generator = torch.Generator(device="cpu").manual_seed(
            self.seed + 15485863 * (int(client_id) + 1)
        )
        if self.has("corrupted_uploads"):
            std = float(self.config["upload_noise_std"])
            with torch.no_grad():
                for parameter in attacked.parameters():
                    noise = torch.randn(
                        parameter.shape, generator=generator, dtype=parameter.dtype,
                        device="cpu",
                    ).to(parameter.device)
                    parameter.add_(noise, alpha=std)
        if self.has("corrupted_bn") or float(self.config["bn_noise_std"]) > 0.0:
            std = float(self.config["bn_noise_std"])
            with torch.no_grad():
                for module in attacked.modules():
                    if isinstance(module, nn.modules.batchnorm._BatchNorm):
                        for statistic in (module.running_mean, module.running_var):
                            if statistic is not None:
                                noise = torch.randn(
                                    statistic.shape, generator=generator,
                                    dtype=statistic.dtype, device="cpu",
                                ).to(statistic.device)
                                statistic.add_(noise, alpha=std)
        return attacked

    def teacher_logits(self, client_id, logits, labels=None):
        """Replace a malicious teacher's outputs with confident wrong logits."""

        active = self.has("malicious_confidence") or bool(
            self.config["malicious_confidence_attack"]
        )
        if not active or not self.is_corrupted(client_id):
            return logits
        if labels is None:
            labels = logits.detach().argmax(dim=-1)
        wrong = (labels.to(logits.device).long() + 1) % logits.shape[-1]
        value = float(self.config["malicious_logit_value"])
        attacked = logits * 0.0 - value
        return attacked.scatter(1, wrong.view(-1, 1), value)

    def summary(self, client_trust=None, retained_client_ids=None):
        trust = client_trust or {}
        bad = [float(value) for cid, value in trust.items() if self.is_corrupted(cid)]
        clean = [float(value) for cid, value in trust.items() if not self.is_corrupted(cid)]
        retained = set(retained_client_ids or [])
        return {
            "mode": sorted(self.modes),
            "corrupted_client_ids": list(self.corrupted_client_ids),
            "clean_client_ids": [
                cid for cid in range(self.num_clients) if not self.is_corrupted(cid)
            ],
            "mean_trust_corrupted": sum(bad) / len(bad) if bad else None,
            "mean_trust_clean": sum(clean) / len(clean) if clean else None,
            "retained_corrupted_client_ids": sorted(
                retained.intersection(self.corrupted_client_ids)
            ),
            "filtered_corrupted_client_ids": sorted(
                set(self.corrupted_client_ids) - retained
            ) if retained else [],
            "client_dropout": "reuses TaskScheduler.client_dropout_rate; no duplicate dropout RNG",
        }
