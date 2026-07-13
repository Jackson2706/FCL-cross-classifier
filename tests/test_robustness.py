"""Fast CPU-only checks for robustness scenarios and privacy-risk proxies."""

import os
import sys
import tempfile
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "system"))

from flcore.robustness.privacy import (  # noqa: E402
    membership_inference_proxy,
    memorization_summary,
    nearest_neighbor_distances,
)
from flcore.robustness.scenarios import RobustnessScenario  # noqa: E402
from flcore.servers.server_ourv2 import OursV2  # noqa: E402


def scenario(**kwargs):
    config = dict(
        robustness_mode="none", corrupted_client_fraction=0.5,
        label_noise_rate=0.0, bn_noise_std=0.0,
        malicious_confidence_attack=False,
    )
    config.update(kwargs)
    return RobustnessScenario(6, 5, seed=17, **config)


def test_corrupted_selection_is_deterministic():
    first = scenario().corrupted_client_ids
    second = scenario().corrupted_client_ids
    different_seed = RobustnessScenario(
        6, 5, seed=18, robustness_mode="none",
        corrupted_client_fraction=0.5,
    ).corrupted_client_ids
    assert first == second
    assert len(first) == 3
    assert first != different_seed


def test_label_noise_rate_is_seeded_and_wrong():
    data = TensorDataset(torch.arange(200).float().view(-1, 1), torch.arange(200) % 5)
    attack = scenario(label_noise_rate=0.3)
    client_id = attack.corrupted_client_ids[0]
    attacked = attack.adapt_dataset(client_id, data, task=2)
    changed = sum(int(attacked[i][1]) != int(data[i][1]) for i in range(len(data)))
    assert changed == 60
    repeat = attack.adapt_dataset(client_id, data, task=2)
    assert [int(attacked[i][1]) for i in range(len(data))] == [
        int(repeat[i][1]) for i in range(len(data))
    ]
    _, batch_labels = next(iter(DataLoader(attacked, batch_size=32)))
    assert batch_labels.dtype == data.tensors[1].dtype


class TinyBNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(2)
        self.conv = nn.Conv2d(2, 5, 1)

    def forward(self, inputs):
        return self.conv(self.bn(inputs)).mean(dim=(2, 3))


def test_upload_and_bn_corruption_change_reports_only():
    model = TinyBNModel()
    attack = scenario(
        robustness_mode="corrupted_uploads,corrupted_bn", bn_noise_std=2.0,
    )
    client_id = attack.corrupted_client_ids[0]
    original_weight = model.conv.weight.detach().clone()
    original_mean = model.bn.running_mean.detach().clone()
    attacked = attack.corrupt_upload(client_id, model)
    assert not torch.equal(attacked.conv.weight, original_weight)
    assert not torch.equal(attacked.bn.running_mean, original_mean)
    assert torch.equal(model.conv.weight, original_weight)
    assert torch.equal(model.bn.running_mean, original_mean)


def test_malicious_teacher_is_wrong_and_confident():
    attack = scenario(malicious_confidence_attack=True)
    client_id = attack.corrupted_client_ids[0]
    labels = torch.tensor([0, 2, 4])
    logits = torch.randn(3, 5)
    attacked = attack.teacher_logits(client_id, logits, labels)
    probabilities = attacked.softmax(dim=1)
    assert torch.equal(attacked.argmax(dim=1), (labels + 1) % 5)
    assert bool((probabilities.max(dim=1).values > 0.999).all())


def test_summary_separates_corrupted_and_clean_trust():
    attack = scenario()
    trust = {
        client_id: (0.1 if attack.is_corrupted(client_id) else 0.9)
        for client_id in range(6)
    }
    summary = attack.summary(trust, retained_client_ids=[0, 1, 2, 3, 4, 5])
    assert abs(summary["mean_trust_corrupted"] - 0.1) < 1e-12
    assert abs(summary["mean_trust_clean"] - 0.9) < 1e-12


def test_imbalance_and_single_class_views_are_skewed():
    labels = torch.tensor([0] * 20 + [1] * 20 + [2] * 20)
    data = TensorDataset(torch.arange(60).float().view(-1, 1), labels)
    imbalance = scenario(
        robustness_mode="extreme_imbalance",
        extreme_imbalance_majority_fraction=0.9,
    )
    client_id = imbalance.corrupted_client_ids[0]
    skewed = imbalance.adapt_dataset(client_id, data)
    skewed_labels = [int(skewed[index][1]) for index in range(len(skewed))]
    assert skewed_labels.count(0) / len(skewed_labels) >= 0.9

    expert = scenario(
        robustness_mode="single_class_expert", single_class_expert_label=2,
    )
    expert_data = expert.adapt_dataset(expert.corrupted_client_ids[0], data)
    assert {int(expert_data[index][1]) for index in range(len(expert_data))} == {2}


def test_nn_distance_and_memorization_are_exact():
    generated = torch.tensor([[0.0], [2.0], [10.0]])
    real = torch.tensor([[0.0], [3.0]])
    distances = nearest_neighbor_distances(generated, real)
    assert torch.equal(distances, torch.tensor([0.0, 1.0, 7.0]))
    summary = memorization_summary(distances, tau=1.0)
    assert summary["min"] == 0.0
    assert abs(summary["mean"] - 8.0 / 3.0) < 1e-6
    assert abs(summary["generator_memorization_score"] - 2.0 / 3.0) < 1e-6


def test_membership_proxy_sign_is_sane():
    targets = torch.tensor([0, 1, 0, 1])
    member = torch.tensor([[8.0, -2.0], [-2.0, 8.0], [6.0, 0.0], [0.0, 6.0]])
    heldout = torch.zeros(4, 2)
    proxy = membership_inference_proxy(member, targets, heldout, targets)
    assert proxy["confidence_gap"] > 0.0
    assert proxy["loss_gap"] > 0.0
    assert proxy["heuristic_only"] is True


def test_privacy_collection_preserves_rng_modes_and_parameters():
    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3 + 2, 4)

        def forward(self, noise, labels):
            one_hot = torch.nn.functional.one_hot(labels, 2).to(noise)
            return self.linear(torch.cat([noise, one_hot], dim=1)).view(-1, 1, 2, 2)

    class Client:
        def __init__(self, dataset):
            self.train_data = dataset

        def load_test_data(self, task):
            del task
            return [(self.train_data.tensors[0], self.train_data.tensors[1])]

    dataset = TensorDataset(torch.randn(8, 1, 2, 2), torch.arange(8) % 2)
    classifier = nn.Sequential(nn.Flatten(), nn.Linear(4, 2)).train()
    generator = Generator().train()
    with tempfile.TemporaryDirectory() as directory:
        fake = SimpleNamespace(
            offlog=True,
            robustness_config={
                "privacy_proxies": True, "privacy_max_samples": 4,
                "memorization_tau": 1.0,
            },
            get_seen_classes=lambda: [0, 1], clients=[Client(dataset)],
            global_model=classifier, global_generator=generator,
            num_tasks=1, nz=3, device=torch.device("cpu"),
            _privacy_records=[], save_folder=directory,
        )
        before_rng = torch.random.get_rng_state().clone()
        before_parameters = [p.detach().clone() for p in classifier.parameters()]
        OursV2._record_privacy_proxies(fake, task=0)
        assert torch.equal(torch.random.get_rng_state(), before_rng)
        assert classifier.training and generator.training
        assert all(torch.equal(before, after) for before, after in zip(
            before_parameters, classifier.parameters()
        ))
        assert os.path.exists(os.path.join(directory, "privacy_log.json"))


if __name__ == "__main__":
    test_corrupted_selection_is_deterministic()
    test_label_noise_rate_is_seeded_and_wrong()
    test_upload_and_bn_corruption_change_reports_only()
    test_malicious_teacher_is_wrong_and_confident()
    test_summary_separates_corrupted_and_clean_trust()
    test_imbalance_and_single_class_views_are_skewed()
    test_nn_distance_and_memorization_are_exact()
    test_membership_proxy_sign_is_sane()
    test_privacy_collection_preserves_rng_modes_and_parameters()
    print("robustness tests passed")
