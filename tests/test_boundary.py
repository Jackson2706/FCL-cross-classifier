"""Fast CPU-only checks for Phase 4a boundary preservation."""

import os
import sys

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "system"))

from flcore.boundary.attacks import (
    density_gate,
    density_gated_adversarial_loss,
    fgsm_attack,
    pgd_light_attack,
)
from flcore.boundary.evaluation import evaluate_robust_accuracy


class TwoClassLinear(nn.Module):
    def forward(self, inputs):
        value = inputs.reshape(inputs.shape[0], -1).sum(dim=1)
        return torch.stack((value, -value), dim=1)


def test_fgsm_matches_exact_sign_gradient_formula():
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, -0.5], [-0.25, 0.75]]))
    inputs = torch.tensor([[0.4, -0.2], [0.1, 0.3]], requires_grad=True)
    targets = torch.tensor([0, 1])
    epsilon = 0.03
    loss = F.cross_entropy(model(inputs), targets)
    gradient = torch.autograd.grad(loss, inputs)[0]
    expected = inputs.detach() + epsilon * gradient.sign()
    actual = fgsm_attack(
        model, inputs, targets, epsilon, clamp_min=-10.0, clamp_max=10.0
    )
    assert torch.equal(actual, expected)
    assert not actual.requires_grad


def test_pgd_remains_in_linf_epsilon_ball_and_valid_range():
    model = TwoClassLinear()
    inputs = torch.tensor([[0.25], [0.9]])
    targets = torch.tensor([0, 0])
    adversarial = pgd_light_attack(
        model, inputs, targets, epsilon=0.2, steps=4, alpha=0.09,
        clamp_min=0.0, clamp_max=1.0,
    )
    assert torch.max(torch.abs(adversarial - inputs)) <= 0.2 + 1e-7
    assert ((0.0 <= adversarial) & (adversarial <= 1.0)).all()


def test_density_gate_uses_strict_threshold_and_batch_median():
    distances = torch.tensor([0.1, 0.4, 0.2])
    fixed, fixed_tau = density_gate(distances, tau=0.3)
    assert torch.equal(fixed, torch.tensor([True, False, True]))
    assert abs(fixed_tau - 0.3) < 1e-6
    median, median_tau = density_gate(distances, tau=None)
    assert torch.equal(median, torch.tensor([True, False, False]))
    assert abs(median_tau - 0.2) < 1e-6


def test_robust_accuracy_matches_hand_built_case():
    model = TwoClassLinear().eval()
    loader = DataLoader(
        TensorDataset(torch.tensor([[0.1]]), torch.tensor([0])), batch_size=1
    )
    correct, total = evaluate_robust_accuracy(
        model, loader, attack="fgsm", epsilon=0.2,
        clamp_min=-1.0, clamp_max=1.0,
    )
    assert (correct, total) == (0, 1)


def test_adversarial_term_is_exactly_zero_when_lambda_is_zero():
    logits = torch.tensor([[2.0, -1.0], [-0.5, 1.0]], requires_grad=True)
    targets = torch.tensor([0, 1])
    weights = torch.tensor([0.4, 0.8])
    gate = torch.tensor([True, True])
    term = density_gated_adversarial_loss(
        logits, targets, weights, gate, lambda_adv=0.0
    )
    assert term.item() == 0.0


if __name__ == "__main__":
    test_fgsm_matches_exact_sign_gradient_formula()
    test_pgd_remains_in_linf_epsilon_ball_and_valid_range()
    test_density_gate_uses_strict_threshold_and_batch_median()
    test_robust_accuracy_matches_hand_built_case()
    test_adversarial_term_is_exactly_zero_when_lambda_is_zero()
    print("boundary tests passed")
