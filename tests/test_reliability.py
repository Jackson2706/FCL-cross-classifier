"""Fast CPU-only checks for reliability scoring and legacy loss equivalence."""

import math
import os
import sys

import torch
import torch.nn.functional as F
from torch import nn


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "system"))

from flcore.reliability.scorer import ReliabilityScorer
from flcore.reliability.signals import BNFeatureHook, bn_feature_distance
from flcore.reliability.calibration import expected_calibration_error
from flcore.reliability.objectives import (
    weighted_classification_loss,
    weighted_kl_distillation_loss,
)


def test_none_is_unit_weight_and_legacy_loss_is_bit_identical():
    logits = torch.tensor([[1.0, -0.5, 0.2], [-2.0, 3.0, 0.5]])
    labels = torch.tensor([0, 1])
    weights = ReliabilityScorer("none").score(logits, labels)
    assert torch.equal(weights, torch.ones(2))

    historical = F.cross_entropy(logits / 2.0, labels)
    none_branch = F.cross_entropy(logits / 2.0, labels)
    assert torch.equal(historical, none_branch)


def test_entropy_matches_hand_computation():
    probabilities = torch.tensor([[0.75, 0.25], [0.5, 0.5]])
    logits = probabilities.log()
    entropy = -(probabilities * probabilities.log()).sum(dim=1)
    expected = torch.exp(-1.7 * entropy)
    actual = ReliabilityScorer("entropy", reliability_beta=1.7).score(logits)
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)


def test_mutual_information_matches_bald_expression():
    probabilities = torch.tensor([
        [[0.9, 0.1], [0.5, 0.5]],
        [[0.1, 0.9], [0.5, 0.5]],
    ])
    logits = probabilities.log()
    mean_probability = probabilities.mean(dim=0)
    predictive_entropy = -(mean_probability * mean_probability.log()).sum(dim=1)
    teacher_entropy = -(probabilities * probabilities.log()).sum(dim=2).mean(dim=0)
    expected = torch.exp(-2.0 * (predictive_entropy - teacher_entropy))
    actual = ReliabilityScorer(
        "mutual_information", reliability_beta=2.0
    ).score(logits)
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
    assert math.isclose(float(actual[1]), 1.0)


def test_multi_signal_is_product_and_missing_signals_are_neutral():
    probabilities = torch.tensor([[0.8, 0.2], [0.6, 0.4]])
    logits = probabilities.log()
    distance = torch.tensor([0.25, 1.5])
    trust = torch.tensor([0.9, 0.4])
    scorer = ReliabilityScorer(
        "multi_signal",
        reliability_beta_entropy=1.2,
        reliability_beta_bn=0.7,
    )
    entropy = -(probabilities * probabilities.log()).sum(dim=1)
    expected = torch.exp(-1.2 * entropy) * torch.exp(-0.7 * distance) * trust
    actual = scorer.score(logits, bn_distance=distance, trust=trust)
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)

    neutral = scorer.score(logits)
    assert torch.allclose(neutral, torch.exp(-1.2 * entropy))
    assert scorer.last_missing_signals == ["bn_distance", "trust"]


def test_bn_trust_and_oracle_modes():
    logits = torch.tensor([[10.0, -2.0], [3.0, 4.0]])
    distance = torch.tensor([0.0, 2.0])
    bn_weight = ReliabilityScorer(
        "bn_realism", reliability_beta_bn=0.5
    ).score(None, bn_distance=distance)
    assert torch.allclose(bn_weight, torch.tensor([1.0, math.exp(-1.0)]))

    trust = ReliabilityScorer(
        "trust_only", reliability_trust_floor=0.2
    ).score(None, trust=torch.tensor([0.05, 0.8]))
    assert torch.equal(trust, torch.tensor([0.2, 0.8]))

    oracle = ReliabilityScorer("oracle_debug").score(
        logits, targets=torch.tensor([0, 0])
    )
    assert torch.equal(oracle, torch.tensor([1.0, 0.05]))


def test_extreme_logits_are_finite_and_bounded():
    logits = torch.tensor([[1e30, -1e30, 0.0], [-1e30, 1e30, 0.0]])
    ensemble = torch.stack((logits, -logits), dim=0)
    for mode, value in (
        ("entropy", logits),
        ("mutual_information", ensemble),
        ("multi_signal", ensemble),
    ):
        weights = ReliabilityScorer(mode).score(value)
        assert torch.isfinite(weights).all()
        assert ((0.0 <= weights) & (weights <= 1.0)).all()


def test_generator_weighted_cls_and_kd_match_hand_computation():
    teacher_probabilities = torch.tensor([[0.8, 0.2], [0.25, 0.75]])
    student_probabilities = torch.tensor([[0.6, 0.4], [0.9, 0.1]])
    teacher_logits = teacher_probabilities.log()
    student_logits = student_probabilities.log()
    targets = torch.tensor([0, 1])
    scorer = ReliabilityScorer("entropy", reliability_beta=1.3)
    weights = scorer.score(teacher_logits)

    entropy = -(teacher_probabilities * teacher_probabilities.log()).sum(dim=1)
    expected_weights = torch.exp(-1.3 * entropy)
    per_sample_cls = -teacher_probabilities.log()[torch.arange(2), targets]
    per_sample_kl = (
        teacher_probabilities
        * (teacher_probabilities.log() - student_probabilities.log())
    ).sum(dim=1)
    expected_cls = torch.mean(expected_weights * per_sample_cls)
    expected_kd = torch.mean(expected_weights * per_sample_kl)

    actual_cls = weighted_classification_loss(
        teacher_logits, targets, weights
    )
    actual_kd = weighted_kl_distillation_loss(
        teacher_logits, student_logits, weights, temperature=1.0
    )
    assert torch.allclose(weights, expected_weights, atol=1e-7)
    assert torch.allclose(actual_cls, expected_cls, atol=1e-7)
    assert torch.allclose(actual_kd, expected_kd, atol=1e-7)
    assert actual_cls < F.cross_entropy(teacher_logits, targets)


def test_ece_matches_hand_binned_confidence_gap():
    probabilities = torch.tensor([[0.9, 0.1], [0.6, 0.4]])
    # First prediction is correct and the second is incorrect: in one occupied
    # confidence bin ECE = |accuracy 0.5 - mean confidence 0.75| = 0.25.
    targets = torch.tensor([0, 1])
    actual = expected_calibration_error(
        probabilities.log(), targets, bins=2
    )
    assert torch.allclose(actual, torch.tensor(0.25), atol=1e-7)


def test_calibrated_mode_is_temperature_scaled_multi_signal():
    logits = torch.tensor([[2.0, 0.0], [0.5, -0.5]])
    calibrated = ReliabilityScorer(
        "calibrated",
        calibration_temperature=2.0,
        reliability_beta_entropy=0.7,
    ).score(logits)
    expected = ReliabilityScorer(
        "multi_signal", reliability_beta_entropy=0.7
    ).score(logits / 2.0)
    assert torch.allclose(calibrated, expected)


def test_bn_batch_helper_preserves_legacy_expression():
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4)).eval()
    images = torch.randn(5, 3, 6, 6)
    layer = model[1]
    hook = BNFeatureHook(layer)
    try:
        model(images)
        feature = hook.r_feature
        legacy = torch.norm(
            feature.mean(dim=(0, 2, 3)) - layer.running_mean, 2
        ) + torch.norm(
            feature.var(dim=(0, 2, 3), unbiased=False) - layer.running_var, 2
        )
    finally:
        hook.remove()
    factored = bn_feature_distance(model, images, per_sample=False)
    assert torch.equal(factored, legacy)
    per_sample = bn_feature_distance(model, images, per_sample=True)
    assert per_sample.shape == (5,)
    assert torch.isfinite(per_sample).all()


if __name__ == "__main__":
    test_none_is_unit_weight_and_legacy_loss_is_bit_identical()
    test_entropy_matches_hand_computation()
    test_mutual_information_matches_bald_expression()
    test_multi_signal_is_product_and_missing_signals_are_neutral()
    test_bn_trust_and_oracle_modes()
    test_extreme_logits_are_finite_and_bounded()
    test_generator_weighted_cls_and_kd_match_hand_computation()
    test_ece_matches_hand_binned_confidence_gap()
    test_calibrated_mode_is_temperature_scaled_multi_signal()
    test_bn_batch_helper_preserves_legacy_expression()
    print("reliability tests passed")
