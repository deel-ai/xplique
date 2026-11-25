"""Coefficient usage moments, activity invariance and sparsity."""

import numpy as np
import pytest

from xplique.metrics.concepts import ConceptSparsity, ConceptUsage, HoyerSparsity

from ._utils import assert_report_equal


@pytest.mark.parametrize("batch_size", [None, 1, 2, 100])
def test_usage_streaming_and_sparsity(batch_size):
    coefficients = np.array([[1.0, -2.0], [3.0, 0.0], [0.0, 0.0]])
    bank = np.array([[3.0, 4.0], [0.0, 2.0]])
    usage = ConceptUsage(batch_size=batch_size)
    report = usage(coefficients, bank)
    assert_report_equal(
        report,
        usage.evaluate_batches(iter([coefficients[:1], coefficients[1:]]), concept_bank=bank),
    )
    np.testing.assert_allclose(report["component_energy"], [250, 16])
    sparsity = ConceptSparsity(batch_size=batch_size)
    assert sparsity(coefficients, bank) == report["sparsity"]
    assert sparsity.detailed_evaluate(coefficients, bank)["sparsity"] == report["sparsity"]


def test_usage_invariance_degeneracies_and_reuse():
    u = np.array([[1.0, -2.0], [3.0, 0.0]])
    bank = np.array([[3.0, 4.0], [0.0, 2.0]])
    scale = np.array([-5.0, 0.25])
    usage = ConceptUsage(active_threshold=1.0)
    report = usage(u, bank)
    scaled = usage(u / scale, bank * scale[:, None])
    for key in [
        "effective_usage",
        "component_energy",
        "component_energy_share",
        "active_fraction",
        "sparsity",
    ]:
        np.testing.assert_allclose(report[key], scaled[key])
    assert not np.allclose(
        ConceptUsage(activity_basis="coefficient", active_threshold=1.0)(u, bank)[
            "active_fraction"
        ],
        ConceptUsage(activity_basis="coefficient", active_threshold=1.0)(
            u / scale, bank * scale[:, None]
        )["active_fraction"],
    )
    zero = usage(u, np.zeros_like(bank))
    assert np.isnan(zero["component_energy_share"]).all()
    assert zero["sparsity"] == 1
    assert report["component_energy"].sum() != np.square(u @ bank).sum()
    assert usage(u[:1], bank)["n_observations"] == 1
    assert usage(np.ones((2, 3)), np.eye(3))["n_concepts"] == 3
    with pytest.raises(ValueError):
        usage.evaluate_batches([u, np.ones((1, 3))], concept_bank=bank)
    assert usage(u, bank)["n_observations"] == 2
    assert ConceptSparsity()(u, bank) == usage(u, bank)["sparsity"]


def test_usage_stable_moments_and_exact_threshold_boundary():
    coefficients = np.array([[1.0, -2.0], [3.0, 0.0]])
    report = ConceptUsage(activity_basis="coefficient", active_threshold=1.0)(
        coefficients, np.eye(2)
    )
    np.testing.assert_allclose(report["active_fraction"], [0.5, 0.5])
    offset = coefficients + 1e10
    np.testing.assert_allclose(
        ConceptUsage(batch_size=1)(offset, np.eye(2))["std_coefficient"], coefficients.std(axis=0)
    )
    assert ConceptUsage(batch_size=1)(offset.reshape(1, 2, 2), np.eye(2))["n_observations"] == 2

    high_offset = float(2**53) + np.array([[0.0], [2.0], [4.0], [6.0]])
    np.testing.assert_allclose(
        ConceptUsage(batch_size=1)(high_offset, np.eye(1))["std_coefficient"],
        [np.sqrt(5.0)],
        rtol=1e-12,
    )
    large_energy = ConceptUsage()(np.array([[1e154, 1e154]]), np.eye(2))
    np.testing.assert_allclose(large_energy["component_energy_share"], [0.5, 0.5])


@pytest.mark.parametrize("batch_size", [None, 1, 3, 100])
def test_hoyer_exact_rows_streaming_and_invariance(batch_size):
    u = np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 0.001], [0.0, 0.0]])
    bank = np.eye(2)
    metric = HoyerSparsity(batch_size=batch_size)
    report = metric.detailed_evaluate(u, bank)
    rows = report["observation_hoyer_sparsity"]
    np.testing.assert_allclose(rows[[0, 1, 3]], [0, 1, np.nan], atol=1e-15)
    assert 0.99 < rows[2] < 1
    assert report["n_zero_contribution_observations"] == 1
    assert report["n_scorable_observations"] == 3
    assert report["hoyer_sparsity"] == pytest.approx(rows[:3].mean())
    assert metric.evaluate_batches(iter([u[:1], u[1:]]), concept_bank=bank) == pytest.approx(
        report["hoyer_sparsity"]
    )
    rescale = np.array([-4.0, 1e-100])
    scaled = metric.detailed_evaluate(u / rescale, bank * rescale[:, None])
    np.testing.assert_allclose(scaled["observation_hoyer_sparsity"], rows, atol=1e-15)
    assert metric(u[::-1], bank) == pytest.approx(report["hoyer_sparsity"])


def test_hoyer_degenerate_banks_and_numeric_range():
    metric = HoyerSparsity()
    one = metric.detailed_evaluate(np.array([[1.0], [0.0]]), np.ones((1, 2)))
    assert np.isnan(one["hoyer_sparsity"])
    assert one["n_concepts"] == 1
    assert one["n_scorable_observations"] == 0
    assert one["n_zero_contribution_observations"] == 1
    assert np.isnan(metric(np.ones((1, 2)), np.zeros((2, 1))))
    assert metric(np.ones((1, 2)), np.array([[0.0], [2.0]])) == 1
    assert metric(np.full((1, 2), 1e308), np.full((2, 4), 1e308)) == pytest.approx(0)
    assert metric(np.full((1, 2), 1e-310), np.full((2, 4), 1e-310)) == pytest.approx(0)
    with pytest.raises(ValueError, match="counts"):
        metric(np.ones((1, 3)), np.eye(2))
    with pytest.raises(ValueError, match="observations"):
        metric.evaluate_batches(iter(()), concept_bank=np.eye(2))


def test_hoyer_multi_concept_analytical_oracle_and_support_contrast():
    # K = 4 with non-unit atom norms so contribution magnitudes are explicitly tested.
    bank = np.diag([1.0, 2.0, 3.0, 4.0])
    codes = np.array(
        [
            [1.0, 0.5, 0.0, 0.0],  # contributions [1, 1, 0, 0] -> (2 - sqrt(2)) / (2 - 1)
            [1.0, 0.5, 1.0 / 3.0, 0.25],  # contributions [1, 1, 1, 1] -> 0
            [4.0, 0.5, 1.0 / 3.0, 0.25],  # contributions [4, 1, 1, 1], same full support
        ]
    )
    support_sizes = ConceptSparsity().detailed_evaluate(codes, bank)["observation_support_sizes"]
    assert support_sizes.tolist() == [2, 4, 4]

    report = HoyerSparsity(batch_size=2).detailed_evaluate(codes, bank)
    expected_row2 = (2.0 - 7.0 / np.sqrt(19.0)) / (2.0 - 1.0)
    expected = np.array([2.0 - np.sqrt(2.0), 0.0, expected_row2])
    assert report["n_concepts"] == 4
    np.testing.assert_allclose(report["observation_hoyer_sparsity"], expected, atol=1e-14)
    assert report["hoyer_sparsity"] == pytest.approx(expected.mean(), abs=1e-14)
    # Rows 1 and 2 have identical full support (4), but Hoyer distinguishes concentration.
    assert report["observation_hoyer_sparsity"][2] > report["observation_hoyer_sparsity"][1]
