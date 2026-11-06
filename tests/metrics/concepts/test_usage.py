"""Coefficient usage moments, activity invariance and sparsity."""

import numpy as np
import pytest

from xplique.metrics.concepts import ConceptSparsity, ConceptUsage

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
