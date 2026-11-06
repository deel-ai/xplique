"""Concept-bank geometry, Babel and observed-support scientific contracts."""

import numpy as np
import pytest

from xplique.metrics.concepts import (
    concept_bank_babel_curve,
    concept_bank_babel_score,
    concept_bank_geometry_metrics,
)


def test_geometry_and_babel_analytical(monkeypatch):
    bank = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    curve = concept_bank_babel_curve(bank)
    np.testing.assert_allclose(curve["babel_curve"], [0, 1 / np.sqrt(2), np.sqrt(2)])
    for order in range(3):
        assert concept_bank_babel_score(bank, order=order) == pytest.approx(
            curve["babel_curve"][order]
        )
    assert concept_bank_babel_curve(bank, max_order=0)["mutual_coherence"] > 0
    report = concept_bank_geometry_metrics(bank)
    assert report["welch_bound"] == 0.5
    assert np.isfinite(report["condition_number"])
    flipped = bank * np.array([[-2.0], [4.0], [0.01]])
    for key in [
        "stable_rank",
        "energy_rank",
        "energy_entropy_rank",
        "condition_number",
        "mutual_coherence",
    ]:
        assert concept_bank_geometry_metrics(flipped)[key] == pytest.approx(report[key])
    np.testing.assert_allclose(
        concept_bank_babel_curve(flipped)["babel_curve"], curve["babel_curve"]
    )
    for spectrum in ["unit", "raw"]:
        geometry = concept_bank_geometry_metrics(bank, spectrum=spectrum)
        matrix = bank / np.linalg.norm(bank, axis=1)[:, None] if spectrum == "unit" else bank
        singular = np.linalg.svd(matrix, compute_uv=False)
        assert geometry["stable_rank"] == pytest.approx(np.sum((singular / singular[0]) ** 2))

    def no_svd(*args, **kwargs):
        raise AssertionError("Babel must not perform an SVD")

    monkeypatch.setattr(np.linalg, "svd", no_svd)
    concept_bank_babel_score(bank, order=1)
    concept_bank_babel_curve(bank)


def test_geometry_boundaries_and_extreme_scaling():
    for scale in [1e-250, 1.0, 1e250]:
        report = concept_bank_geometry_metrics(np.eye(3) * scale, energy_fraction=1.0)
        assert report["energy_rank"] == 3
        assert report["stable_rank"] == pytest.approx(3)
        assert report["energy_entropy_rank"] == pytest.approx(3)
    one = np.ones((1, 3))
    assert concept_bank_geometry_metrics(one)["mutual_coherence"] == 0
    assert concept_bank_babel_score(one, order=0) == 0
    for bad in [1, -1, 0.5, True]:
        with pytest.raises(ValueError):
            concept_bank_babel_score(one, order=bad)
    parallel = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]])
    report = concept_bank_geometry_metrics(parallel)
    assert report["positive_duplicate_count"] == 1
    assert report["antipodal_count"] == 2
    assert report["condition_number"] == np.inf
    assert (
        concept_bank_geometry_metrics(parallel, duplicate_threshold=1.0)["positive_duplicate_count"]
        == 0
    )
    for settings in [
        dict(spectrum="bad"),
        dict(energy_fraction=0),
        dict(rank_rtol=-1),
        dict(duplicate_threshold=np.nan),
    ]:
        with pytest.raises(ValueError):
            concept_bank_geometry_metrics(one, **settings)


@pytest.mark.parametrize("setting", ["energy_fraction", "duplicate_threshold", "rank_rtol"])
@pytest.mark.parametrize("value", [True, np.bool_(False)])
def test_boolean_geometry_settings_rejected(setting, value):
    with pytest.raises(ValueError):
        concept_bank_geometry_metrics(np.eye(2), **{setting: value})


@pytest.mark.parametrize("value", [np.float32(0.5), np.float64(0.5), np.int64(0)])
def test_numpy_geometry_settings_accepted(value):
    fraction = value if value > 0 else np.int64(1)
    report = concept_bank_geometry_metrics(
        np.eye(2), energy_fraction=fraction, duplicate_threshold=value, rank_rtol=value
    )
    for setting in ["energy_fraction", "duplicate_threshold", "rank_rtol"]:
        assert type(report[setting]) is float


@pytest.mark.parametrize("spectrum", ["unit", "raw"])
def test_random_rank_deficient_energy_endpoint(spectrum):
    rng = np.random.default_rng(42)
    shapes = [(1, 3, 5), (2, 7, 9), (3, 8, 5), (4, 6, 10)]
    for index in range(200):
        rank, k, f = shapes[index % len(shapes)]
        bank = rng.normal(size=(k, rank)) @ rng.normal(size=(rank, f))
        report = concept_bank_geometry_metrics(bank, spectrum=spectrum, energy_fraction=1.0)
        assert report["energy_rank"] == rank


@pytest.mark.parametrize("spectrum", ["unit", "raw"])
def test_energy_rank_endpoints_and_tolerance(spectrum):
    bank = np.array([[1.0, 0.0], [1.0, 1e-10]])
    assert (
        concept_bank_geometry_metrics(bank, spectrum=spectrum, energy_fraction=1.0)["energy_rank"]
        == 2
    )
    assert (
        concept_bank_geometry_metrics(bank, spectrum=spectrum, energy_fraction=1e-12)["energy_rank"]
        == 1
    )
    report = concept_bank_geometry_metrics(
        bank, spectrum=spectrum, energy_fraction=1.0, rank_rtol=1e-8
    )
    assert report["energy_rank"] == 1
    default = concept_bank_geometry_metrics(bank, spectrum=spectrum)
    for key in ["stable_rank", "energy_entropy_rank"]:
        assert report[key] == default[key]
