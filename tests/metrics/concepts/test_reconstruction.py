"""Reconstruction formulas, oracles, streaming and validation."""

import numpy as np
import pytest
from sklearn.metrics import r2_score

from xplique.metrics.concepts import ReconstructionError, ReconstructionR2

from ._utils import assert_report_equal


@pytest.mark.parametrize("batch_size", [None, 1, 4, 100])
@pytest.mark.parametrize("centering", ["feature", "global"])
def test_reconstruction_oracles_and_streaming(batch_size, centering):
    rng = np.random.default_rng(23)
    x = rng.normal(size=(3, 3, 5)) + np.arange(5) * 30
    y = x + rng.normal(size=x.shape)
    metric = ReconstructionR2(r2_centering=centering, batch_size=batch_size)
    flat = x.reshape(-1, 5)
    expected = (
        r2_score(flat, y.reshape(-1, 5), multioutput="variance_weighted")
        if centering == "feature"
        else r2_score(x.ravel(), y.ravel())
    )
    assert metric(x, y) == pytest.approx(expected)
    assert metric.detailed_evaluate(x, y)["r2"] == metric.evaluate(x, y)
    assert metric.evaluate_batches(iter([(x[:1], y[:1]), (x[1:], y[1:])])) == pytest.approx(
        expected
    )
    assert_report_equal(
        metric.detailed_evaluate(x, y),
        ReconstructionR2(r2_centering=centering).detailed_evaluate(x, y),
    )


def test_r2_constants_offsets_and_negative_scores():
    x = np.array([[1.0, 100.0], [3.0, 100.0]])
    y = np.array([[1.0, 101.0], [3.0, 101.0]])
    assert ReconstructionR2()(x, y) == 0
    assert r2_score(x, y, multioutput="variance_weighted") == 1
    assert ReconstructionR2(r2_centering="global")(x, y) > 0.99
    assert ReconstructionR2()(x, y + 100) < 0
    for shape in [(1, 3), (3, 2)]:
        constant = np.full(shape, 0.1)
        assert np.isnan(ReconstructionR2()(constant, constant))
        assert ReconstructionR2(force_finite=True)(constant, constant) == 1
        assert ReconstructionR2(force_finite=True)(constant, constant + 1) == 0
    offset = 1e10 + np.arange(40).reshape(20, 2) * 0.25
    assert ReconstructionR2(batch_size=1)(offset, offset + 0.125) == pytest.approx(
        ReconstructionR2()(offset, offset + 0.125), abs=1e-12
    )


@pytest.mark.parametrize("error", ["mse", "rmse", "mae", "relative_l2"])
@pytest.mark.parametrize("batch_size", [None, 1, 4, 100])
def test_error_reductions_and_vectors(error, batch_size):
    x = np.arange(30.0).reshape(2, 3, 5)
    y = x + np.arange(6).reshape(2, 3, 1)
    metric = ReconstructionError(error=error, batch_size=batch_size)
    report = metric.detailed_evaluate(x, y)
    key = "relative_l2_error" if error == "relative_l2" else error
    assert metric(x, y) == report[key]
    assert metric.evaluate_batches(iter([(x[:1], y[:1]), (x[1:], y[1:])])) == pytest.approx(
        report[key]
    )
    assert_report_equal(report, ReconstructionError(error=error).detailed_evaluate(x, y))
    assert report["n_elements"] == 30
    assert report["n_observations"] == 6
    assert report["observation_mse_std"] == pytest.approx(np.std(np.arange(6.0) ** 2))
    for name in ["mse", "rmse", "mae", "relative_l2_error"]:
        assert report[f"observation_{name}"].shape == (6,)
    assert report["rmse"] != np.mean(report["observation_rmse"])
    zero = np.zeros((2, 3))
    relative = ReconstructionError(error="relative_l2")
    assert relative(zero, zero) == 0
    assert relative(zero, zero + 1) == np.inf


@pytest.mark.parametrize("factory", [ReconstructionR2, ReconstructionError])
def test_reconstruction_validation_and_recovery(factory):
    metric = factory()
    x = np.arange(12.0).reshape(4, 3)
    for bad in [
        np.ones(3),
        np.empty((0, 3)),
        np.full((4, 3), np.nan),
        np.ones((4, 3), dtype=complex),
        np.full((4, 3), "x", dtype=object),
    ]:
        with pytest.raises(ValueError):
            metric(x, bad)
    with pytest.raises(ValueError):
        metric(x.reshape(2, 2, 3), x)
    with pytest.raises(ValueError):
        metric.evaluate_batches([])
    with pytest.raises(ValueError):
        metric.evaluate_batches([(x, x), (np.ones((2, 2)), np.ones((2, 2)))])
    assert metric(x, x) == (1 if factory is ReconstructionR2 else 0)
    assert metric(np.eye(2), np.eye(2)) == (1 if factory is ReconstructionR2 else 0)


@pytest.mark.parametrize("factory", [ReconstructionR2, ReconstructionError])
def test_reconstruction_batch_pair_format(factory):
    metric = factory()
    x = np.arange(30.0).reshape(2, 5, 3)
    y = x + 1
    for batches in [(x, y), [x], [(x,)], [(x, y, x)], [None], ["xy"], [iter((x, y))]]:
        with pytest.raises(ValueError, match="iterable of pairs"):
            metric.evaluate_batches(batches)
    assert metric.evaluate_batches([[x, y]]) == metric(x, y)
