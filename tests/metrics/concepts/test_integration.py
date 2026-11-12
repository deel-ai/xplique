"""Analytical, numerical, and execution contracts for standalone concept metrics."""

import subprocess
import sys

import numpy as np
import pytest

from xplique.metrics.concepts import (
    CoactiveSupportCoherence,
    ConceptAlignment,
    ConceptSparsity,
    ConceptUsage,
    ReconstructionError,
    ReconstructionR2,
    concept_bank_babel_curve,
    concept_bank_babel_score,
    concept_bank_geometry_metrics,
)
from xplique.metrics.concepts._utils import observation_chunks

from ._utils import assert_report_equal


@pytest.mark.parametrize(
    "factory",
    [
        ReconstructionR2,
        ReconstructionError,
        ConceptUsage,
        ConceptSparsity,
        CoactiveSupportCoherence,
    ],
)
@pytest.mark.parametrize("bad", [0, -1, 1.2, True, np.bool_(False), "3"])
def test_invalid_batch_sizes(factory, bad):
    with pytest.raises(ValueError):
        factory(batch_size=bad)


@pytest.mark.parametrize("layout", ["transpose", "slice", "fortran"])
def test_strided_inputs_are_bounded_and_ordered(layout):
    x = np.arange(120.0).reshape(4, 5, 6)
    x = {
        "transpose": x.transpose(1, 0, 2),
        "slice": x[:, ::2, ::-1],
        "fortran": np.asfortranarray(x),
    }[layout]
    chunks = list(observation_chunks(x, 3))
    assert all(len(chunk) <= 3 for chunk in chunks)
    np.testing.assert_array_equal(np.concatenate(chunks), x.reshape(-1, x.shape[-1]))
    for factory in [ReconstructionR2, ReconstructionError]:
        assert_report_equal(
            factory(batch_size=3).detailed_evaluate(x, x + 1),
            factory().detailed_evaluate(x.copy(), x.copy() + 1),
        )
    bank = np.eye(x.shape[-1])
    for factory in [ConceptUsage, ConceptSparsity, CoactiveSupportCoherence]:
        metric = factory(batch_size=3)
        reference = factory()
        if factory is ConceptUsage:
            assert_report_equal(metric(x, bank), reference(x.copy(), bank))
        else:
            assert_report_equal(
                metric.detailed_evaluate(x, bank), reference.detailed_evaluate(x.copy(), bank)
            )


@pytest.mark.parametrize("batch_size", [None, 1, 2, 100])
@pytest.mark.parametrize("basis", ["contribution", "coefficient"])
def test_usage_sparsity_and_coactive(batch_size, basis):
    bank = np.array([[3.0, 4.0], [0.0, 2.0]])
    coefficients = np.array([[1.0, -2.0], [3.0, 0.0], [0.0, 0.0]])
    original = coefficients.copy()
    settings = dict(activity_basis=basis, active_threshold=1.0, batch_size=batch_size)
    usage = ConceptUsage(**settings)
    report = usage(coefficients, bank)
    np.testing.assert_allclose(report["mean_coefficient"], [4 / 3, -2 / 3])
    np.testing.assert_allclose(report["std_coefficient"], coefficients.std(axis=0))
    np.testing.assert_allclose(report["effective_usage"], [20 / 3, 4 / 3])
    np.testing.assert_allclose(report["component_energy"], [250, 16])
    assert report["sparsity"] == pytest.approx(1 / 2 if basis == "contribution" else 2 / 3)
    assert_report_equal(
        report,
        usage.evaluate_batches(iter([coefficients[:1], coefficients[1:]]), concept_bank=bank),
    )
    assert_report_equal(
        report, ConceptUsage(**{**settings, "batch_size": None})(coefficients, bank)
    )
    sparsity = ConceptSparsity(**settings)
    detailed = sparsity.detailed_evaluate(coefficients, bank)
    assert sparsity(coefficients, bank) == detailed["sparsity"] == report["sparsity"]
    assert sparsity.evaluate_batches([coefficients], concept_bank=bank) == report["sparsity"]
    coherence = CoactiveSupportCoherence(**settings)
    details = coherence.detailed_evaluate(coefficients, bank)
    assert coherence(coefficients, bank) == pytest.approx(details["mean_support_babel"])
    np.testing.assert_array_equal(
        detailed["observation_support_sizes"], details["observation_support_sizes"]
    )
    assert_report_equal(
        details,
        CoactiveSupportCoherence(**{**settings, "batch_size": None}).detailed_evaluate(
            coefficients, bank
        ),
    )
    np.testing.assert_array_equal(coefficients, original)
    assert not hasattr(usage, "detailed_evaluate")


def test_factorizer_integration():
    from xplique.concepts.tf.factorizer import TfSklearnNMFFactorizer

    x = np.random.default_rng(12).uniform(0.1, 1.0, (20, 6)).astype(np.float32)
    factorizer = TfSklearnNMFFactorizer(n_components=3, max_iter=30, random_state=3, init="random")
    fit_bank, fit_coefficients = factorizer.fit(x)
    u = factorizer.encode(x)
    bank = factorizer.get_concept_bank()
    y = factorizer.decode(u)
    snapshots = [v.copy() for v in (x, u, bank, y)]
    assert ConceptUsage()(fit_coefficients, fit_bank)["n_concepts"] == 3
    xd, yd = x.astype(float), y.astype(float)
    sse = np.square(xd - yd).sum()
    assert ReconstructionR2()(x, y) == pytest.approx(
        1 - sse / np.square(xd - xd.mean(axis=0)).sum()
    )
    assert ReconstructionError()(x, y) == pytest.approx(sse / x.size)
    assert ConceptSparsity().detailed_evaluate(u, bank)["n_concepts"] == 3
    assert CoactiveSupportCoherence().detailed_evaluate(u, bank)["n_concepts"] == 3
    assert ConceptAlignment()(bank, bank) == pytest.approx(1)
    alignment = ConceptAlignment().detailed_evaluate(bank, bank)
    assert alignment.coverage == 1
    assert alignment.mean_similarity == pytest.approx(1, abs=1e-14)
    second = TfSklearnNMFFactorizer(n_components=3, max_iter=30, random_state=4, init="random")
    other, _ = second.fit(x)
    assert ConceptAlignment().detailed_evaluate(bank, other).n_matched_pairs == 3
    assert concept_bank_geometry_metrics(bank)["n_features"] == 6
    assert concept_bank_babel_curve(bank)["n_concepts"] == 3
    assert np.isfinite(concept_bank_babel_score(bank, order=2))
    for value, snapshot in zip((x, u, bank, y), snapshots):
        np.testing.assert_array_equal(value, snapshot)


@pytest.mark.parametrize("factory", [ConceptUsage, ConceptSparsity, CoactiveSupportCoherence])
def test_coefficient_validation_and_independent_evaluations(factory):
    metric = factory(batch_size=2)
    bank = np.eye(3)
    values = np.ones((5, 3))
    for bad in [
        np.empty((0, 3)),
        np.empty((2, 0)),
        np.ones(3),
        np.ones((2, 2)),
        np.full((5, 3), np.nan),
        np.ones((5, 3), dtype=complex),
        np.ones((5, 3), dtype=object),
        np.ones((5, 3), dtype=bool),
    ]:
        with pytest.raises(ValueError):
            metric(bad, bank)
    for bad_bank in [np.empty((0, 3)), np.ones(3), np.full((3, 3), np.inf)]:
        with pytest.raises(ValueError):
            metric(values, bad_bank)
    before = dict(vars(metric))
    result = metric(values, bank)
    assert vars(metric) == before
    assert metric(np.ones((2, 2)), np.eye(2)) is not None
    again = metric(values, bank)
    if factory is ConceptUsage:
        assert_report_equal(again, result)
    else:
        assert again == result
    if factory is not CoactiveSupportCoherence:
        with pytest.raises(ValueError):
            metric.evaluate_batches(iter([]), concept_bank=bank)


@pytest.mark.parametrize("factory", [ConceptUsage, ConceptSparsity, CoactiveSupportCoherence])
@pytest.mark.parametrize(
    "settings",
    [
        dict(activity_basis="invalid"),
        dict(active_threshold=-1),
        dict(active_threshold=np.nan),
        dict(active_threshold=np.inf),
    ],
)
def test_invalid_activity_configuration(factory, settings):
    with pytest.raises(ValueError):
        factory(**settings)


@pytest.mark.parametrize("factory", [ConceptUsage, ConceptSparsity, CoactiveSupportCoherence])
@pytest.mark.parametrize("threshold", [True, np.bool_(False)])
def test_boolean_activity_threshold_rejected(factory, threshold):
    with pytest.raises(ValueError):
        factory(active_threshold=threshold)


@pytest.mark.parametrize("value", [np.float32(0.5), np.float64(0.5), np.int64(0)])
def test_numpy_numeric_settings_accepted(value):
    for factory in [ConceptUsage, ConceptSparsity, CoactiveSupportCoherence]:
        assert factory(active_threshold=value).active_threshold == float(value)


def test_one_pass_iterables_and_energy_observation_scaling():
    class Once:
        def __init__(self, values):
            self.values = values
            self.iterated = False

        def __iter__(self):
            assert not self.iterated
            self.iterated = True
            yield from self.values

    values = np.arange(20.0).reshape(5, 4)
    for factory in [ReconstructionR2, ReconstructionError]:
        metric = factory(batch_size=1)
        assert metric.evaluate_batches(
            Once([(values[:2], values[:2]), (values[2:], values[2:])])
        ) == metric(values, values)
    usage = ConceptUsage(batch_size=2)
    report = usage.evaluate_batches(Once([values[:1], values[1:]]), concept_bank=np.eye(4))
    doubled = usage(np.concatenate([values, values]), np.eye(4))
    np.testing.assert_allclose(doubled["component_energy"], report["component_energy"] * 2)
    np.testing.assert_allclose(doubled["effective_usage"], report["effective_usage"])


def test_optional_dependencies_not_required():
    """Exercise the normal package import with optional scientific backends blocked."""
    code = """
import builtins
original = builtins.__import__
def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split('.')[0] in ('torch', 'torchmetrics', 'overcomplete'):
        raise ImportError('Optional dependency intentionally blocked')
    return original(name, globals, locals, fromlist, level)
builtins.__import__ = guarded
import numpy as np
from xplique.metrics.concepts import (
    ReconstructionR2, ReconstructionError, ConceptUsage, ConceptSparsity,
    ConceptAlignment, CoactiveSupportCoherence, concept_bank_geometry_metrics,
    concept_bank_babel_score, concept_bank_babel_curve, compare_concept_banks,
)
bank = np.eye(2)
assert ReconstructionR2()(bank, bank) == 1
assert ReconstructionError()(bank, bank) == 0
assert ConceptUsage()(bank, bank)['n_concepts'] == 2
assert ConceptSparsity()(bank, bank) == .5
assert ConceptAlignment()(bank, bank) == 1
assert CoactiveSupportCoherence()(bank, bank) == 0
assert concept_bank_geometry_metrics(bank)['stable_rank'] == 2
assert concept_bank_babel_score(bank, order=1) == 0
assert concept_bank_babel_curve(bank)['babel_score'] == 0
assert len(compare_concept_banks({'a': bank, 'b': bank})) == 1
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stderr


def test_hand_computable_errors_and_no_scalar_detail_collection(monkeypatch):
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([[0.0, 2.0], [5.0, 4.0]])
    expected = dict(mse=1.25, rmse=np.sqrt(1.25), mae=0.75, relative_l2=np.sqrt(5 / 30))
    for error, value in expected.items():
        assert ReconstructionError(error=error)(x, y) == pytest.approx(value)
    details = ReconstructionError().detailed_evaluate(x, y)
    assert details["relative_l2_error"] != np.mean(details["observation_relative_l2_error"])
    zero_rows = ReconstructionError().detailed_evaluate(np.zeros((2, 2)), np.eye(2))
    assert np.isinf(zero_rows["observation_relative_l2_error"]).all()

    def no_concatenate(*args, **kwargs):
        raise AssertionError("Scalar evaluation must not collect observation details")

    monkeypatch.setattr(np, "concatenate", no_concatenate)
    ReconstructionError(batch_size=1)(x, y)
    ReconstructionR2(batch_size=1)(x, y)
    ConceptUsage(batch_size=1)(x, np.eye(2))
    ConceptSparsity(batch_size=1)(x, np.eye(2))
    CoactiveSupportCoherence(batch_size=1)(x, np.eye(2))
