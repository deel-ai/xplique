"""Concept alignment objectives, assignments and comparison helpers."""

from dataclasses import replace

import numpy as np
import pytest

from xplique.metrics.concepts import ConceptAlignment, MatchedCodeConsistency, compare_concept_banks


@pytest.mark.parametrize("mode", ["cosine", "absolute_cosine", "dot"])
def test_alignment_modes_coverage_and_helper(mode):
    bank = np.eye(3)
    compared = bank[[2, 0]] * np.array([[2.0], [-3.0]])
    metric = ConceptAlignment(similarity=mode)
    details = metric.detailed_evaluate(bank, compared)
    assert details.n_reference_concepts == 3
    assert details.n_compared_concepts == 2
    assert details.n_matched_pairs == 2
    assert details.coverage == pytest.approx(2 / 3)
    assert metric(bank, compared) == details.coverage_weighted_similarity
    assert details.coverage_weighted_similarity == pytest.approx(
        details.mean_similarity * details.coverage
    )
    if mode == "absolute_cosine":
        assert details.mean_similarity == 1
        assert sorted(details.matched_polarities) == [-1, 1]
    banks = {"a": bank, "b": compared, "c": bank[::-1]}
    assert len(compare_concept_banks(banks, similarity=mode)) == 3
    for (a, b), result in compare_concept_banks(banks, similarity=mode, reference_key="b").items():
        assert result.coverage_weighted_similarity == metric(banks[a], banks[b])


def test_alignment_ties_rescaling_and_invalid_inputs():
    bank = np.eye(2)
    assert ConceptAlignment()(bank, bank * 3) == 1
    assert ConceptAlignment(similarity="dot")(bank, bank * 3) == 3
    assert ConceptAlignment()(np.ones((2, 2)), np.ones((3, 2))) == pytest.approx(2 / 3)
    for bad in [np.zeros((2, 2)), np.ones((2, 3)), np.full((2, 2), np.inf)]:
        with pytest.raises(ValueError):
            ConceptAlignment()(bank, bad)
    with pytest.raises(ValueError):
        ConceptAlignment(similarity="unknown")
    with pytest.raises(ValueError):
        compare_concept_banks({"a": bank})
    with pytest.raises(ValueError):
        compare_concept_banks({"a": bank, "b": bank}, reference_key="missing")
    with pytest.raises(ValueError, match="mapping"):
        compare_concept_banks([bank, bank])
    with pytest.raises(ValueError, match="overflowed"):
        ConceptAlignment(similarity="dot")(np.eye(2) * 1e200, np.eye(2) * 1e200)


def test_alignment_rectangular_oracle_symmetry_and_polarities():
    from itertools import permutations

    reference = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    compared = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    ref_u = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    cmp_u = compared / np.linalg.norm(compared, axis=1, keepdims=True)
    cos = ref_u @ cmp_u.T

    for mode, matrix in [
        ("cosine", cos),
        ("absolute_cosine", np.abs(cos)),
        ("dot", reference @ compared.T),
    ]:
        best = max(
            sum(matrix[r, c] for r, c in zip(rows, range(len(compared))))
            for rows in permutations(range(len(reference)), len(compared))
        )
        expected = best / max(len(reference), len(compared))
        metric = ConceptAlignment(similarity=mode)
        assert metric(reference, compared) == pytest.approx(expected)
        assert metric(compared, reference) == pytest.approx(expected)

    # Zero cosine (including -0.0) maps to polarity +1 by convention.
    orthogonal = ConceptAlignment(similarity="cosine").detailed_evaluate(
        np.array([[1.0, 0.0]]), np.array([[0.0, -1.0]])
    )
    assert orthogonal.matched_polarities.tolist() == [1]

    # Negative similarities with unequal bank sizes pad unmatched concepts with 0.0.
    assert ConceptAlignment(similarity="cosine")(
        np.array([[1.0, 0.0]]), np.array([[-1.0, 0.0], [-1.0, 0.0]])
    ) == pytest.approx(-0.5)


@pytest.mark.parametrize("mode", ["cosine", "absolute_cosine", "dot"])
def test_random_alignment_exact_scalar_recovery(mode):
    rng = np.random.default_rng(42)
    metric = ConceptAlignment(similarity=mode)
    for n_reference, n_compared in [(7, 3), (3, 7), (4, 4)]:
        for _ in range(20):
            reference = rng.normal(size=(n_reference, 9))
            compared = rng.normal(size=(n_compared, 9))
            details = metric.detailed_evaluate(reference, compared)
            assert metric(reference, compared) == details.coverage_weighted_similarity


def test_matching_result_identity_equality():
    metric = ConceptAlignment()
    first = metric.detailed_evaluate(np.eye(2), np.eye(2))
    second = metric.detailed_evaluate(np.eye(2), np.eye(2))
    assert first == first
    assert first != second


def test_matched_code_correlations_are_distinct_from_bank_alignment():
    bank = np.eye(4)
    matching = ConceptAlignment().detailed_evaluate(bank, bank)
    left = np.tile(np.array([-1, -1, 1, 1])[:, None], (1, 4))
    right = np.column_stack((left[:, 0], -left[:, 1], [-1, 1, -1, 1], np.zeros(4)))
    metric = MatchedCodeConsistency()
    report = metric.detailed_evaluate(left, right, matching=matching)
    np.testing.assert_allclose(report["matched_correlations"], [1, -1, 0, np.nan])
    assert report["n_scorable_pairs"] == 3
    assert report["n_matched_pairs"] == 4
    assert matching.coverage_weighted_similarity == 1
    assert metric(left, right, matching=matching) == pytest.approx(0)
    assert np.isnan(metric(left[:1], right[:1], matching=matching))
    assert np.isnan(metric(np.zeros_like(left), right, matching=matching))


def test_matched_code_polarity_and_rectangular_coverage():
    reference = np.eye(3)[:2]
    compared = np.eye(3)[[1, 0, 2]] * np.array([[-2], [3], [1]])
    matching = ConceptAlignment(similarity="absolute_cosine").detailed_evaluate(reference, compared)
    u = np.array([[0.0, 1.0], [1.0, 3.0], [3.0, 0.0]])
    v = np.column_stack((-u[:, 1] / 2, u[:, 0] / 3, np.zeros(len(u))))
    report = MatchedCodeConsistency().detailed_evaluate(u, v, matching=matching)
    assert report["mean_correlation"] == pytest.approx(1)
    assert report["coverage"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("mode", ["cosine", "dot"])
def test_matched_code_signed_modes_do_not_correct_polarity(mode):
    matching = ConceptAlignment(similarity=mode).detailed_evaluate([[1.0]], [[-2.0]])
    u = np.arange(3.0)[:, None]
    assert MatchedCodeConsistency()(u, -u / 2, matching=matching) == pytest.approx(-1)


def test_matched_code_streaming_offsets_strides_and_validation():
    matching = ConceptAlignment().detailed_evaluate(np.eye(2), np.eye(2))
    pattern = np.arange(60.0, dtype=np.float64).reshape(5, 6, 2)[::2, ::2]
    u = float(2**53) + 2 * pattern
    v = np.stack((pattern[..., 0] / 4, -pattern[..., 1] * 3), axis=-1)
    ordinary = MatchedCodeConsistency().detailed_evaluate(u, v, matching=matching)
    np.testing.assert_allclose(ordinary["matched_correlations"], [1, -1], atol=1e-14)
    for size in (1, 4, 100):
        chunked = MatchedCodeConsistency(batch_size=size)
        details = chunked.detailed_evaluate(u[:, ::-1], v[:, ::-1], matching=matching)
        np.testing.assert_allclose(details["matched_correlations"], [1, -1], atol=1e-14)
        assert chunked.evaluate_batches(
            iter([(u[:2], v[:2]), (u[2:], v[2:])]), matching=matching
        ) == pytest.approx(ordinary["mean_correlation"], abs=1e-14)

    large = np.array([[0.0, 1e-310], [1e308, 2e-310], [-1e308, 3e-310]])
    for size in (None, 1):
        np.testing.assert_allclose(
            MatchedCodeConsistency(batch_size=size).detailed_evaluate(
                large, large, matching=matching
            )["matched_correlations"],
            [1, 1],
        )

    metric = MatchedCodeConsistency(batch_size=2)
    with pytest.raises(ValueError, match="observation shapes"):
        metric(u, v.reshape(-1, 2), matching=matching)
    with pytest.raises(ValueError, match="widths"):
        metric(u[..., :1], v, matching=matching)
    with pytest.raises(ValueError, match="HungarianMatchingResult"):
        metric(u, v, matching="not-a-matching")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="counts"):
        metric(u, v, matching=replace(matching, n_matched_pairs=1))
    with pytest.raises(ValueError, match="counts"):
        metric(u, v, matching=replace(matching, n_reference_concepts=True))
    for bad_indices in ([0, 0], [-1, 0], [0, 2], np.array([0.0, 1.0])):
        with pytest.raises(ValueError, match="indices"):
            metric(u, v, matching=replace(matching, assignment_compared_indices=bad_indices))
    with pytest.raises(ValueError, match="polarity"):
        metric(u, v, matching=replace(matching, matched_polarities=[1, 0]))
    with pytest.raises(ValueError, match="similarity"):
        metric(u, v, matching=replace(matching, similarity="unknown"))
    with pytest.raises(ValueError, match="pair"):
        metric.evaluate_batches((u, v), matching=matching)
    with pytest.raises(ValueError, match="observations"):
        metric.evaluate_batches(iter(()), matching=matching)

    # NaN in an unmatched column must still be rejected.
    rect_matching = ConceptAlignment().detailed_evaluate(np.eye(3)[:2], np.eye(3))
    v_with_nan = np.zeros((3, 3))
    v_with_nan[0, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        metric(np.ones((3, 2)), v_with_nan, matching=rect_matching)

    # NumPy integer counts in matching still yield a Python float coverage.
    np_count_matching = replace(
        rect_matching,
        n_reference_concepts=np.int64(2),
        n_compared_concepts=np.int64(3),
        n_matched_pairs=np.int64(2),
    )
    rep = metric.detailed_evaluate(np.ones((3, 2)), np.ones((3, 3)), matching=np_count_matching)
    assert type(rep["coverage"]) is float
    assert rep["coverage"] == pytest.approx(2 / 3)


def test_matched_code_nondegenerate_oracle_and_cross_chunk_scale_jumps():
    rng = np.random.default_rng(7)
    scales = np.array([1.0, 1e10, 1e-10, 1e200, 1e-200])
    u = rng.normal(size=(30, 5)) * scales + np.array([0.0, 1e12, 0.0, 0.0, 0.0])
    v = 0.65 * u + 0.35 * rng.normal(size=(30, 5)) * scales
    # Force a 400-order-of-magnitude scale jump between early and late chunks on column 3.
    u[:6, 3] = rng.normal(size=6) * 1e-200
    v[:6, 3] = rng.normal(size=6) * 1e-200
    matching = ConceptAlignment().detailed_evaluate(np.eye(5), np.eye(5))

    expected = np.array(
        [
            np.corrcoef(
                u[:, k] / np.max(np.abs(u[:, k])),
                v[:, k] / np.max(np.abs(v[:, k])),
            )[0, 1]
            for k in range(5)
        ]
    )
    assert np.all((np.abs(expected) > 0.1) & (np.abs(expected) < 0.99))

    for size in (None, 1, 4, 13):
        details = MatchedCodeConsistency(batch_size=size).detailed_evaluate(u, v, matching=matching)
        np.testing.assert_allclose(details["matched_correlations"], expected, atol=1e-12)
        assert details["mean_correlation"] == pytest.approx(expected.mean(), abs=1e-12)

    streamed = MatchedCodeConsistency(batch_size=3).evaluate_batches(
        iter([(u[:4], v[:4]), (u[4:11], v[4:11]), (u[11:], v[11:])]),
        matching=matching,
    )
    assert streamed == pytest.approx(expected.mean(), abs=1e-12)
