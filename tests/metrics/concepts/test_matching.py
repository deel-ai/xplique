"""Concept alignment objectives, assignments and comparison helpers."""

import numpy as np
import pytest

from xplique.metrics.concepts import ConceptAlignment, compare_concept_banks


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
