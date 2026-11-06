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
    assert details.matched_pairs == 2
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
