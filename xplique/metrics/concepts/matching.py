"""One-shot matching of complete concept banks in shared feature coordinates."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ._utils import concept_bank_input


@dataclass(eq=False)
class HungarianMatchingResult:
    """Bank-level assignment and matched-pair scientific quantities.

    Results compare by identity. Compare array fields individually with NumPy
    testing functions when checking scientific equivalence.

    Attributes
    ----------
    similarity : str
        cosine, absolute_cosine, or dot objective.
    n_reference_concepts, n_compared_concepts, n_matched_pairs : int
        Bank sizes and min(bank sizes) pair count.
    assignment_reference_indices, assignment_compared_indices : np.ndarray
        Integer arrays of shape (M,) indexing matched rows.
    matched_similarities : np.ndarray
        Shape (M,), dimensionless cosines or scale-sensitive dot products.
    matched_absolute_cosines : np.ndarray
        Shape (M,), absolute directional agreements in [0,1].
    matched_polarities : np.ndarray
        Shape (M,), signs -1/+1; zero cosine arbitrarily uses +1.
    mean_similarity, coverage, coverage_weighted_similarity : float
        Mean objective, M/max(bank sizes), and their product. Signed scores may
        be negative; dot scores are unbounded. No significance test is implied.
    """

    similarity: str
    n_reference_concepts: int
    n_compared_concepts: int
    n_matched_pairs: int
    assignment_reference_indices: np.ndarray
    assignment_compared_indices: np.ndarray
    matched_similarities: np.ndarray
    matched_absolute_cosines: np.ndarray
    matched_polarities: np.ndarray
    mean_similarity: float
    coverage: float
    coverage_weighted_similarity: float


def _matched_summary(similarity, rows, columns):
    """Share matched values and the exact coverage-weighted calculation."""
    matched = similarity[rows, columns]
    coverage = len(rows) / max(similarity.shape)
    return matched, float(coverage), float(matched.mean() * coverage)


class ConceptAlignment:
    """Maximum-sum Hungarian matching, with no stored bank or assignment.

    Parameters
    ----------
    similarity : {'cosine', 'absolute_cosine', 'dot'}
        Signed directional agreement, agreement modulo valid sign symmetry, or
        scale-sensitive dot product. Equal feature counts alone do not establish
        comparable coordinate systems. Zero concept vectors are rejected.
    """

    def __init__(self, *, similarity="cosine"):
        if similarity not in ("cosine", "absolute_cosine", "dot"):
            raise ValueError("Unknown similarity mode.")
        self.similarity = similarity

    def _assignment(self, reference_concept_bank, compared_concept_bank):
        reference, _, reference_unit = concept_bank_input(reference_concept_bank, True)
        compared, _, compared_unit = concept_bank_input(compared_concept_bank, True)
        if reference.shape[1] != compared.shape[1]:
            raise ValueError("Concept banks must share feature dimensions.")
        cosine = np.clip(reference_unit @ compared_unit.T, -1, 1)
        similarity = cosine
        if self.similarity == "absolute_cosine":
            similarity = np.abs(cosine)
        elif self.similarity == "dot":
            similarity = reference @ compared.T
        rows, columns = linear_sum_assignment(similarity, maximize=True)
        return similarity, cosine, rows, columns

    def evaluate(self, reference_concept_bank, compared_concept_bank) -> float:
        """Return bank-level matched mean times coverage, not an observation average.

        See detailed_evaluate for the assignment and coverage components.
        """
        similarity, _, rows, columns = self._assignment(
            reference_concept_bank, compared_concept_bank
        )
        return _matched_summary(similarity, rows, columns)[2]

    def __call__(self, reference_concept_bank, compared_concept_bank) -> float:
        """Delegate to evaluate."""
        return self.evaluate(reference_concept_bank, compared_concept_bank)

    def detailed_evaluate(
        self, reference_concept_bank, compared_concept_bank
    ) -> HungarianMatchingResult:
        """Inspect matched-pair and bank-level quantities.

        Returns
        -------
        HungarianMatchingResult
            coverage_weighted_similarity reproduces evaluate. All attributes,
            shapes and units are listed on HungarianMatchingResult. Nonempty
            nonzero banks always give M pairs; ties need not have unique indices.
            Zero cosine has polarity +1 by convention, not positive association.
        """
        similarity, cosine, rows, columns = self._assignment(
            reference_concept_bank, compared_concept_bank
        )
        matched, coverage, coverage_weighted = _matched_summary(similarity, rows, columns)
        matched_cosine = cosine[rows, columns]
        return HungarianMatchingResult(
            self.similarity,
            similarity.shape[0],
            similarity.shape[1],
            len(rows),
            rows,
            columns,
            matched,
            np.abs(matched_cosine),
            np.where(matched_cosine < 0, -1, 1),
            float(matched.mean()),
            float(coverage),
            coverage_weighted,
        )


def compare_concept_banks(banks, *, similarity="cosine", reference_key=None) -> dict:
    """Compare all bank pairs, or a reference star, using ConceptAlignment.

    Parameters
    ----------
    banks : dict
        At least two run-key to concept-bank entries in comparable coordinates.
    similarity : str
        ConceptAlignment objective.
    reference_key : optional
        Existing key for reference-star comparisons; None selects all pairs.

    Returns
    -------
    dict
        (reference run, compared run) keys to HungarianMatchingResult values.
        These are not independent replicates or a statistical stability estimator.
    """
    if len(banks) < 2:
        raise ValueError("At least two concept banks are required.")
    if reference_key is not None and reference_key not in banks:
        raise ValueError("reference_key must identify a supplied bank.")
    metric = ConceptAlignment(similarity=similarity)
    pairs = (
        ((reference_key, key) for key in banks if key != reference_key)
        if reference_key is not None
        else combinations(banks, 2)
    )
    return {pair: metric.detailed_evaluate(banks[pair[0]], banks[pair[1]]) for pair in pairs}
