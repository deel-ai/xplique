"""One-shot matching of complete concept banks in shared feature coordinates."""

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ...types import Any, Dict, Optional, Tuple
from ._utils import concept_bank_input


@dataclass(eq=False)
class HungarianMatchingResult:
    """Structured result of optimal one-to-one Hungarian matching between two concept
    banks, containing matched indices, per-pair similarities, polarities, and
    bank-level summary scores.

    Parameters
    ----------
    similarity : str
        Matching objective used: ``'cosine'``, ``'absolute_cosine'``, or ``'dot'``.
    n_reference_concepts : int
        Number of concepts ``K_reference`` in the reference bank.
    n_compared_concepts : int
        Number of concepts ``K_compared`` in the compared bank.
    n_matched_pairs : int
        Number of matched concept pairs ``M = min(K_reference, K_compared)``.
    assignment_reference_indices : np.ndarray
        Integer row indices into the reference bank, shape ``(M,)``.
    assignment_compared_indices : np.ndarray
        Integer row indices into the compared bank, shape ``(M,)``.
    matched_similarities : np.ndarray
        Objective value for each matched pair, shape ``(M,)`` (dimensionless cosines
        or scale-sensitive dot products).
    matched_absolute_cosines : np.ndarray
        Absolute cosine similarity in ``[0, 1]`` for each matched pair, shape ``(M,)``.
    matched_polarities : np.ndarray
        Sign (``+1`` or ``-1``) of each matched pair's cosine similarity, shape
        ``(M,)`` (zero cosine uses ``+1`` by convention).
    mean_similarity : float
        Mean of ``matched_similarities`` over the ``M`` matched pairs.
    coverage : float
        Fraction of concepts covered, ``M / max(K_reference, K_compared)`` in
        ``(0, 1]``.
    coverage_weighted_similarity : float
        Headline alignment score ``mean_similarity * coverage``.

    Notes
    -----
    Instances compare by identity (``eq=False``) to avoid ambiguous array truth values.
    Compare array fields individually with NumPy testing utilities when checking
    scientific equivalence.
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
    """Measure alignment between two concept banks via maximum-sum Hungarian matching.
    Returns ``mean_similarity * coverage``, penalizing unequal bank sizes without
    storing banks or assignments between calls.

    Parameters
    ----------
    similarity : {'cosine', 'absolute_cosine', 'dot'}
        Pairwise similarity objective. ``'cosine'`` (default) measures signed
        directional agreement; ``'absolute_cosine'`` ignores concept sign flips (use
        only when simultaneous atom/coefficient negation is a valid symmetry);
        ``'dot'`` is scale-sensitive. Zero concept vectors are rejected.

    Notes
    -----
    Both banks must live in the same feature coordinate system; matching feature
    dimensions alone do not make two banks comparable across models or preprocessing.

    Examples
    --------
    >>> import numpy as np
    >>> from xplique.metrics.concepts import ConceptAlignment
    >>> bank = np.eye(3)
    >>> ConceptAlignment()(bank, bank[[2, 0, 1]])
    1.0
    """

    def __init__(self, *, similarity: str = "cosine"):
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

    def evaluate(
        self, reference_concept_bank: np.ndarray, compared_concept_bank: np.ndarray
    ) -> float:
        """Compute the coverage-weighted Hungarian alignment score between two banks.
        Higher values indicate closer alignment; see ``detailed_evaluate`` for the
        matched indices, per-pair similarities, and coverage breakdown.

        Parameters
        ----------
        reference_concept_bank : np.ndarray
            Finite, nonempty 2-D real integer/floating bank of shape
            ``(K_reference, F)`` with nonzero rows in shared feature coordinates.
        compared_concept_bank : np.ndarray
            Finite, nonempty 2-D real integer/floating bank of shape
            ``(K_compared, F)`` with nonzero rows and the same feature count ``F``.

        Returns
        -------
        score : float
            Coverage-weighted mean matched similarity (``mean_similarity * coverage``).
            Cosine scores lie in ``[-1, 1]`` (or ``[0, 1]`` for ``'absolute_cosine'``);
            dot scores are unbounded and carry product-of-bank units.

        Raises
        ------
        ValueError
            If banks are not 2-D real integer/floating arrays, are empty/nonfinite,
            contain zero concept vectors, or have different feature counts.
        """
        similarity, _, rows, columns = self._assignment(
            reference_concept_bank, compared_concept_bank
        )
        return _matched_summary(similarity, rows, columns)[2]

    def __call__(
        self, reference_concept_bank: np.ndarray, compared_concept_bank: np.ndarray
    ) -> float:
        """Alias for ``evaluate``."""
        return self.evaluate(reference_concept_bank, compared_concept_bank)

    def detailed_evaluate(
        self, reference_concept_bank: np.ndarray, compared_concept_bank: np.ndarray
    ) -> HungarianMatchingResult:
        """Compute the full Hungarian assignment, per-pair similarities, polarities,
        and bank-level alignment summary.

        Parameters
        ----------
        reference_concept_bank : np.ndarray
            Finite, nonempty 2-D real integer/floating bank of shape
            ``(K_reference, F)`` with nonzero rows in shared feature coordinates.
        compared_concept_bank : np.ndarray
            Finite, nonempty 2-D real integer/floating bank of shape
            ``(K_compared, F)`` with nonzero rows and the same feature count ``F``.

        Returns
        -------
        result : HungarianMatchingResult
            Full matching result whose ``coverage_weighted_similarity`` equals
            ``evaluate``. Tied optimal assignments may vary across SciPy versions;
            zero cosine receives polarity ``+1`` by convention.

        Raises
        ------
        ValueError
            If banks have invalid dtypes/dimensions, are empty/nonfinite, contain
            zero vectors, or have different feature counts.
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


def compare_concept_banks(
    banks: Mapping[Any, np.ndarray],
    *,
    similarity: str = "cosine",
    reference_key: Optional[Any] = None,
) -> Dict[Tuple[Any, Any], HungarianMatchingResult]:
    """Compare multiple concept banks pairwise (or against one reference bank) using
    ``ConceptAlignment.detailed_evaluate``. Useful for inspecting agreement across
    random seeds, hyperparameters, or concept-extraction runs.

    Parameters
    ----------
    banks : dict
        Mapping of at least two run keys to finite, nonempty 2-D real concept banks
        ``(K, F)`` with nonzero rows in shared feature coordinates. ``K`` may vary
        across entries; ``F`` must match for every compared pair.
    similarity : {'cosine', 'absolute_cosine', 'dot'}
        Pairwise similarity objective passed to ``ConceptAlignment``.
    reference_key : optional
        Key in ``banks`` to compare all other banks against (star topology). When
        ``None`` (default), evaluates every unordered pair.

    Returns
    -------
    comparisons : dict
        Mapping from ``(reference_run, compared_run)`` key pairs to
        ``HungarianMatchingResult`` objects. Pairwise comparisons across repeated runs
        are not statistically independent replicates.

    Raises
    ------
    ValueError
        If fewer than two banks are supplied, ``reference_key`` is not in ``banks``,
        or any compared pair violates ``ConceptAlignment`` input rules.
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
