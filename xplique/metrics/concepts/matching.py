"""One-shot matching of complete concept banks in shared feature coordinates."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ...types import Any, Dict, Optional, Tuple
from ._utils import array_input, concept_bank_input, observation_chunks, validate_batch_size


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
        Headline alignment score ``mean_similarity * coverage`` (equivalently
        padding unmatched concepts with ``0.0``).

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
    mean_similarity = float(matched.mean())
    coverage = float(len(rows) / max(similarity.shape))
    return matched, mean_similarity, coverage, float(mean_similarity * coverage)


def _assign_prepared(
    prepared_reference, prepared_compared, similarity_mode, *, need_matched_cosine
):
    """Run Hungarian matching on pre-validated/normalized banks."""
    reference, _, reference_unit = prepared_reference
    compared, _, compared_unit = prepared_compared
    if reference.shape[1] != compared.shape[1]:
        raise ValueError("Concept banks must share feature dimensions.")
    if similarity_mode == "dot":
        # In dot mode, raw inner products determine the assignment directly,
        # so skip the full K1 x K2 cosine matrix and compute cosines only
        # for the M matched pairs when detailed_evaluate requests them.
        similarity = reference @ compared.T
        if not np.isfinite(similarity).all():
            raise ValueError(
                "Dot-product similarity matrix overflowed float64; rescale concept banks first."
            )
        rows, columns = linear_sum_assignment(similarity, maximize=True)
        matched_cosine = (
            np.clip(np.sum(reference_unit[rows] * compared_unit[columns], axis=1), -1, 1)
            if need_matched_cosine
            else None
        )
        return similarity, matched_cosine, rows, columns
    cosine = np.clip(reference_unit @ compared_unit.T, -1, 1)
    similarity = np.abs(cosine) if similarity_mode == "absolute_cosine" else cosine
    rows, columns = linear_sum_assignment(similarity, maximize=True)
    matched_cosine = cosine[rows, columns] if need_matched_cosine else None
    return similarity, matched_cosine, rows, columns


def _detailed_from_prepared(prepared_reference, prepared_compared, similarity_mode):
    """Build a HungarianMatchingResult from pre-validated/normalized banks."""
    similarity, matched_cosine, rows, columns = _assign_prepared(
        prepared_reference, prepared_compared, similarity_mode, need_matched_cosine=True
    )
    assert matched_cosine is not None
    matched, mean_similarity, coverage, coverage_weighted = _matched_summary(
        similarity, rows, columns
    )
    return HungarianMatchingResult(
        similarity_mode,
        similarity.shape[0],
        similarity.shape[1],
        len(rows),
        rows,
        columns,
        matched,
        np.abs(matched_cosine),
        np.where(matched_cosine < 0, -1, 1),
        mean_similarity,
        coverage,
        coverage_weighted,
    )


class ConceptAlignment:
    """Measure alignment between two concept banks via maximum-sum Hungarian matching.
    Returns ``mean_similarity * coverage`` (equivalently padding unmatched concepts
    with ``0.0``), which pulls both positive and negative scores toward ``0.0`` and
    penalizes unequal bank sizes whenever matched similarities are nonnegative. No
    banks or assignments are stored between calls.

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

    In ``'dot'`` mode, inner products are computed in float64 without rescaling;
    extremely large finite banks (beyond roughly ``1e150`` in magnitude) can overflow
    and raise ``ValueError``.

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
            contain zero concept vectors, have different feature counts, or overflow
            float64 dot products.
        """
        similarity, _, rows, columns = _assign_prepared(
            concept_bank_input(reference_concept_bank, True),
            concept_bank_input(compared_concept_bank, True),
            self.similarity,
            need_matched_cosine=False,
        )
        return _matched_summary(similarity, rows, columns)[3]

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
            zero vectors, have different feature counts, or overflow float64 dot
            products.
        """
        return _detailed_from_prepared(
            concept_bank_input(reference_concept_bank, True),
            concept_bank_input(compared_concept_bank, True),
            self.similarity,
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
        If ``banks`` is not a mapping, fewer than two banks are supplied,
        ``reference_key`` is not in ``banks``, or any compared pair violates
        ``ConceptAlignment`` input rules.
    """
    # Reject sequences explicitly: lists like [bank_a, bank_b] satisfy len >= 2
    # and combinations() but would otherwise fail later with a confusing indexing error.
    if not isinstance(banks, Mapping):
        raise ValueError("banks must be a mapping from run keys to concept banks.")
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
    # Validate and normalize each bank once into an evaluation-local dict rather
    # than repeating float64 conversion and row normalization M - 1 times across
    # pairs, while keeping ConceptAlignment stateless across calls.
    prepared = {key: concept_bank_input(bank, True) for key, bank in banks.items()}
    return {
        pair: _detailed_from_prepared(prepared[pair[0]], prepared[pair[1]], metric.similarity)
        for pair in pairs
    }


def _matching_columns(matching):
    """Read the parts of a matching result needed to compare columns safely."""
    if not isinstance(matching, HungarianMatchingResult):
        raise ValueError("matching must be a HungarianMatchingResult.")
    widths = (matching.n_reference_concepts, matching.n_compared_concepts)
    counts = (*widths, matching.n_matched_pairs)
    if any(
        not isinstance(count, (int, np.integer)) or isinstance(count, (bool, np.bool_)) or count < 1
        for count in counts
    ) or matching.n_matched_pairs != min(widths):
        raise ValueError("matching has inconsistent concept and pair counts.")
    columns = tuple(
        np.asarray(indices)
        for indices in (matching.assignment_reference_indices, matching.assignment_compared_indices)
    )
    shape = (matching.n_matched_pairs,)
    for indices, width in zip(columns, widths):
        if (
            indices.shape != shape
            or indices.dtype.kind not in "iu"
            or np.any(indices < 0)
            or np.any(indices >= width)
            or len(np.unique(indices)) != shape[0]
        ):
            raise ValueError("matching must contain unique, in-range integer indices.")
    polarity = np.asarray(matching.matched_polarities)
    if (
        polarity.shape != shape
        or polarity.dtype.kind not in "iuf"
        or not np.isin(polarity, (-1, 1)).all()
    ):
        raise ValueError("matching must contain one polarity (+1 or -1) per pair.")
    if matching.similarity not in ("cosine", "absolute_cosine", "dot"):
        raise ValueError("Unknown matching similarity mode.")
    return columns, widths, polarity if matching.similarity == "absolute_cosine" else 1


class _PairedMoments:
    """Merge centered moments for corresponding columns, not all column pairs."""

    def __init__(self, n_pairs):
        self.count = 0
        self.exponent = np.full((2, n_pairs), -1073, dtype=np.int32)
        self.origin = np.zeros((2, n_pairs))
        self.mean = np.zeros((2, n_pairs))
        self.variance_sum = np.zeros((2, n_pairs))
        self.covariance_sum = np.zeros(n_pairs)

    def merge(self, paired):
        """Accumulate a chunk with a fixed origin and adaptive power-of-two scales."""
        size = paired.shape[1]
        mantissa, exponent = np.frexp(np.max(np.abs(paired), axis=1))
        exponent = np.maximum(self.exponent, np.where(mantissa == 0, self.exponent, exponent))
        shift = self.exponent - exponent
        if self.count:
            self.origin = np.ldexp(self.origin, shift)
            self.mean = np.ldexp(self.mean, shift)
            self.variance_sum = np.ldexp(self.variance_sum, 2 * shift)
            self.covariance_sum = np.ldexp(self.covariance_sum, shift.sum(axis=0))
        self.exponent = exponent
        scaled = np.ldexp(paired, -exponent[:, None, :])
        if not self.count:
            self.origin = scaled[:, 0, :].copy()
        shifted = scaled - self.origin[:, None, :]
        mean = shifted.mean(axis=1)
        centered = shifted - mean[:, None, :]
        total = self.count + size
        weight = self.count * size / total
        delta = mean - self.mean
        self.variance_sum += np.square(centered).sum(axis=1) + weight * np.square(delta)
        cross = (centered[0] * centered[1]).sum(axis=0)
        self.covariance_sum += cross + weight * delta[0] * delta[1]
        self.mean += delta * (size / total)
        self.count = total

    def correlations(self):
        """Return per-pair Pearson coefficients, with NaN for constant columns."""
        std = np.sqrt(self.variance_sum)
        denominator = std[0] * std[1]
        correlations = np.full_like(denominator, np.nan)
        np.divide(self.covariance_sum, denominator, out=correlations, where=denominator > 0)
        return np.clip(correlations, -1.0, 1.0)


class MatchedCodeConsistency:
    """Measure code agreement on common observations after decoder-bank matching.

    Parameters
    ----------
    batch_size : int or None
        Maximum flattened observations processed per chunk. ``None`` processes
        each supplied coefficient array at once.

    Notes
    -----
    Pearson correlation measures co-variation, not agreement in offsets or firing
    supports. The mean weights scorable pairs equally and does not penalize bank
    coverage. Inspect both coverage and the number of scorable pairs.
    """

    def __init__(self, *, batch_size: Optional[int] = None):
        self.batch_size = validate_batch_size(batch_size)

    def _evaluate(self, batches, matching):
        (rows, columns), widths, polarity = _matching_columns(matching)
        moments = _PairedMoments(len(rows))
        for batch in batches:
            if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                raise ValueError("Each batch must be a pair of coefficient arrays.")
            reference = array_input(batch[0], "reference_coefficients")
            compared = array_input(batch[1], "compared_coefficients")
            if reference.shape[:-1] != compared.shape[:-1]:
                raise ValueError("Coefficient arrays must have identical observation shapes.")
            if (reference.shape[-1], compared.shape[-1]) != widths:
                raise ValueError("Coefficient widths must match the assignment's concept counts.")
            for left, right in zip(
                observation_chunks(reference, self.batch_size),
                observation_chunks(compared, self.batch_size),
            ):
                moments.merge(np.stack((left[:, rows], right[:, columns])))
        if not moments.count:
            raise ValueError("An evaluation must contain observations.")
        correlations = moments.correlations() * polarity
        scorable = np.isfinite(correlations)
        return {
            "mean_correlation": float(correlations[scorable].mean()) if scorable.any() else np.nan,
            "matched_correlations": correlations,
            "n_matched_pairs": len(rows),
            "n_scorable_pairs": int(scorable.sum()),
            "coverage": len(rows) / max(widths),
            "n_observations": moments.count,
        }

    def evaluate(self, reference_coefficients, compared_coefficients, *, matching) -> float:
        """Return mean signed Pearson correlation over scorable matched pairs.

        Parameters
        ----------
        reference_coefficients, compared_coefficients : np.ndarray
            Finite arrays of shape ``(..., K_ref)`` and ``(..., K_cmp)``. Their
            leading dimensions and observation order must match exactly.
        matching : HungarianMatchingResult
            Decoder-bank assignment from ``ConceptAlignment.detailed_evaluate``.
            Matched polarity is applied only for an ``absolute_cosine`` assignment.

        Returns
        -------
        float
            Mean correlation in ``[-1, 1]``, or ``NaN`` if no pair is scorable.
        """
        return self.evaluate_batches(
            ((reference_coefficients, compared_coefficients),), matching=matching
        )

    __call__ = evaluate

    def evaluate_batches(self, batches: Iterable, *, matching: HungarianMatchingResult) -> float:
        """Return the same score from a one-pass iterable of aligned coefficient pairs.

        Parameters
        ----------
        batches : iterable of pairs of np.ndarray
            Each pair follows the shape and order contract of ``evaluate``.
        matching : HungarianMatchingResult
            Fixed decoder-bank assignment for all batches.

        Returns
        -------
        float
            Pooled mean of per-pair correlations, not a mean of batch correlations.
        """
        return self._evaluate(batches, matching)["mean_correlation"]

    def detailed_evaluate(self, reference_coefficients, compared_coefficients, *, matching) -> dict:
        """Report correlations in assignment order and the score's denominators.

        Parameters
        ----------
        reference_coefficients, compared_coefficients : np.ndarray
            Aligned coefficient arrays as in ``evaluate``.
        matching : HungarianMatchingResult
            Decoder-bank assignment for the supplied coefficients.

        Returns
        -------
        dict
            ``mean_correlation``, ``matched_correlations``, ``n_matched_pairs``,
            ``n_scorable_pairs``, ``coverage``, and ``n_observations``. A pair with
            a constant column or fewer than two observations has correlation ``NaN``.
        """
        return self._evaluate(((reference_coefficients, compared_coefficients),), matching)
