"""Angular, spectral, Babel, and observed-support concept-bank diagnostics.

References
----------
Tropp (2004), Greed is good, IEEE Information Theory 50(10):2231–2242.
Welch (1974), Lower bounds on the maximum cross correlation of signals,
IEEE Information Theory 20(3):397–399.
"""

from collections.abc import Iterable

import numpy as np

from ...types import Any, Dict, Optional
from ._utils import (
    _is_real_scalar,
    activity_config,
    activity_mask,
    array_input,
    concept_bank_input,
    observation_chunks,
    validate_batch_size,
)


def _gram(unit):
    gram = np.abs(np.clip(unit @ unit.T, -1, 1))
    np.fill_diagonal(gram, 0)
    return gram


def _order(order, n_concepts):
    if (
        isinstance(order, (bool, np.bool_))
        or not isinstance(order, (int, np.integer))
        or not 0 <= order < n_concepts
    ):
        raise ValueError(f"Order must be an integer in [0, {n_concepts - 1}].")
    return int(order)


def _finite_scalar(value):
    return _is_real_scalar(value) and np.isfinite(value)


def concept_bank_babel_score(concept_bank: np.ndarray, order: int) -> float:
    """Compute the Babel cumulative coherence $\\mu_1(p)$ of a concept bank at order
    ``p``. Lower values indicate less angular overlap among groups of concepts;
    order ``1`` equals mutual coherence and order ``0`` is always ``0.0``.

    Parameters
    ----------
    concept_bank : np.ndarray
        Nonempty 2-D finite real integer/floating bank of shape ``(K, F)`` with
        nonzero rows. Rows are concepts; no axes are flattened.
    order : int
        Required integer order ``p`` in ``[0, K - 1]`` (positional or keyword).
        Booleans and fractional values are rejected.

    Returns
    -------
    babel_score : float
        Dimensionless cumulative coherence in ``[0, order]``, invariant to concept
        row scaling and sign flips. No SVD is performed.

    Raises
    ------
    ValueError
        If the bank is empty, nonfinite, not 2-D real integer/floating, has zero
        rows, or ``order`` is not a non-boolean integer in ``[0, K - 1]``.

    References
    ----------
    Tropp (2004), *Greed is good: algorithmic results for sparse approximation*,
    IEEE Trans. Information Theory 50(10):2231-2242.
    """
    _, _, unit = concept_bank_input(concept_bank, True)
    order = _order(order, len(unit))
    if order == 0:
        return 0.0
    gram = _gram(unit)
    largest = np.partition(gram, len(unit) - order, axis=1)[:, -order:]
    return float(largest.sum(axis=1).max())


def concept_bank_babel_curve(
    concept_bank: np.ndarray, *, max_order: Optional[int] = None
) -> Dict[str, Any]:
    """Compute the nondecreasing Babel cumulative coherence curve $\\mu_1(p)$ for
    orders ``0 .. max_order`` without performing an SVD. Useful for seeing how
    quickly worst-case angular overlap grows with support size.

    Parameters
    ----------
    concept_bank : np.ndarray
        Nonempty 2-D finite real integer/floating bank of shape ``(K, F)`` with
        nonzero rows.
    max_order : int or None
        Maximum order in ``[0, K - 1]`` (default ``K - 1``).

    Returns
    -------
    report : dict
        Dictionary containing the Babel curve and summary endpoints:
        ``n_concepts`` (int): Number of concepts ``K``.
        ``n_features`` (int): Feature dimension ``F``.
        ``requested_max_order`` (int or None): Caller-supplied ``max_order``.
        ``max_order`` (int): Resolved endpoint order ``p``.
        ``orders`` (np.ndarray): Integer array ``[0, ..., p]`` of shape ``(p + 1,)``.
        ``babel_curve`` (np.ndarray): Float64 cumulative coherence values of shape
        ``(p + 1,)`` (bounded by ``orders``, not by ``1``).
        ``babel_score`` (float): Endpoint score ``babel_curve[-1]``.
        ``mutual_coherence`` (float): Actual bank mutual coherence (``0.0`` only for
        a single-row or orthogonal bank, even when ``max_order=0``).

    Raises
    ------
    ValueError
        If the bank is empty, nonfinite, not 2-D real integer/floating, has zero
        rows, or ``max_order`` is neither ``None`` nor a non-boolean integer in
        ``[0, K - 1]``.

    References
    ----------
    Tropp (2004), *Greed is good: algorithmic results for sparse approximation*,
    IEEE Trans. Information Theory 50(10):2231-2242.
    """
    _, _, unit = concept_bank_input(concept_bank, True)
    resolved = len(unit) - 1 if max_order is None else _order(max_order, len(unit))
    gram = _gram(unit)
    curve = np.zeros(resolved + 1)
    if resolved:
        curve[1:] = np.cumsum(np.sort(gram, axis=1)[:, ::-1][:, :resolved], axis=1).max(axis=0)
    return {
        "n_concepts": len(unit),
        "n_features": unit.shape[1],
        "requested_max_order": max_order,
        "max_order": resolved,
        "orders": np.arange(resolved + 1),
        "babel_curve": curve,
        "babel_score": float(curve[-1]),
        "mutual_coherence": float(gram.max()),
    }


def concept_bank_geometry_metrics(
    concept_bank: np.ndarray,
    *,
    spectrum: str = "unit",
    energy_fraction: float = 0.9,
    duplicate_threshold: float = 0.8,
    rank_rtol: Optional[float] = None,
) -> Dict[str, Any]:
    """Summarize angular redundancy, concept norms, effective spectral ranks,
    conditioning, and the Welch bound of a concept bank. Helps diagnose duplicate
    concepts, rank collapse, or poorly conditioned dictionaries.

    Parameters
    ----------
    concept_bank : np.ndarray
        Nonempty 2-D finite real integer/floating bank of shape ``(K, F)`` with
        nonzero rows.
    spectrum : {'unit', 'raw'}
        Matrix used for thin SVD: ``'unit'`` (default) normalizes rows to unit length
        for scale-invariant spectral geometry; ``'raw'`` preserves relative atom
        scales. Original norms are always reported.
    energy_fraction : float
        Target cumulative squared-singular-value energy in ``(0, 1]`` for
        ``energy_rank`` (default ``0.9``).
    duplicate_threshold : float
        Strict signed cosine threshold in ``[0, 1]`` (default ``0.8``) used to count
        positive near-duplicate and antipodal concept pairs.
    rank_rtol : float or None
        Relative singular-value tolerance in ``[0, 1)`` (default
        ``max(K, F) * eps(float64)``). Singular values at or below ``rank_rtol`` are
        excluded from ``energy_rank`` and yield infinite ``condition_number``.

    Returns
    -------
    report : dict
        Dictionary of angular, norm, and spectral geometry diagnostics:
        ``n_concepts``, ``n_features`` (int): Bank dimensions ``K`` and ``F``.
        ``spectrum`` (str), ``energy_fraction``, ``duplicate_threshold``,
        ``rank_rtol`` (float): Resolved settings.
        ``mutual_coherence`` (float): Maximum off-diagonal absolute cosine (``0.0``
        for ``K = 1``).
        ``mean_abs_cosine`` (float): Mean off-diagonal absolute cosine across
        unordered pairs (``0.0`` for ``K = 1``).
        ``positive_duplicate_count`` (int): Unordered pairs with cosine strictly
        above ``duplicate_threshold``.
        ``antipodal_count`` (int): Unordered pairs with cosine strictly below
        ``-duplicate_threshold``.
        ``norm_min``, ``norm_mean``, ``norm_max`` (float): Original concept-vector L2
        norms in activation units.
        ``energy_rank`` (int): Smallest rank reaching ``energy_fraction`` of retained
        spectral energy (equals numerical rank when ``energy_fraction=1.0``).
        ``stable_rank`` (float): Squared Frobenius-to-spectral norm ratio
        ``sum(sigma ** 2) / sigma_max ** 2``.
        ``energy_entropy_rank`` (float): Exponential Shannon entropy of squared
        singular-value energy shares.
        ``condition_number`` (float): Thin-SVD ratio ``sigma_max / sigma_min``
        (``inf`` when ``sigma_min <= rank_rtol * sigma_max``).
        ``welch_bound`` (float): Ambient-dimensional lower bound on mutual coherence
        for overcomplete banks (``0.0`` when ``K <= F``).

    Raises
    ------
    ValueError
        If the bank is empty, nonfinite, not 2-D real integer/floating, has zero
        rows, ``spectrum`` is unknown, or a numeric setting is boolean, nonfinite,
        not a real scalar, or outside its documented range.

    Notes
    -----
    Only ``energy_rank`` thresholds and renormalizes the energy spectrum by
    ``rank_rtol``; fractions exactly on an interior cumulative boundary remain
    subject to float64 roundoff. ``stable_rank`` and ``energy_entropy_rank`` use the
    full spectrum.

    References
    ----------
    Welch (1974), *Lower bounds on the maximum cross correlation of signals*,
    IEEE Trans. Information Theory 20(3):397-399.
    """
    if spectrum not in ("unit", "raw"):
        raise ValueError("spectrum must be 'unit' or 'raw'.")
    if not _finite_scalar(energy_fraction) or not 0 < energy_fraction <= 1:
        raise ValueError("energy_fraction must be in (0,1].")
    if not _finite_scalar(duplicate_threshold) or not 0 <= duplicate_threshold <= 1:
        raise ValueError("duplicate_threshold must be in [0,1].")
    bank, norms, unit = concept_bank_input(concept_bank, True)
    k, f = bank.shape
    tolerance = max(k, f) * np.finfo(np.float64).eps if rank_rtol is None else rank_rtol
    if not _finite_scalar(tolerance) or not 0 <= tolerance < 1:
        raise ValueError("rank_rtol must be in [0,1).")
    cosine = np.clip(unit @ unit.T, -1, 1)[np.triu_indices(k, 1)]
    # Uniform scaling preserves all reported spectral ratios and avoids squaring
    # large raw singular values. Original norms are reported independently.
    matrix = unit if spectrum == "unit" else bank / np.max(np.abs(bank))
    singular = np.linalg.svd(matrix, compute_uv=False)
    relative = singular / singular[0]
    energy = relative**2
    probabilities = energy / energy.sum()
    retained_energy = energy[relative > tolerance]
    cumulative = np.cumsum(retained_energy / retained_energy.sum())
    cumulative[-1] = 1.0
    energy_rank = (
        len(retained_energy)
        if energy_fraction == 1.0
        else int(np.searchsorted(cumulative, energy_fraction) + 1)
    )
    positive = probabilities[probabilities > 0]
    return {
        "n_concepts": k,
        "n_features": f,
        "spectrum": spectrum,
        "energy_fraction": float(energy_fraction),
        "duplicate_threshold": float(duplicate_threshold),
        "rank_rtol": float(tolerance),
        "mutual_coherence": float(np.abs(cosine).max()) if cosine.size else 0.0,
        "mean_abs_cosine": float(np.abs(cosine).mean()) if cosine.size else 0.0,
        "positive_duplicate_count": int(np.sum(cosine > duplicate_threshold)),
        "antipodal_count": int(np.sum(cosine < -duplicate_threshold)),
        "norm_min": float(norms.min()),
        "norm_mean": float(norms.mean()),
        "norm_max": float(norms.max()),
        "energy_rank": energy_rank,
        "stable_rank": float(energy.sum()),
        "energy_entropy_rank": float(np.exp(-np.sum(positive * np.log(positive)))),
        "condition_number": float(1 / relative[-1]) if relative[-1] > tolerance else np.inf,
        "welch_bound": float(np.sqrt((k - f) / (f * (k - 1)))) if k > f else 0.0,
    }


class CoactiveSupportCoherence:
    """Measure Babel-style angular coherence restricted to the concepts that actually
    fire together on each observation. Complements bank-wide Babel curves by testing
    whether correlated concepts co-activate in practice; lower scores indicate less
    overlap among co-active concepts.

    Parameters
    ----------
    activity_basis : {'contribution', 'coefficient'}
        Support definition shared with ``ConceptUsage`` (default ``'contribution'``).
    active_threshold : float
        Finite nonnegative strict activity threshold (default ``0.0``).
    batch_size : int or None
        Number of flattened observations processed per chunk. Bounds temporary
        ``O(chunk_size * K)`` scoring memory; does not bound ``O(K ** 2)`` Gram
        storage or ``O(N)`` opt-in exact quantiles in ``detailed_evaluate``.

    Notes
    -----
    Evaluation uses ``O(K ** 2 * F)`` work to form the absolute Gram matrix and
    ``O(N * K ** 2)`` matrix work to score all observations. No persistent bank or
    state is retained between calls.

    Examples
    --------
    >>> import numpy as np
    >>> from xplique.metrics.concepts import CoactiveSupportCoherence
    >>> bank = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    >>> u = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    >>> CoactiveSupportCoherence()(u, bank)  # doctest: +ELLIPSIS
    0.3535...
    """

    def __init__(
        self,
        *,
        activity_basis: str = "contribution",
        active_threshold: float = 0.0,
        batch_size: Optional[int] = None,
    ):
        activity_config(activity_basis, active_threshold)
        self.activity_basis = activity_basis
        self.active_threshold = float(active_threshold)
        self.batch_size = validate_batch_size(batch_size)

    def _batch_scores(self, batches, concept_bank):
        _, norms, unit = concept_bank_input(concept_bank, True)
        gram = _gram(unit)
        n_observations = 0
        for coefficients in batches:
            values = array_input(coefficients, "coefficients")
            if values.shape[-1] != len(unit):
                raise ValueError("Coefficient and bank concept counts must agree.")
            for chunk in observation_chunks(values, self.batch_size):
                active = activity_mask(chunk, norms, self.activity_basis, self.active_threshold)
                sizes = active.sum(axis=1)
                scores = np.zeros(len(sizes), dtype=np.float64)
                multi = sizes >= 2
                # Empty and singleton supports have score 0.0 by definition; slicing
                # active[multi] before multiplying by gram skips wasted O(K^2) work
                # on sparse codes without changing the scores.
                if np.any(multi):
                    active_multi = active[multi]
                    support_sums = active_multi.astype(np.float64) @ gram
                    scores[multi] = np.where(active_multi, support_sums, -np.inf).max(axis=1)
                n_observations += len(scores)
                yield scores, sizes
        if n_observations == 0:
            raise ValueError("An evaluation must contain observations.")

    def evaluate(self, coefficients: np.ndarray, concept_bank: np.ndarray) -> float:
        """Return the unconditional mean coactive support coherence across all
        observations (empty and singleton supports contribute ``0.0``). See
        ``detailed_evaluate`` for observation scores, quantiles, and multi-support
        summaries.

        Parameters
        ----------
        coefficients : np.ndarray
            Finite, nonempty real integer/floating array of shape ``(..., K)`` with at
            least two axes. Leading axes flatten into equally weighted observations.
        concept_bank : np.ndarray
            Finite, nonempty 2-D real integer/floating bank of shape ``(K, F)``
            matching ``K`` and containing no zero rows.

        Returns
        -------
        mean_support_babel : float
            Dimensionless unconditional mean of per-observation support coherence
            (nonnegative; ``0.0`` when every support has size at most ``1``).

        Raises
        ------
        ValueError
            If arrays have invalid dtypes/dimensions, are empty/nonfinite, have
            mismatched concept counts, or the bank contains zero concept vectors.
        """
        return self.evaluate_batches((coefficients,), concept_bank=concept_bank)

    def __call__(self, coefficients: np.ndarray, concept_bank: np.ndarray) -> float:
        """Alias for ``evaluate``."""
        return self.evaluate(coefficients, concept_bank)

    def evaluate_batches(self, batches: Iterable[np.ndarray], *, concept_bank: np.ndarray) -> float:
        """Return the unconditional mean coactive support coherence in a single pass
        over streamed coefficient batches with a fixed concept bank.

        Parameters
        ----------
        batches : iterable of np.ndarray
            One-pass iterable of finite, nonempty real integer/floating coefficient
            arrays of shape ``(..., K)`` with fixed ``K``.
        concept_bank : np.ndarray
            Finite, nonempty 2-D real integer/floating bank of shape ``(K, F)``
            matching ``K`` and containing no zero rows.

        Returns
        -------
        mean_support_babel : float
            Dimensionless unconditional mean of per-observation support coherence
            across all streamed observations, matching ``evaluate`` on concatenated
            batches.

        Raises
        ------
        ValueError
            If the stream is empty, arrays have invalid dtypes/dimensions, are
            empty/nonfinite, have mismatched concept counts, or the bank contains
            zero concept vectors.
        """
        # Stream scalar mean with O(1) running (total, count) state for API
        # symmetry with other observation-level metrics; detailed_evaluate stays
        # single-array because exact quantiles require O(N) observation scores.
        total, count = 0.0, 0
        for scores, _ in self._batch_scores(batches, concept_bank):
            total += float(scores.sum())
            count += len(scores)
        return float(total / count)

    def detailed_evaluate(
        self, coefficients: np.ndarray, concept_bank: np.ndarray
    ) -> Dict[str, Any]:
        """Return the full distribution of per-observation coactive support coherence
        scores, linear quantiles, and support-size breakdowns.

        Parameters
        ----------
        coefficients : np.ndarray
            Finite, nonempty real integer/floating array of shape ``(..., K)`` with at
            least two axes.
        concept_bank : np.ndarray
            Finite, nonempty 2-D real integer/floating bank of shape ``(K, F)``
            matching ``K`` and containing no zero rows.

        Returns
        -------
        details : dict
            Dictionary of observation-level coherence scores and summaries:
            ``mean_support_babel`` (float): Unconditional mean matching ``evaluate``.
            ``mean_multi_support_babel`` (float): Mean conditioned on supports of size
            at least ``2`` (``NaN`` if no observation has 2+ active concepts).
            ``max_support_babel`` (float): Maximum observed support coherence.
            ``support_babel_q25``, ``support_babel_median``, ``support_babel_q75``,
            ``support_babel_q95`` (float): Exact NumPy linear quantiles of scores.
            ``n_observations``, ``n_concepts`` (int): Observation and concept counts.
            ``empty_support_count``, ``singleton_support_count``,
            ``multi_support_count`` (int): Counts of supports of size 0, 1, and >= 2.
            ``mean_support_size`` (float), ``max_support_size`` (int): Active concept
            count summaries.
            ``activity_basis`` (str), ``active_threshold`` (float): Resolved settings.
            ``observation_scores`` (np.ndarray): Per-observation coherence scores of
            shape ``(N,)`` (``0.0`` for empty/singleton supports).
            ``observation_support_sizes`` (np.ndarray): Per-observation active concept
            counts of shape ``(N,)``.

        Raises
        ------
        ValueError
            If arrays have invalid dtypes/dimensions, are empty/nonfinite, have
            mismatched concept counts, or the bank contains zero concept vectors.
        """
        chunks = list(self._batch_scores((coefficients,), concept_bank))
        scores = np.concatenate([chunk[0] for chunk in chunks])
        sizes = np.concatenate([chunk[1] for chunk in chunks])
        multi = sizes >= 2
        quantiles = np.quantile(scores, [0.25, 0.5, 0.75, 0.95], method="linear")
        return {
            "n_observations": len(scores),
            "n_concepts": np.asarray(concept_bank).shape[0],
            "empty_support_count": int(np.sum(sizes == 0)),
            "singleton_support_count": int(np.sum(sizes == 1)),
            "multi_support_count": int(multi.sum()),
            "mean_support_size": float(sizes.mean()),
            "max_support_size": int(sizes.max()),
            "mean_support_babel": float(scores.mean()),
            "max_support_babel": float(scores.max()),
            "mean_multi_support_babel": float(scores[multi].mean()) if multi.any() else np.nan,
            **dict(
                zip(
                    (
                        "support_babel_q25",
                        "support_babel_median",
                        "support_babel_q75",
                        "support_babel_q95",
                    ),
                    map(float, quantiles),
                )
            ),
            "activity_basis": self.activity_basis,
            "active_threshold": self.active_threshold,
            "observation_scores": scores,
            "observation_support_sizes": sizes,
        }
