"""Concept coefficient usage and strict-support sparsity."""

from collections.abc import Iterable

import numpy as np

from ...types import Any, Dict, Optional
from ._utils import (
    _concept_bank_row_parts,
    _Moments,
    activity_config,
    array_input,
    concept_bank_input,
    observation_chunks,
    validate_batch_size,
)


def _coefficient_statistics(
    batches, concept_bank, basis, threshold, batch_size, *, usage_stats=True, details=False
):
    _, norms, _ = concept_bank_input(concept_bank)
    n_concepts = len(norms)
    moments = _Moments()
    absolute_sum = np.zeros(n_concepts)
    energy = np.zeros(n_concepts)
    active_counts = np.zeros(n_concepts, dtype=np.int64)
    supports = []
    n_observations = 0
    for coefficients in batches:
        values = array_input(coefficients, "coefficients")
        if values.shape[-1] != n_concepts:
            raise ValueError("Coefficient and concept-bank concept counts must agree.")
        for chunk in observation_chunks(values, batch_size):
            n_observations += len(chunk)
            # Reuse |chunk| / contribution across usage accumulators and the
            # support mask rather than calling activity_mask redundantly, and
            # skip moment/energy reductions in sparsity-only mode.
            abs_chunk = np.abs(chunk)
            contribution = abs_chunk * norms if (usage_stats or basis == "contribution") else None
            if usage_stats:
                assert contribution is not None
                moments.merge(chunk)
                absolute_sum += contribution.sum(axis=0)
                energy += (contribution**2).sum(axis=0)
            active = (contribution if basis == "contribution" else abs_chunk) > threshold
            active_counts += active.sum(axis=0)
            if details:
                supports.append(active.sum(axis=1))
    if n_observations == 0:
        raise ValueError("An evaluation must contain observations.")
    result: Dict[str, Any] = {
        "n_observations": n_observations,
        "n_concepts": n_concepts,
        "n_elements": n_observations * n_concepts,
        "activity_basis": basis,
        "active_threshold": threshold,
        "active_fraction": active_counts / n_observations,
        "sparsity": float(1 - active_counts.sum() / (n_observations * n_concepts)),
    }
    if usage_stats:
        assert moments.mean is not None and moments.centered is not None
        max_energy = float(energy.max())
        if max_energy > 0 and np.isfinite(max_energy):
            # Rescale by max_energy before summing so finite nonzero component
            # energies cannot overflow energy.sum() to inf and yield zero shares.
            scaled_energy = energy / max_energy
            energy_share = scaled_energy / scaled_energy.sum()
        else:
            energy_share = np.full(n_concepts, np.nan)
        result.update(
            mean_coefficient=moments.mean,
            std_coefficient=np.sqrt(moments.centered / n_observations),
            effective_usage=absolute_sum / n_observations,
            component_energy=energy,
            component_energy_share=energy_share,
            n_active_concepts=int(np.count_nonzero(active_counts)),
        )
    if details:
        result["observation_support_sizes"] = np.concatenate(supports)
    return result


class ConceptUsage:
    """Multivariate report quantifying how often and how strongly each concept is used.
    Combines signed coefficient moments, norm-weighted linear component magnitudes and
    energies, per-concept firing rates, dead-concept counts, and overall sparsity.

    Parameters
    ----------
    activity_basis : {'contribution', 'coefficient'}
        Quantity compared strictly against ``active_threshold``. ``'contribution'``
        (default) uses ``abs(U) * norm(D)``, which is invariant to reciprocal
        rescaling of coefficients and concept vectors; ``'coefficient'`` uses
        ``abs(U)`` directly.
    active_threshold : float
        Finite nonnegative threshold (default ``0.0``) applied with strict inequality
        (``>``) in the units of ``activity_basis``.
    batch_size : int or None
        Number of flattened observations processed per chunk. ``None`` (default)
        processes each supplied array at once.

    Notes
    -----
    Norm-weighted quantities describe linear decoder components ``u_k * d_k`` and
    exclude affine decoder offsets. Because concept vectors are generally
    non-orthogonal, ``component_energy_share`` is not additive explained variance.
    Zero concept vectors are permitted.

    Squared component energies are accumulated in float64 without rescaling.
    Extremely large finite contributions (beyond roughly ``1e150`` in magnitude)
    can overflow to ``inf``/``NaN``; rescale inputs first if needed.

    Examples
    --------
    >>> import numpy as np
    >>> from xplique.metrics.concepts import ConceptUsage
    >>> u = np.array([[1.0, 0.0], [0.0, 2.0]])
    >>> bank = np.eye(2)
    >>> report = ConceptUsage()(u, bank)
    >>> report["n_active_concepts"], report["sparsity"]
    (2, 0.5)
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

    def evaluate(self, coefficients: np.ndarray, concept_bank: np.ndarray) -> Dict[str, Any]:
        """Compute pooled per-concept and global usage statistics (no separate
        ``detailed_evaluate`` method is needed because all intermediates are returned).

        Parameters
        ----------
        coefficients : np.ndarray
            Finite, nonempty real integer/floating array of shape ``(..., K)`` with at
            least two axes. Leading axes flatten into equally weighted observations.
        concept_bank : np.ndarray
            Finite, nonempty real integer/floating bank of shape ``(K, F)``. Zero
            concept rows are allowed; ``K`` must match ``coefficients.shape[-1]``.

        Returns
        -------
        report : dict
            Dictionary of concept-level and global usage diagnostics:
            ``n_observations``, ``n_concepts``, ``n_elements`` (int): Flattened counts.
            ``activity_basis`` (str), ``active_threshold`` (float): Resolved settings.
            ``mean_coefficient`` (np.ndarray): Signed mean per concept, shape ``(K,)``.
            ``std_coefficient`` (np.ndarray): Population std per concept, shape ``(K,)``.
            ``effective_usage`` (np.ndarray): Mean ``abs(U) * norm(D)`` per concept in
            activation units, shape ``(K,)``.
            ``component_energy`` (np.ndarray): Sum of ``(U * norm(D)) ** 2`` per
            concept, shape ``(K,)``.
            ``component_energy_share`` (np.ndarray): Normalized ``component_energy``
            of shape ``(K,)`` (all ``NaN`` when total energy is zero).
            ``active_fraction`` (np.ndarray): Firing probability per concept in
            ``[0, 1]``, shape ``(K,)``.
            ``n_active_concepts`` (int): Number of concepts firing at least once.
            ``sparsity`` (float): Global inactive fraction in ``[0, 1]``.

        Raises
        ------
        ValueError
            If arrays have invalid dtypes/dimensions, are empty/nonfinite, or
            coefficient and bank concept counts differ.
        """
        return self.evaluate_batches((coefficients,), concept_bank=concept_bank)

    def __call__(self, coefficients: np.ndarray, concept_bank: np.ndarray) -> Dict[str, Any]:
        """Alias for ``evaluate``."""
        return self.evaluate(coefficients, concept_bank)

    def evaluate_batches(
        self, batches: Iterable[np.ndarray], *, concept_bank: np.ndarray
    ) -> Dict[str, Any]:
        """Compute the pooled usage report in a single pass over streamed coefficient
        batches with a fixed concept bank and O(K) local state.

        Parameters
        ----------
        batches : iterable of np.ndarray
            One-pass iterable of finite, nonempty real integer/floating coefficient
            arrays of shape ``(..., K)`` with fixed ``K``.
        concept_bank : np.ndarray
            Finite, nonempty real integer/floating bank of shape ``(K, F)``, fixed
            for this evaluation. Zero rows are allowed.

        Returns
        -------
        report : dict
            Pooled concept and global usage report with the exact keys, shapes, and
            zero-energy semantics of ``evaluate``.

        Raises
        ------
        ValueError
            If the stream is empty, arrays have invalid dtypes/dimensions, are
            empty/nonfinite, or any batch's concept count differs from the bank.
        """
        return _coefficient_statistics(
            batches,
            concept_bank,
            self.activity_basis,
            self.active_threshold,
            self.batch_size,
        )


class ConceptSparsity:
    """Global inactive fraction of concept activations in ``[0, 1]``, using the same
    strict activity rule as ``ConceptUsage``. Higher values indicate sparser concept
    representations (fewer concepts firing per observation).

    Parameters
    ----------
    activity_basis : {'contribution', 'coefficient'}
        Magnitude compared against ``active_threshold``. ``'contribution'`` (default)
        is invariant to reciprocal concept/coefficient rescaling.
    active_threshold : float
        Finite nonnegative strict activity threshold (default ``0.0``).
    batch_size : int or None
        Number of flattened observations processed per chunk (default ``None``).
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

    def _statistics(self, batches, concept_bank, details=False):
        return _coefficient_statistics(
            batches,
            concept_bank,
            self.activity_basis,
            self.active_threshold,
            self.batch_size,
            usage_stats=False,
            details=details,
        )

    def evaluate(self, coefficients: np.ndarray, concept_bank: np.ndarray) -> float:
        """Return the pooled inactive fraction across all observations and concepts.
        See ``detailed_evaluate`` for per-concept firing rates and observation support
        sizes.

        Parameters
        ----------
        coefficients : np.ndarray
            Finite, nonempty real integer/floating array of shape ``(..., K)`` with at
            least two axes. Leading axes flatten into equally weighted observations.
        concept_bank : np.ndarray
            Finite, nonempty real integer/floating bank of shape ``(K, F)`` matching
            ``K``. Zero concept rows are allowed.

        Returns
        -------
        sparsity : float
            Dimensionless pooled inactive fraction in ``[0, 1]`` (``1.0`` when no
            concept ever fires).

        Raises
        ------
        ValueError
            If arrays have invalid dtypes/dimensions, are empty/nonfinite, or
            coefficient and bank concept counts differ.
        """
        return self.evaluate_batches((coefficients,), concept_bank=concept_bank)

    def __call__(self, coefficients: np.ndarray, concept_bank: np.ndarray) -> float:
        """Alias for ``evaluate``."""
        return self.evaluate(coefficients, concept_bank)

    def evaluate_batches(self, batches: Iterable[np.ndarray], *, concept_bank: np.ndarray) -> float:
        """Return pooled sparsity in a single pass over streamed coefficient batches.

        Parameters
        ----------
        batches : iterable of np.ndarray
            One-pass iterable of finite, nonempty real integer/floating coefficient
            arrays of shape ``(..., K)`` with fixed ``K``.
        concept_bank : np.ndarray
            Finite, nonempty real integer/floating bank of shape ``(K, F)``. Zero
            rows are allowed.

        Returns
        -------
        sparsity : float
            Dimensionless pooled inactive fraction in ``[0, 1]``, matching
            ``evaluate`` on concatenated batches.

        Raises
        ------
        ValueError
            If the stream is empty, arrays have invalid dtypes/dimensions, are
            empty/nonfinite, or any batch's concept count differs from the bank.
        """
        return self._statistics(batches, concept_bank)["sparsity"]

    def detailed_evaluate(
        self, coefficients: np.ndarray, concept_bank: np.ndarray
    ) -> Dict[str, Any]:
        """Inspect global sparsity alongside per-concept firing rates and
        per-observation active support sizes.

        Parameters
        ----------
        coefficients : np.ndarray
            Finite, nonempty real integer/floating array of shape ``(..., K)`` with at
            least two axes.
        concept_bank : np.ndarray
            Finite, nonempty real integer/floating bank of shape ``(K, F)`` matching
            ``K``. Zero rows are allowed.

        Returns
        -------
        details : dict
            Dictionary of global, concept-level, and observation-level sparsity fields:
            ``sparsity`` (float): Global inactive fraction in ``[0, 1]``, matching
            ``evaluate``.
            ``active_fraction`` (np.ndarray): Per-concept firing rate in ``[0, 1]``,
            shape ``(K,)``.
            ``observation_support_sizes`` (np.ndarray): Active concept count per
            observation, integer array of shape ``(N,)``.
            ``n_observations``, ``n_concepts``, ``n_elements`` (int): Flattened counts.
            ``activity_basis`` (str), ``active_threshold`` (float): Resolved settings.

        Raises
        ------
        ValueError
            If arrays have invalid dtypes/dimensions, are empty/nonfinite, or
            coefficient and bank concept counts differ.
        """
        report = self._statistics((coefficients,), concept_bank, True)
        keys = (
            "sparsity",
            "active_fraction",
            "observation_support_sizes",
            "n_observations",
            "n_concepts",
            "n_elements",
            "activity_basis",
            "active_threshold",
        )
        return {key: report[key] for key in keys}


def _hoyer_rows(chunk, norm_mantissa, norm_exponent):
    """Normalize contributions by a shared power of two before taking norms."""
    mantissa, exponent = np.frexp(np.abs(chunk))
    mantissa *= norm_mantissa
    exponent += norm_exponent
    nonzero = mantissa != 0
    # Zero entries must not set the row exponent, including for subnormal inputs.
    row_exponent = np.max(np.where(nonzero, exponent, -4096), axis=1, keepdims=True)
    scaled = np.ldexp(mantissa, np.where(nonzero, exponent - row_exponent, 0))
    n_concepts = chunk.shape[1]
    nonzero_rows = nonzero.any(axis=1)
    scores = np.full(len(chunk), np.nan)
    if n_concepts > 1:
        root_k = np.sqrt(n_concepts)
        l1 = scaled.sum(axis=1)
        l2 = np.sqrt(np.square(scaled).sum(axis=1))
        scores[nonzero_rows] = (root_k - l1[nonzero_rows] / l2[nonzero_rows]) / (root_k - 1)
        np.clip(scores, 0.0, 1.0, out=scores)
    return scores, int(np.count_nonzero(~nonzero_rows))


def _bank_norm_factors(concept_bank):
    """Represent the bank row norms without forming a potentially overflowing norm."""
    _, scale, _, length = _concept_bank_row_parts(concept_bank)
    scale_mantissa, scale_exponent = np.frexp(scale)
    mantissa, exponent = np.frexp(scale_mantissa * length)
    return mantissa, exponent + scale_exponent


class HoyerSparsity:
    """Measure per-observation concentration of linear component magnitudes using
    Hoyer's normalized L1/L2 sparsity applied to ``abs(U) * norm(D)`` over all concept
    slots without thresholding. Equal nonzero magnitudes score ``0.0``, a single nonzero
    magnitude scores ``1.0``, and reciprocal rescaling of coefficients and atoms leaves
    the score unchanged.

    Parameters
    ----------
    batch_size : int or None
        Maximum flattened observations processed per chunk. ``None`` (default)
        processes each supplied array at once.

    Notes
    -----
    Zero-contribution observations and all observations with ``K == 1`` are undefined
    (``NaN``) and excluded from the reported mean. Zero concept vectors are allowed.
    Report ``HoyerSparsity`` alongside active support counts and reconstruction quality
    rather than treating higher sparsity as always better.

    References
    ----------
    Hoyer, P. O. (2004). Non-negative matrix factorization with sparseness
    constraints. Journal of Machine Learning Research, 5, 1457-1469.

    Examples
    --------
    >>> import numpy as np
    >>> from xplique.metrics.concepts import HoyerSparsity
    >>> bank = np.eye(4)
    >>> codes = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]])
    >>> HoyerSparsity()(codes, bank)
    0.5
    """

    def __init__(self, *, batch_size: Optional[int] = None):
        self.batch_size = validate_batch_size(batch_size)

    def _evaluate(
        self, batches: Iterable[np.ndarray], concept_bank: np.ndarray, *, details: bool = False
    ) -> Dict[str, Any]:
        norm_mantissa, norm_exponent = _bank_norm_factors(concept_bank)
        n_concepts = len(norm_mantissa)
        n_observations = n_scorable = n_zero = 0
        total = 0.0
        rows = []
        for coefficients in batches:
            values = array_input(coefficients, "coefficients")
            if values.shape[-1] != n_concepts:
                raise ValueError("Coefficient and concept-bank concept counts must agree.")
            for chunk in observation_chunks(values, self.batch_size):
                scores, n_zero_chunk = _hoyer_rows(chunk, norm_mantissa, norm_exponent)
                defined = np.isfinite(scores)
                total += float(scores[defined].sum())
                n_observations += len(scores)
                n_scorable += int(defined.sum())
                n_zero += n_zero_chunk
                if details:
                    rows.append(scores)
        if not n_observations:
            raise ValueError("An evaluation must contain observations.")
        report: Dict[str, Any] = {
            "hoyer_sparsity": total / n_scorable if n_scorable else np.nan,
            "n_observations": n_observations,
            "n_concepts": n_concepts,
            "n_scorable_observations": n_scorable,
            "n_zero_contribution_observations": n_zero,
        }
        if details:
            report["observation_hoyer_sparsity"] = np.concatenate(rows)
        return report

    def evaluate(self, coefficients: np.ndarray, concept_bank: np.ndarray) -> float:
        """Return the mean Hoyer sparsity over scorable observations. See
        ``detailed_evaluate`` for counts and per-observation Hoyer scores.

        Parameters
        ----------
        coefficients : np.ndarray
            Finite, nonempty real integer/floating array of shape ``(..., K)`` with at
            least two axes. Leading axes flatten into equally weighted observations.
        concept_bank : np.ndarray
            Finite, nonempty real integer/floating bank of shape ``(K, F)`` matching
            ``K``. Zero concept rows are allowed.

        Returns
        -------
        hoyer_sparsity : float
            Mean Hoyer sparsity in ``[0, 1]`` over scorable observations, or ``NaN`` if
            no observation is scorable (all-zero contributions or ``K == 1``).

        Raises
        ------
        ValueError
            If arrays have invalid dtypes/dimensions, are empty/nonfinite, or
            coefficient and bank concept counts differ.
        """
        return self.evaluate_batches((coefficients,), concept_bank=concept_bank)

    def __call__(self, coefficients: np.ndarray, concept_bank: np.ndarray) -> float:
        """Alias for ``evaluate``."""
        return self.evaluate(coefficients, concept_bank)

    def evaluate_batches(self, batches: Iterable[np.ndarray], *, concept_bank: np.ndarray) -> float:
        """Return the pooled observation mean in a single pass over streamed batches
        with O(K) evaluation-local state.

        Parameters
        ----------
        batches : iterable of np.ndarray
            One-pass iterable of finite, nonempty real integer/floating coefficient
            arrays of shape ``(..., K)`` with fixed ``K``.
        concept_bank : np.ndarray
            Finite, nonempty real integer/floating bank of shape ``(K, F)``, fixed
            for this evaluation. Zero rows are allowed.

        Returns
        -------
        hoyer_sparsity : float
            Mean Hoyer sparsity in ``[0, 1]`` over scorable observations (or ``NaN`` if
            none are scorable), matching ``evaluate`` on concatenated batches.

        Raises
        ------
        ValueError
            If the stream is empty, arrays have invalid dtypes/dimensions, are
            empty/nonfinite, or any batch's concept count differs from the bank.
        """
        return self._evaluate(batches, concept_bank)["hoyer_sparsity"]

    def detailed_evaluate(
        self, coefficients: np.ndarray, concept_bank: np.ndarray
    ) -> Dict[str, Any]:
        """Return Hoyer sparsity, observation counts, and per-observation scores.

        Parameters
        ----------
        coefficients : np.ndarray
            Finite, nonempty real integer/floating array of shape ``(..., K)`` with at
            least two axes.
        concept_bank : np.ndarray
            Finite, nonempty real integer/floating bank of shape ``(K, F)`` matching
            ``K``. Zero rows are allowed.

        Returns
        -------
        details : dict
            Dictionary of global and observation-level Hoyer sparsity fields:
            ``hoyer_sparsity`` (float): Mean Hoyer sparsity over scorable observations,
            matching ``evaluate`` (``NaN`` if none are scorable).
            ``n_observations`` (int): Total flattened observations evaluated.
            ``n_concepts`` (int): Concept count ``K``.
            ``n_scorable_observations`` (int): Observations with defined Hoyer score.
            ``n_zero_contribution_observations`` (int): Observations with all-zero
            linear contributions.
            ``observation_hoyer_sparsity`` (np.ndarray): Per-observation scores in
            flattened order, shape ``(N,)`` (``NaN`` for undefined observations).

        Raises
        ------
        ValueError
            If arrays have invalid dtypes/dimensions, are empty/nonfinite, or
            coefficient and bank concept counts differ.
        """
        return self._evaluate((coefficients,), concept_bank, details=True)
