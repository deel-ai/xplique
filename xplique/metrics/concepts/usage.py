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
    defined = nonzero.any(axis=1)
    scores = np.full(len(chunk), np.nan)
    if n_concepts > 1:
        root_k = np.sqrt(n_concepts)
        l1 = scaled.sum(axis=1)
        l2 = np.sqrt(np.square(scaled).sum(axis=1))
        scores[defined] = (root_k - l1[defined] / l2[defined]) / (root_k - 1)
        np.clip(scores, 0.0, 1.0, out=scores)
    return scores, int(np.count_nonzero(~defined))


def _bank_norm_factors(concept_bank):
    """Represent the bank row norms without forming a potentially overflowing norm."""
    _, scale, _, length = _concept_bank_row_parts(concept_bank)
    scale_mantissa, scale_exponent = np.frexp(scale)
    mantissa, exponent = np.frexp(scale_mantissa * length)
    return mantissa, exponent + scale_exponent


class HoyerSparsity:
    """Measure per-observation sparsity of linear component magnitudes.

    For each observation, apply Hoyer's normalized L1/L2 measure to
    ``abs(U) * norm(D)`` over all concept slots, without thresholding.
    Equal nonzero magnitudes give 0 and a single nonzero magnitude gives 1.
    Reciprocal rescaling of coefficients and atoms does not change the score.

    Parameters
    ----------
    batch_size : int or None
        Maximum flattened observations processed per chunk. ``None`` processes
        each supplied array at once.

    References
    ----------
    Hoyer, P. O. (2004). Non-negative matrix factorization with sparseness
    constraints. Journal of Machine Learning Research, 5, 1457-1469.
    """

    def __init__(self, *, batch_size: Optional[int] = None):
        self.batch_size = validate_batch_size(batch_size)

    def _evaluate(self, batches, concept_bank, *, details=False):
        norm_mantissa, norm_exponent = _bank_norm_factors(concept_bank)
        n_observations = n_scorable = n_zero = 0
        total = 0.0
        rows = []
        for coefficients in batches:
            values = array_input(coefficients, "coefficients")
            if values.shape[-1] != len(norm_mantissa):
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
        report = {
            "hoyer_sparsity": total / n_scorable if n_scorable else np.nan,
            "n_observations": n_observations,
            "n_scorable_observations": n_scorable,
            "n_zero_contribution_observations": n_zero,
        }
        if details:
            report["observation_hoyer_sparsity"] = np.concatenate(rows)
        return report

    def evaluate(self, coefficients: np.ndarray, concept_bank: np.ndarray) -> float:
        """Return the mean over observations with defined Hoyer sparsity.

        Parameters
        ----------
        coefficients : np.ndarray
            Finite array of shape ``(..., K)`` with at least two axes. Leading
            axes are flattened into equally weighted observations.
        concept_bank : np.ndarray
            Finite array of shape ``(K, F)``. Zero concept rows are allowed.

        Returns
        -------
        float
            Mean in ``[0, 1]``, or ``NaN`` if no row is scorable. All-zero
            contribution rows and all rows with ``K == 1`` are undefined.
        """
        return self.evaluate_batches((coefficients,), concept_bank=concept_bank)

    __call__ = evaluate

    def evaluate_batches(self, batches: Iterable[np.ndarray], *, concept_bank: np.ndarray) -> float:
        """Return the pooled observation mean from a one-pass stream.

        Parameters
        ----------
        batches : iterable of np.ndarray
            Coefficient arrays as in ``evaluate``, with the same concept width.
        concept_bank : np.ndarray
            Fixed concept bank for all batches.

        Returns
        -------
        float
            Mean over scorable observations, or ``NaN`` if none are scorable.
        """
        return self._evaluate(batches, concept_bank)["hoyer_sparsity"]

    def detailed_evaluate(self, coefficients: np.ndarray, concept_bank: np.ndarray) -> dict:
        """Return Hoyer sparsity, counts, and individual observation scores.

        Parameters
        ----------
        coefficients, concept_bank : np.ndarray
            Inputs as in ``evaluate``.

        Returns
        -------
        dict
            ``hoyer_sparsity``, ``n_observations``, ``n_scorable_observations``,
            ``n_zero_contribution_observations``, and ``observation_hoyer_sparsity``
            in flattened observation order. Undefined rows are ``NaN``.
        """
        return self._evaluate((coefficients,), concept_bank, details=True)
