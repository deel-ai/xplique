"""Concept coefficient usage and strict-support sparsity."""

from collections.abc import Iterable

import numpy as np

from ...types import Any, Dict, Optional
from ._utils import (
    _Moments,
    activity_config,
    activity_mask,
    array_input,
    concept_bank_input,
    observation_chunks,
    validate_batch_size,
)


def _coefficient_statistics(batches, concept_bank, basis, threshold, batch_size, details=False):
    _, norms, _ = concept_bank_input(concept_bank)
    n_concepts = len(norms)
    moments = _Moments()
    absolute_sum = np.zeros(n_concepts)
    energy = np.zeros(n_concepts)
    active_counts = np.zeros(n_concepts, dtype=np.int64)
    supports = []
    for coefficients in batches:
        values = array_input(coefficients, "coefficients")
        if values.shape[-1] != n_concepts:
            raise ValueError("Coefficient and concept-bank concept counts must agree.")
        for chunk in observation_chunks(values, batch_size):
            moments.merge(chunk)
            contribution = np.abs(chunk) * norms
            absolute_sum += contribution.sum(axis=0)
            energy += (contribution**2).sum(axis=0)
            active = activity_mask(chunk, norms, basis, threshold)
            active_counts += active.sum(axis=0)
            if details:
                supports.append(active.sum(axis=1))
    if moments.count == 0:
        raise ValueError("An evaluation must contain observations.")
    result = {
        "n_observations": moments.count,
        "n_concepts": n_concepts,
        "n_elements": moments.count * n_concepts,
        "activity_basis": basis,
        "active_threshold": threshold,
        "mean_coefficient": moments.mean,
        "std_coefficient": np.sqrt(moments.centered / moments.count),
        "effective_usage": absolute_sum / moments.count,
        "component_energy": energy,
        "component_energy_share": (
            energy / energy.sum() if energy.sum() != 0 else np.full(n_concepts, np.nan)
        ),
        "active_fraction": active_counts / moments.count,
        "n_active_concepts": int(np.count_nonzero(active_counts)),
        "sparsity": float(1 - active_counts.sum() / (moments.count * n_concepts)),
    }
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
            details,
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
