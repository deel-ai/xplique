"""Concept coefficient usage and strict-support sparsity."""

import numpy as np

from ...types import Optional
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
    """Multivariate signed coefficient and linear-component usage report.

    Parameters
    ----------
    activity_basis : {'contribution', 'coefficient'}
        Strict activity uses abs(U) times concept norms, or abs(U) alone.
    active_threshold : float
        Finite nonnegative threshold, default zero. Contribution activity,
        effective usage, and component energy are invariant to reciprocal
        concept/coefficient rescaling. Affine decoder offsets are excluded.
    batch_size : int or None
        Observations processed at once; None processes each array at once.

    Notes
    -----
    Component energies are not explained-variance shares: nonorthogonal linear
    components have cross-terms. Zero concept vectors are allowed.
    """

    def __init__(
        self,
        *,
        activity_basis="contribution",
        active_threshold=0.0,
        batch_size: Optional[int] = None,
    ):
        activity_config(activity_basis, active_threshold)
        self.activity_basis = activity_basis
        self.active_threshold = float(active_threshold)
        self.batch_size = validate_batch_size(batch_size)

    def evaluate(self, coefficients, concept_bank) -> dict:
        """Return pooled usage statistics, not an average of batch reports.

        Parameters
        ----------
        coefficients : np.ndarray
            Finite, nonempty real integer/floating array (..., K), at least two
            axes. Leading axes flatten into equally weighted observations.
        concept_bank : np.ndarray
            Finite, nonempty real integer/floating bank (K, F). Zero rows are
            allowed; K must match the coefficient count.

        Returns
        -------
        dict
            Integer n_observations/n_concepts/n_elements counts and activity_basis/
            active_threshold configuration. Concept-level arrays of shape (K,):
            mean_coefficient and std_coefficient (signed mean and population std,
            coefficient units), effective_usage (mean component norm, activation
            units), component_energy (total squared component norm),
            component_energy_share (dimensionless, all NaN at zero total energy),
            active_fraction (firing probability). n_active_concepts counts concepts
            firing at least once. sparsity is the global inactive fraction in [0,1].
            Zero contributions give zero effective usage/energy; no detailed API
            is provided because this report already contains the intermediates.

        Raises
        ------
        ValueError
            If arrays have invalid dtypes/dimensions, are empty/nonfinite, or
            coefficient and bank concept counts differ.
        """
        return self.evaluate_batches((coefficients,), concept_bank=concept_bank)

    def __call__(self, coefficients, concept_bank) -> dict:
        """Delegate to evaluate; see its Parameters, Returns and Raises sections."""
        return self.evaluate(coefficients, concept_bank)

    def evaluate_batches(self, batches, *, concept_bank) -> dict:
        """Consume coefficient batches once with a fixed bank and O(K) local state.

        Parameters
        ----------
        batches : iterable of np.ndarray
            Finite, nonempty real integer/floating coefficient arrays (..., K),
            with at least two axes and fixed K. All leading axes flatten into
            equally weighted observations across batches.
        concept_bank : np.ndarray
            Finite, nonempty real integer/floating bank (K, F), fixed for this
            evaluation. Zero rows are allowed.

        Returns
        -------
        dict
            Pooled concept/global usage report with the fields, units, shapes and
            zero-energy conventions of evaluate, not an average of batch reports.

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
    """Global inactive fraction using ConceptUsage's activity configuration.

    Parameters
    ----------
    activity_basis : {'contribution', 'coefficient'}
        Magnitude used for strict thresholding; contribution is rescaling-invariant.
    active_threshold : float
        Finite nonnegative activity threshold, default zero.
    batch_size : int or None
        Observations processed at once, default all observations.
    """

    def __init__(
        self,
        *,
        activity_basis="contribution",
        active_threshold=0.0,
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

    def evaluate(self, coefficients, concept_bank) -> float:
        """Return pooled inactive fraction; see detailed_evaluate for firing/supports.

        Parameters
        ----------
        coefficients : np.ndarray
            Finite, nonempty real integer/floating array (..., K), at least two
            axes. Leading axes flatten into equally weighted observations.
        concept_bank : np.ndarray
            Finite, nonempty real integer/floating bank (K, F), matching K.
            Zero concept vectors are allowed.

        Returns
        -------
        float
            Dimensionless pooled inactive fraction in [0,1], not an unweighted
            mean of batch fractions. Zero coefficients give 1.

        Raises
        ------
        ValueError
            If arrays have invalid dtypes/dimensions, are empty/nonfinite, or
            coefficient and bank concept counts differ.
        """
        return self.evaluate_batches((coefficients,), concept_bank=concept_bank)

    def __call__(self, coefficients, concept_bank) -> float:
        """Delegate to evaluate; see its Parameters, Returns and Raises sections."""
        return self.evaluate(coefficients, concept_bank)

    def evaluate_batches(self, batches, *, concept_bank) -> float:
        """Return pooled sparsity from one-pass coefficient batches.

        Parameters
        ----------
        batches : iterable of np.ndarray
            Finite, nonempty real integer/floating coefficient arrays (..., K),
            at least two axes and fixed K. Leading axes flatten into equally
            weighted observations across batches.
        concept_bank : np.ndarray
            Finite, nonempty real integer/floating bank (K, F). Zero rows allowed.

        Returns
        -------
        float
            Dimensionless pooled inactive fraction in [0,1], with the activity
            definition of evaluate. All-zero coefficients give 1.

        Raises
        ------
        ValueError
            If the stream is empty, arrays have invalid dtypes/dimensions, are
            empty/nonfinite, or any batch's concept count differs from the bank.
        """
        return self._statistics(batches, concept_bank)["sparsity"]

    def detailed_evaluate(self, coefficients, concept_bank) -> dict:
        """Inspect activity at concept and observation granularity.

        Parameters
        ----------
        coefficients : np.ndarray
            Finite, nonempty real integer/floating array (..., K), at least two
            axes. Leading axes flatten into equally weighted observations.
        concept_bank : np.ndarray
            Finite, nonempty real integer/floating bank (K, F), matching K.
            Zero concept vectors are allowed.

        Returns
        -------
        dict
            ``sparsity`` reproduces evaluate, a dimensionless inactive fraction.
            ``active_fraction`` is a (K,) firing-probability array;
            ``observation_support_sizes`` is an integer (N,) array of active counts.
            Zero coefficients give sparsity 1 and zero firing/supports. Integer
            n_observations/n_concepts/n_elements and activity configuration are
            included. No undefined quantities arise for valid nonempty inputs.

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
