"""Angular, spectral, Babel, and observed-support concept-bank diagnostics.

References
----------
Tropp (2004), Greed is good, IEEE Information Theory 50(10):2231–2242.
Welch (1974), Lower bounds on the maximum cross correlation of signals,
IEEE Information Theory 20(3):397–399.
"""

import numpy as np

from ...types import Optional
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


def concept_bank_babel_score(concept_bank, *, order: int) -> float:
    """Return maximum row sum of the order largest off-diagonal absolute cosines.

    Parameters
    ----------
    concept_bank : np.ndarray
        Nonempty (K,F) real bank with finite, nonzero concept vectors.
    order : int
        Integer in [0,K-1]; booleans and fractional values are rejected.

    Returns
    -------
    float
        Dimensionless cumulative coherence in [0,order], zero at order zero.
        Sign/scale invariant; order one is mutual coherence. No SVD is performed.
    """
    _, _, unit = concept_bank_input(concept_bank, True)
    order = _order(order, len(unit))
    if order == 0:
        return 0.0
    gram = _gram(unit)
    largest = np.partition(gram, len(unit) - order, axis=1)[:, -order:]
    return float(largest.sum(axis=1).max())


def concept_bank_babel_curve(concept_bank, *, max_order=None) -> dict:
    """Return the nondecreasing cumulative coherence curve without an SVD.

    Parameters
    ----------
    concept_bank : np.ndarray
        Nonempty (K,F) finite bank, with nonzero rows.
    max_order : int or None
        Last order in [0,K-1], default K-1.

    Returns
    -------
    dict
        n_concepts/n_features counts, requested_max_order (possibly None), resolved
        max_order, integer orders and float64 babel_curve arrays of shape (p+1,),
        endpoint babel_score and actual mutual_coherence. Scores are dimensionless,
        zero at order zero; mutual coherence is zero only for a one-row/orthogonal
        bank, even if the requested curve ends at zero. Curves are not in [0,1].
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
    concept_bank,
    *,
    spectrum="unit",
    energy_fraction=0.9,
    duplicate_threshold=0.8,
    rank_rtol=None,
) -> dict:
    """Describe angular redundancy, original norms, and thin-SVD geometry.

    Parameters
    ----------
    concept_bank : np.ndarray
        Nonempty finite (K,F) bank with nonzero rows.
    spectrum : {'unit', 'raw'}
        SVD of unit rows (scale invariant) or original bank (scale sensitive).
    energy_fraction : float
        Cumulative squared-singular-value energy target in (0,1], default 0.9.
    duplicate_threshold : float
        Strict signed pair-count threshold in [0,1], default 0.8.
    rank_rtol : float or None
        Relative singular-value tolerance in [0,1), default max(K,F)*float64 eps.
        Singular values at or below this relative tolerance are excluded from
        energy_rank; energy_fraction=1 returns this numerical rank exactly.

    Returns
    -------
    dict
        n_concepts/n_features counts and resolved configuration. Dimensionless
        mutual_coherence/mean_abs_cosine exclude diagonal, zero for K=1.
        positive_duplicate_count/antipodal_count count unordered signed pairs.
        norm_min/mean/max describe original rows in activation units.
        energy_rank is the smallest energy-target rank; stable_rank is squared
        Frobenius/spectral norm; energy_entropy_rank is exp(entropy of squared
        spectrum probabilities). condition_number uses the thin spectrum and is
        infinity at relative rank deficiency. welch_bound is the ambient-dimensional
        theoretical reference, zero if K<=F. No universal quality direction applies.

    Notes
    -----
    Only energy_rank thresholds and renormalizes the energy spectrum by rank_rtol.
    Fractions exactly on an interior cumulative boundary remain subject to float64
    roundoff. Stable and entropy ranks use the full spectrum; the condition number
    retains its existing rank_rtol cutoff.
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
    """Babel-style absolute coherence restricted to observed active supports.

    Parameters
    ----------
    activity_basis : {'contribution', 'coefficient'}
        Shared ConceptUsage support rule, default contribution.
    active_threshold : float
        Finite nonnegative strict activity threshold, default zero.
    batch_size : int or None
        Observations processed at once; does not bound O(K squared) Gram storage
        or O(N) opt-in exact quantiles. No persistent bank/state is retained.
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

    def _scores(self, coefficients, concept_bank):
        _, norms, unit = concept_bank_input(concept_bank, True)
        values = array_input(coefficients, "coefficients")
        if values.shape[-1] != len(unit):
            raise ValueError("Coefficient and bank concept counts must agree.")
        gram = _gram(unit)
        for chunk in observation_chunks(values, self.batch_size):
            active = activity_mask(chunk, norms, self.activity_basis, self.active_threshold)
            sizes = active.sum(axis=1)
            scores = np.zeros(len(chunk))
            for index in np.flatnonzero(sizes > 1):
                support = np.flatnonzero(active[index])
                scores[index] = gram[np.ix_(support, support)].sum(axis=1).max()
            yield scores, sizes

    def evaluate(self, coefficients, concept_bank) -> float:
        """Return unconditional mean observation coherence; see detailed_evaluate.

        Empty and singleton supports contribute zero, not missing observations.
        """
        total, count = 0.0, 0
        for scores, _ in self._scores(coefficients, concept_bank):
            total += scores.sum()
            count += len(scores)
        return float(total / count)

    def __call__(self, coefficients, concept_bank) -> float:
        """Delegate to evaluate."""
        return self.evaluate(coefficients, concept_bank)

    def detailed_evaluate(self, coefficients, concept_bank) -> dict:
        """Inspect exact observation distribution and support counts.

        Returns
        -------
        dict
            mean_support_babel reproduces evaluate, the mean of dimensionless
            observation_scores (N,). observation_support_sizes is an integer (N,)
            array. n_observations/n_concepts and empty_support_count/
            singleton_support_count/multi_support_count are integers. Mean/max
            support size measure active concepts. max_support_babel and
            support_babel_q25/median/q75/q95 summarize dimensionless scores using
            NumPy linear quantiles. mean_multi_support_babel conditions on size>=2,
            NaN if absent. Empty/singleton scores are zero. Activity configuration
            is included. Exact details require O(N) additional storage.
        """
        chunks = list(self._scores(coefficients, concept_bank))
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
