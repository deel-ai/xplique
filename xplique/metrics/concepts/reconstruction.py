"""Pooled reconstruction metrics with evaluation-local NumPy reductions."""

from collections.abc import Iterable

import numpy as np

from ...types import Any, Dict, Optional, Tuple
from ._utils import _Moments, array_input, observation_chunks, validate_batch_size


def _relative(residual_squared, target_squared):
    """Ratio of norms with explicit zero-target semantics."""
    residual_squared, target_squared = np.broadcast_arrays(residual_squared, target_squared)
    # Take square roots before dividing so representable norm ratios do not
    # overflow when squared sums span more than float64 max (~1e308).
    residual_norm = np.sqrt(residual_squared)
    target_norm = np.sqrt(target_squared)
    result = np.full(residual_norm.shape, np.inf)
    np.divide(residual_norm, target_norm, out=result, where=target_norm != 0)
    result[(target_norm == 0) & (residual_norm == 0)] = 0.0
    return result


def _validate_pair(pair, n_features):
    """Validate one (activations, reconstructed) batch pair."""
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise ValueError(
            "Expected an iterable of pairs: [(activations, reconstructed), ...], "
            "not a single (activations, reconstructed) pair."
        )
    activations, reconstructed = pair
    x = array_input(activations, "activations")
    y = array_input(reconstructed, "reconstructed")
    if x.shape != y.shape:
        raise ValueError("Original reconstruction input shapes must match.")
    if n_features is not None and x.shape[-1] != n_features:
        raise ValueError("Feature counts must agree within an evaluation.")
    return x, y


def _reduce(batches, batch_size, centering=None, error=None, details=False):
    """Share array/streamed residual work; compute only requested reductions."""
    need_sse = details or centering is not None or error in (None, "mse", "rmse", "relative_l2")
    need_sae = details or (centering is None and error in (None, "mae"))
    need_target = details or (centering is None and error in (None, "relative_l2"))
    moments = _Moments()
    sse: Optional[np.ndarray] = None
    sae: Optional[np.ndarray] = None
    target_squared: Optional[np.ndarray] = None
    vectors: Dict[str, list] = {key: [] for key in ("mse", "rmse", "mae", "relative_l2_error")}
    n_observations = 0
    n_features: Optional[int] = None
    for pair in batches:
        x, y = _validate_pair(pair, n_features)
        n_features = x.shape[-1]
        for xb, yb in zip(observation_chunks(x, batch_size), observation_chunks(y, batch_size)):
            residual = xb - yb
            squared = residual**2 if need_sse else None
            absolute = np.abs(residual) if need_sae else None
            x_squared = xb**2 if need_target else None
            if squared is not None:
                bsse = squared.sum(axis=0)
                if sse is None:
                    sse = bsse
                else:
                    sse += bsse
            if absolute is not None:
                bsae = absolute.sum(axis=0)
                if sae is None:
                    sae = bsae
                else:
                    sae += bsae
            if x_squared is not None:
                btarget = x_squared.sum(axis=0)
                if target_squared is None:
                    target_squared = btarget
                else:
                    target_squared += btarget
            n_observations += len(xb)
            if centering == "feature":
                moments.merge(xb)
            elif centering == "global":
                moments.merge(xb.reshape(-1, 1))
            if details:
                assert squared is not None and absolute is not None and x_squared is not None
                mse = squared.mean(axis=1)
                vectors["mse"].append(mse)
                vectors["rmse"].append(np.sqrt(mse))
                vectors["mae"].append(absolute.mean(axis=1))
                vectors["relative_l2_error"].append(
                    _relative(squared.sum(axis=1), x_squared.sum(axis=1))
                )
    if n_observations == 0:
        raise ValueError("An evaluation must contain observations.")
    assert n_features is not None
    n_elements = n_observations * n_features
    result: Dict[str, Any] = {
        "n_observations": n_observations,
        "n_features": n_features,
        "n_elements": n_elements,
    }
    if sse is not None:
        result["sse"] = float(sse.sum())
    if centering is not None:
        assert sse is not None and moments.centered is not None
        result["sst"] = float(moments.centered.sum())
        result["feature_residual_sums"] = sse
        if centering == "feature":
            result["feature_centered_sums"] = moments.centered
    else:
        if sse is not None:
            mse_scalar = float(sse.sum() / n_elements)
            result["mse"] = mse_scalar
            result["rmse"] = float(np.sqrt(mse_scalar))
        if sae is not None:
            result["mae"] = float(sae.sum() / n_elements)
        if sse is not None and target_squared is not None:
            result["relative_l2_error"] = float(_relative(sse.sum(), target_squared.sum()))
    if details:
        result.update({f"observation_{key}": np.concatenate(v) for key, v in vectors.items()})
        obs_mse = result["observation_mse"]
        result.update(
            observation_mse_mean=float(obs_mse.mean()),
            observation_mse_std=float(obs_mse.std()),
            observation_mse_min=float(obs_mse.min()),
            observation_mse_max=float(obs_mse.max()),
            observation_mse_median=float(np.median(obs_mse)),
        )
    return result


class ReconstructionR2:
    """Pooled reconstruction $R^2$ measuring how much activation variance a concept
    representation preserves. Higher values (closer to ``1.0``) indicate better
    reconstruction; ``0.0`` matches predicting the mean activation and negative values
    indicate worse-than-mean reconstruction.

    Parameters
    ----------
    r2_centering : {'feature', 'global'}
        Centering strategy for total sum of squares (SST). ``'feature'`` (default)
        centers each feature across observations; ``'global'`` centers all scalar
        elements around a single grand mean.
    force_finite : bool
        Behavior when SST is exactly zero (constant activations or a single
        observation with feature centering). When ``False`` (default), returns
        ``NaN``; when ``True``, returns ``1.0`` for exact reconstruction and ``0.0``
        otherwise.
    batch_size : int or None
        Number of flattened observations processed per chunk. ``None`` (default)
        processes each supplied array at once.

    Notes
    -----
    With nonconstant features, ``r2_centering='feature'`` matches scikit-learn's
    ``r2_score(..., multioutput='variance_weighted')`` while still retaining residuals
    from constant features in the pooled SSE. Global centering matches ``r2_score`` on
    flattened arrays whenever SST is positive.

    Squared sums are accumulated in float64 without rescaling. Extremely large finite
    inputs (beyond roughly ``1e150`` in magnitude) can overflow to ``inf``/``NaN``;
    rescale inputs first if needed.

    Examples
    --------
    >>> import numpy as np
    >>> from xplique.metrics.concepts import ReconstructionR2
    >>> x = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    >>> ReconstructionR2()(x, x + 0.1)  # doctest: +ELLIPSIS
    0.97...
    """

    def __init__(
        self,
        *,
        r2_centering: str = "feature",
        force_finite: bool = False,
        batch_size: Optional[int] = None,
    ):
        if r2_centering not in ("feature", "global"):
            raise ValueError("r2_centering must be 'feature' or 'global'.")
        if not isinstance(force_finite, (bool, np.bool_)):
            raise ValueError("force_finite must be boolean.")
        self.r2_centering = r2_centering
        self.force_finite = force_finite
        self.batch_size = validate_batch_size(batch_size)

    def _finish(self, result):
        sse, sst = result["sse"], result["sst"]
        result["r2"] = (
            float(1 - sse / sst)
            if sst != 0
            else (float(sse == 0) if self.force_finite else float("nan"))
        )
        return result

    def evaluate(self, activations: np.ndarray, reconstructed: np.ndarray) -> float:
        """Compute pooled $R^2$ across all observations (not an average of per-feature
        or per-observation scores). See ``detailed_evaluate`` for the underlying sums
        and counts.

        Parameters
        ----------
        activations : np.ndarray
            Reference activations with real integer/floating dtype, finite values,
            and shape ``(..., F)`` (at least 2 axes, non-empty). Leading axes are
            flattened into equally weighted observations.
        reconstructed : np.ndarray
            Reconstructed activations with the exact same shape as ``activations``.

        Returns
        -------
        r2 : float
            Dimensionless pooled $R^2$ score (at most ``1.0``; negative values are
            permitted). Returns ``NaN`` at zero SST unless ``force_finite=True``.

        Raises
        ------
        ValueError
            If shapes differ, arrays have fewer than two axes, are empty, contain
            nonfinite values, or do not have real integer/floating dtypes.
        """
        return self.evaluate_batches(((activations, reconstructed),))

    def __call__(self, activations: np.ndarray, reconstructed: np.ndarray) -> float:
        """Alias for ``evaluate``."""
        return self.evaluate(activations, reconstructed)

    def evaluate_batches(self, batches: Iterable[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Compute exact pooled $R^2$ in a single pass over streamed pairs using O(F)
        local memory, without averaging individual batch scores.

        Parameters
        ----------
        batches : iterable of (np.ndarray, np.ndarray)
            One-pass iterable yielding ``(activations, reconstructed)`` tuple or list
            pairs, each with matching shape ``(..., F)`` and constant feature count
            ``F`` across batches. Pass ``[(x, y)]``, not ``(x, y)``.

        Returns
        -------
        r2 : float
            Dimensionless $R^2$ pooled across all observations in the stream, matching
            ``evaluate`` on the concatenated arrays up to float64 roundoff.

        Raises
        ------
        ValueError
            If any item is not a length-2 tuple/list, the stream is empty, feature
            counts change between batches, or an array violates ``evaluate`` rules.
        """
        return self._finish(_reduce(batches, self.batch_size, self.r2_centering))["r2"]

    def detailed_evaluate(
        self, activations: np.ndarray, reconstructed: np.ndarray
    ) -> Dict[str, Any]:
        """Return pooled $R^2$ together with global and per-feature residual and
        centered variance sums.

        Parameters
        ----------
        activations : np.ndarray
            Reference activations with finite real integer/floating values and shape
            ``(..., F)`` (at least 2 axes).
        reconstructed : np.ndarray
            Reconstructed activations with the exact same shape as ``activations``.

        Returns
        -------
        details : dict
            Dictionary containing global and per-feature sufficient statistics:
            ``r2`` (float): Pooled $R^2$, matching ``evaluate``.
            ``sse`` (float): Total sum of squared residuals in squared activation units.
            ``sst`` (float): Total centered sum of squares in squared activation units.
            ``feature_residual_sums`` (np.ndarray): Per-feature SSE of shape ``(F,)``.
            ``feature_centered_sums`` (np.ndarray): Per-feature SST of shape ``(F,)``
            (present only when ``r2_centering='feature'``).
            ``n_observations`` (int): Number of flattened observations.
            ``n_features`` (int): Feature count ``F``.
            ``n_elements`` (int): Total scalar elements ``n_observations * n_features``.

        Raises
        ------
        ValueError
            If arrays have invalid dtypes/dimensions, differ in shape, are empty, or
            contain nonfinite values.
        """
        return self._finish(
            _reduce(((activations, reconstructed),), self.batch_size, self.r2_centering)
        )


class ReconstructionError:
    """Pooled MSE, RMSE, MAE, or relative L2 (Frobenius) reconstruction error. Lower
    values indicate closer activation reconstruction, with ``0.0`` corresponding to
    exact reconstruction.

    Parameters
    ----------
    error : {'mse', 'rmse', 'mae', 'relative_l2'}
        Error reduction returned by ``evaluate`` (default ``'mse'``). Relative L2 is
        ``0.0`` for two zero arrays and ``inf`` for a zero target with a nonzero
        reconstruction.
    batch_size : int or None
        Number of flattened observations processed per chunk. Set this for large
        activation sets to bound temporary memory.

    Notes
    -----
    Pooled RMSE and relative L2 are computed from global squared sums, not by
    averaging observation-level RMSE or relative errors. Squared sums use float64
    without rescaling; extremely large finite inputs (beyond roughly ``1e150``) can
    overflow to ``inf``/``NaN``.

    Examples
    --------
    >>> import numpy as np
    >>> from xplique.metrics.concepts import ReconstructionError
    >>> x = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> ReconstructionError(error="rmse")(x, x + 0.5)
    0.5
    """

    def __init__(self, *, error: str = "mse", batch_size: Optional[int] = None):
        if error not in ("mse", "rmse", "mae", "relative_l2"):
            raise ValueError("Unknown reconstruction error.")
        self.error = error
        self.batch_size = validate_batch_size(batch_size)

    @property
    def _key(self):
        return "relative_l2_error" if self.error == "relative_l2" else self.error

    def evaluate(self, activations: np.ndarray, reconstructed: np.ndarray) -> float:
        """Compute the selected pooled reconstruction error across all observations.
        See ``detailed_evaluate`` for all error types and per-observation distributions.

        Parameters
        ----------
        activations : np.ndarray
            Reference activations with finite real integer/floating values and shape
            ``(..., F)`` (at least 2 axes). Leading axes flatten into equally weighted
            observations.
        reconstructed : np.ndarray
            Reconstructed activations with the exact same shape as ``activations``.

        Returns
        -------
        error : float
            Nonnegative pooled error: MSE (squared activation units), RMSE or MAE
            (activation units), or relative L2 error (dimensionless; ``0.0`` for two
            zero arrays and ``inf`` for a zero target with nonzero reconstruction).

        Raises
        ------
        ValueError
            If arrays differ in shape, have invalid dtypes/dimensions, are empty, or
            contain nonfinite values.
        """
        return self.evaluate_batches(((activations, reconstructed),))

    def __call__(self, activations: np.ndarray, reconstructed: np.ndarray) -> float:
        """Alias for ``evaluate``."""
        return self.evaluate(activations, reconstructed)

    def evaluate_batches(self, batches: Iterable[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Compute the selected pooled error in one pass over streamed pairs, without
        averaging batch-level errors.

        Parameters
        ----------
        batches : iterable of (np.ndarray, np.ndarray)
            One-pass iterable of ``(activations, reconstructed)`` tuple/list pairs
            with matching shape ``(..., F)`` and fixed ``F``. Pass ``[(x, y)]``, not
            ``(x, y)``.

        Returns
        -------
        error : float
            Selected pooled error across all streamed observations, matching
            ``evaluate`` on the concatenated arrays up to float64 roundoff.

        Raises
        ------
        ValueError
            If an item is not a length-2 tuple/list, the stream is empty, feature
            counts change, or arrays violate ``evaluate`` input rules.
        """
        return _reduce(batches, self.batch_size, error=self.error)[self._key]

    def detailed_evaluate(
        self, activations: np.ndarray, reconstructed: np.ndarray
    ) -> Dict[str, Any]:
        """Return all pooled reconstruction errors together with opt-in O(N)
        per-observation error vectors and MSE summary statistics.

        Parameters
        ----------
        activations : np.ndarray
            Reference activations with finite real integer/floating values and shape
            ``(..., F)`` (at least 2 axes).
        reconstructed : np.ndarray
            Reconstructed activations with the exact same shape as ``activations``.

        Returns
        -------
        details : dict
            Dictionary containing pooled scalars, observation vectors, and summaries:
            ``mse`` (float): Pooled mean squared error (squared activation units).
            ``rmse`` (float): Pooled root mean squared error (activation units).
            ``mae`` (float): Pooled mean absolute error (activation units).
            ``relative_l2_error`` (float): Pooled Frobenius norm ratio (dimensionless);
            matches ``evaluate`` when ``error='relative_l2'``.
            ``sse`` (float): Global sum of squared residuals.
            ``n_observations``, ``n_features``, ``n_elements`` (int): Element counts.
            ``observation_mse``, ``observation_rmse``, ``observation_mae``,
            ``observation_relative_l2_error`` (np.ndarray): Per-observation arrays of
            shape ``(N,)`` following the same units and zero-target rules.
            ``observation_mse_mean``, ``observation_mse_std``, ``observation_mse_min``,
            ``observation_mse_max``, ``observation_mse_median`` (float): Summary
            statistics of ``observation_mse`` (population std, exact NumPy median).

        Raises
        ------
        ValueError
            If arrays have invalid dtypes/dimensions, differ in shape, are empty, or
            contain nonfinite values.
        """
        return _reduce(((activations, reconstructed),), self.batch_size, details=True)
