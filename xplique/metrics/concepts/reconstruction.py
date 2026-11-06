"""Pooled reconstruction metrics with evaluation-local NumPy reductions."""

import numpy as np

from ...types import Optional
from ._utils import _Moments, array_input, observation_chunks, validate_batch_size


def _relative(residual_squared, target_squared):
    """Ratio of norms with explicit zero-target semantics."""
    residual_squared, target_squared = np.broadcast_arrays(residual_squared, target_squared)
    result = np.full(residual_squared.shape, np.inf)
    np.divide(residual_squared, target_squared, out=result, where=target_squared != 0)
    result[(target_squared == 0) & (residual_squared == 0)] = 0
    return np.sqrt(result)


def _reduce(batches, batch_size, centering=None, details=False):
    """Share array/streamed residual work; collect observation vectors only on request."""
    moments = _Moments()
    sse = sae = target_squared = None
    vectors = {key: [] for key in ("mse", "rmse", "mae", "relative_l2_error")}
    n_observations = 0
    n_features = None
    for activations, reconstructed in batches:
        x = array_input(activations, "activations")
        y = array_input(reconstructed, "reconstructed")
        if x.shape != y.shape:
            raise ValueError("Original reconstruction input shapes must match.")
        if n_features is not None and x.shape[-1] != n_features:
            raise ValueError("Feature counts must agree within an evaluation.")
        n_features = x.shape[-1]
        for xb, yb in zip(observation_chunks(x, batch_size), observation_chunks(y, batch_size)):
            residual = xb - yb
            squared = residual**2
            absolute = np.abs(residual)
            bsse = squared.sum(axis=0)
            bsae = absolute.sum(axis=0)
            btarget = (xb**2).sum(axis=0)
            if sse is None:
                sse, sae, target_squared = bsse, bsae, btarget
            else:
                sse += bsse
                sae += bsae
                target_squared += btarget
            n_observations += len(xb)
            if centering == "feature":
                moments.merge(xb)
            elif centering == "global":
                moments.merge(xb.reshape(-1, 1))
            if details:
                mse = squared.mean(axis=1)
                vectors["mse"].append(mse)
                vectors["rmse"].append(np.sqrt(mse))
                vectors["mae"].append(absolute.mean(axis=1))
                vectors["relative_l2_error"].append(
                    _relative(squared.sum(axis=1), (xb**2).sum(axis=1))
                )
    if n_observations == 0:
        raise ValueError("An evaluation must contain observations.")
    result = {
        "n_observations": n_observations,
        "n_features": n_features,
        "n_elements": n_observations * n_features,
        "sse": float(sse.sum()),
    }
    if centering is not None:
        result["sst"] = float(moments.centered.sum())
        result["feature_residual_sums"] = sse
        if centering == "feature":
            result["feature_centered_sums"] = moments.centered
    else:
        result.update(
            mse=float(sse.sum() / result["n_elements"]),
            mae=float(sae.sum() / result["n_elements"]),
            relative_l2_error=float(_relative(sse.sum(), target_squared.sum())),
        )
        result["rmse"] = float(np.sqrt(result["mse"]))
    if details:
        result.update({f"observation_{key}": np.concatenate(v) for key, v in vectors.items()})
        mse = result["observation_mse"]
        result.update(
            observation_mse_mean=float(mse.mean()),
            observation_mse_std=float(mse.std()),
            observation_mse_min=float(mse.min()),
            observation_mse_max=float(mse.max()),
            observation_mse_median=float(np.median(mse)),
        )
    return result


class ReconstructionR2:
    """Feature-centered pooled reconstruction R2 (not a mean of feature scores).

    Parameters
    ----------
    r2_centering : {'feature', 'global'}
        Center each feature across observations, or all scalar elements together.
    force_finite : bool
        At exactly zero SST, return 1 for exact reconstruction and 0 otherwise.
        Without this setting, return NaN in both cases. Negative R2 is permitted.
    batch_size : int or None
        Observations processed at once; None processes each supplied array at once.

    Notes
    -----
    With nonconstant features this equals sklearn's variance-weighted r2_score.
    Unlike sklearn it includes constant-feature residuals in pooled SSE. Global
    centering equals r2_score on flattened arrays when SST is positive.
    """

    def __init__(
        self, *, r2_centering="feature", force_finite=False, batch_size: Optional[int] = None
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

    def evaluate(self, activations, reconstructed) -> float:
        """Return pooled R2; see detailed_evaluate for its sufficient statistics."""
        return self.evaluate_batches(((activations, reconstructed),))

    def __call__(self, activations, reconstructed) -> float:
        """Delegate to evaluate with identical arguments."""
        return self.evaluate(activations, reconstructed)

    def evaluate_batches(self, batches) -> float:
        """Consume pairs once, returning pooled R2 with O(F) local state."""
        return self._finish(_reduce(batches, self.batch_size, self.r2_centering))["r2"]

    def detailed_evaluate(self, activations, reconstructed) -> dict:
        """Inspect the global and feature-level reduction, without observation R2.

        Returns
        -------
        dict
            ``r2`` reproduces evaluate (dimensionless, NaN at zero SST unless
            force_finite). ``sse`` and ``sst`` are global squared-activation sums.
            ``feature_residual_sums`` has shape (F,), in squared activation units;
            ``feature_centered_sums`` has shape (F,) and is present only for feature
            centering. All sums are zero for exact/constant inputs respectively.
            ``n_observations``, ``n_features``, ``n_elements`` are integer counts.
        """
        return self._finish(
            _reduce(((activations, reconstructed),), self.batch_size, self.r2_centering)
        )


class ReconstructionError:
    """Pooled MSE, RMSE, MAE, or relative Frobenius reconstruction error.

    Parameters
    ----------
    error : {'mse', 'rmse', 'mae', 'relative_l2'}
        Global reduction to return. Relative error is 0 for two zero arrays,
        infinity for a zero target and nonzero reconstruction.
    batch_size : int or None
        Observations processed at once; set for large activation sets.
    """

    def __init__(self, *, error="mse", batch_size: Optional[int] = None):
        if error not in ("mse", "rmse", "mae", "relative_l2"):
            raise ValueError("Unknown reconstruction error.")
        self.error = error
        self.batch_size = validate_batch_size(batch_size)

    @property
    def _key(self):
        return "relative_l2_error" if self.error == "relative_l2" else self.error

    def evaluate(self, activations, reconstructed) -> float:
        """Return the pooled error, not mean observation RMSE/relative error.

        See detailed_evaluate for observation vectors and pooled quantities.
        """
        return self.evaluate_batches(((activations, reconstructed),))

    def __call__(self, activations, reconstructed) -> float:
        """Delegate to evaluate."""
        return self.evaluate(activations, reconstructed)

    def evaluate_batches(self, batches) -> float:
        """Consume pairs once; never average individual batch errors."""
        return _reduce(batches, self.batch_size)[self._key]

    def detailed_evaluate(self, activations, reconstructed) -> dict:
        """Return opt-in observation errors and global reductions.

        Returns
        -------
        dict
            Global ``mse`` (squared units), ``rmse``/``mae`` (activation units),
            ``relative_l2_error`` (dimensionless) reproduce evaluate according to
            error, mapping relative_l2 to relative_l2_error. ``sse`` is a squared
            sum; n_observations/n_features/n_elements are integer counts.
            ``observation_mse``, ``observation_rmse``, ``observation_mae``, and
            ``observation_relative_l2_error`` have shape (N,), the same units and
            zero-target conventions as their global counterparts. Observation-MSE
            summaries ``observation_mse_mean/std/min/max/median`` are floats in
            squared units (population std, NumPy median). Exact errors are zero;
            relative errors may be infinity. Storage is O(N), not O(N F).
        """
        return _reduce(((activations, reconstructed),), self.batch_size, details=True)
