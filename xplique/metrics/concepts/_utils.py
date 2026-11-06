"""Small NumPy input and activity primitives for concept metrics."""

import numpy as np


def _is_real_scalar(value):
    """Accept integer/floating scalars, but not boolean settings."""
    return not isinstance(value, (bool, np.bool_)) and isinstance(
        value, (int, float, np.integer, np.floating)
    )


def validate_batch_size(batch_size):
    """Validate an optional observation chunk size."""
    if batch_size is not None and (
        isinstance(batch_size, (bool, np.bool_))
        or not isinstance(batch_size, (int, np.integer))
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer or None.")
    return batch_size


def array_input(values, name, bank=False):
    """Check metadata without making a full float64 copy or finiteness mask."""
    values = np.asarray(values)
    if values.dtype.kind not in "iuf" or values.size == 0:
        raise ValueError(f"{name} must be a nonempty real integer/floating array.")
    if (bank and values.ndim != 2) or (not bank and values.ndim < 2):
        raise ValueError(f"{name} has invalid dimensions: {values.shape}.")
    return values


def observation_chunks(values, batch_size):
    """Yield bounded float64 chunks, including for arbitrary strided arrays.

    C-contiguous inputs use flattened views. Other layouts gather only the current
    observation chunk, in C observation order, rather than copying a full reshape.
    """
    n_features = values.shape[-1]
    n_observations = values.size // n_features
    step = n_observations if batch_size is None else batch_size
    flat = values.reshape(-1, n_features) if values.flags.c_contiguous else None
    for start in range(0, n_observations, step):
        stop = min(start + step, n_observations)
        if flat is not None:
            chunk = flat[start:stop]
        else:
            indices = np.unravel_index(np.arange(start, stop), values.shape[:-1])
            chunk = values[indices]
        chunk = np.asarray(chunk, dtype=np.float64)
        if not np.isfinite(chunk).all():
            raise ValueError("Inputs must contain only finite float64-representable values.")
        yield chunk


def concept_bank_input(concept_bank, nonzero=False):
    """Return a validated bank, scale-safe row norms, and unit rows."""
    bank = array_input(concept_bank, "concept_bank", bank=True).astype(np.float64)
    if not np.isfinite(bank).all():
        raise ValueError("concept_bank must be finite.")
    scale = np.max(np.abs(bank), axis=1)
    if nonzero and np.any(scale == 0):
        raise ValueError("Angular metrics are undefined for zero concept vectors.")
    safe_scale = np.where(scale == 0, 1.0, scale)
    scaled = bank / safe_scale[:, None]
    length = np.sqrt(np.sum(scaled * scaled, axis=1))
    unit = scaled / np.where(length == 0, 1.0, length)[:, None]
    return bank, scale * length, unit


def activity_config(activity_basis, active_threshold):
    """Validate the common strict activity definition."""
    if activity_basis not in ("contribution", "coefficient"):
        raise ValueError("activity_basis must be 'contribution' or 'coefficient'.")
    if not _is_real_scalar(active_threshold):
        raise ValueError("active_threshold must be finite and nonnegative.")
    if not np.isfinite(active_threshold) or active_threshold < 0:
        raise ValueError("active_threshold must be finite and nonnegative.")


def activity_mask(coefficients, norms, basis, threshold):
    """Return the shared usage/coherence support mask."""
    magnitude = np.abs(coefficients)
    if basis == "contribution":
        magnitude = magnitude * norms
    return magnitude > threshold


class _Moments:
    """Evaluation-local, mergeable central moments with shifted batch means."""

    def __init__(self):
        self.count = 0
        self.mean = None
        self.centered = None

    def merge(self, values):
        """Merge observations without subtracting large raw squared moments."""
        count = len(values)
        shifted = values - values[0]
        shift_mean = shifted.mean(axis=0)
        mean = values[0] + shift_mean
        centered = np.sum((shifted - shift_mean) ** 2, axis=0)
        if self.count == 0:
            self.mean, self.centered = mean, centered
        else:
            delta = mean - self.mean
            total = self.count + count
            self.centered += centered + delta**2 * (self.count * count / total)
            self.mean += delta * (count / total)
        self.count += count
