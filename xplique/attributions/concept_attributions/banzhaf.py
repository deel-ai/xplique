"""Signed Banzhaf effects for exact active concept channels."""

from numbers import Integral

import numpy as np
import tensorflow as tf

from ...types import Callable, OperatorSignature, Optional, Union
from .base import _ConceptChannelExplainer


class Banzhaf(_ConceptChannelExplainer):
    """Estimate signed conditional-mean effects under uniform binary coalitions.

    Each active channel is retained or removed globally across all positions.
    Positive effects increase the fixed-target score on average; negative effects
    suppress it. Effects are in score units and need not sum to the full-versus-empty
    score difference. Balanced XOR or parity interactions can have zero effects.

    Parameters
    ----------
    model
        Model consuming already-encoded coefficients, optionally decoding them
        before prediction. Masked coefficients must not be re-encoded.
    batch_size
        Maximum perturbations evaluated together. None evaluates the entire design
        for each input in one call and is not memory-bounded.
    operator
        Standard Xplique operator returning finite scores of shape (B,) or (B, 1).
        Defaults to the standard fixed-target prediction operator.
    nb_samples
        Positive even evaluation budget per input. Enumerate all 2**d coalitions
        when they fit, where d is the exact active-channel count. Otherwise sample
        half this many independent masks and append their complements. Empty
        support requires no evaluations.
    seed
        Signed 64-bit integer seed for stateless sampling, folded with input index.
        Reproducibility depends on input order and grouping into explanation calls.
        Batch-size invariance requires deterministic, batch-independent inference.

    Notes
    -----
    Execution is eager. Inputs are sanitized to float32 before exact support
    detection. Returned effects have the input shape and are broadcast across
    positions; average rather than sum positions to recover channel effects.
    """

    def __init__(
        self,
        model: Callable,
        batch_size: Optional[int] = 32,
        operator: Optional[Union[str, OperatorSignature]] = None,
        nb_samples: int = 1024,
        seed: int = 0,
    ):
        if (
            isinstance(nb_samples, (bool, np.bool_))
            or not isinstance(nb_samples, Integral)
            or nb_samples <= 0
            or nb_samples % 2
        ):
            raise ValueError("nb_samples must be a positive even integer.")
        super().__init__(model, batch_size, operator, seed)
        self.nb_samples = int(nb_samples)

    def _sample_masks(self, nb_active: int, input_index: int) -> tf.Tensor:
        # Compare bit lengths first to avoid constructing 2**d for large supports.
        if nb_active < self.nb_samples.bit_length():
            rows = tf.range(2**nb_active, dtype=tf.int64)[:, None]
            bits = tf.range(nb_active, dtype=tf.int64)[None, :]
            return tf.cast(
                tf.bitwise.bitwise_and(tf.bitwise.right_shift(rows, bits), 1), tf.float32
            )

        seed = tf.random.experimental.stateless_fold_in(
            tf.constant([self.seed, 0], dtype=tf.int64), tf.cast(input_index, tf.int64)
        )
        half = tf.cast(
            tf.random.stateless_uniform([self.nb_samples // 2, nb_active], seed=seed) < 0.5,
            tf.float32,
        )
        return tf.concat([half, 1.0 - half], axis=0)

    def _estimate(self, masks: tf.Tensor, outputs: tf.Tensor) -> tf.Tensor:
        # Float64 centered scores reduce cancellation without changing the contrast.
        masks = tf.cast(masks, tf.float64)
        scores = tf.cast(outputs, tf.float64)
        scores -= tf.reduce_mean(scores)
        retained = tf.linalg.matvec(masks, scores, transpose_a=True)
        removed = tf.linalg.matvec(1.0 - masks, scores, transpose_a=True)
        effects = retained / tf.reduce_sum(masks, axis=0)
        effects -= removed / tf.reduce_sum(1.0 - masks, axis=0)
        return tf.cast(effects, tf.float32)


class KernelBanzhaf(Banzhaf):
    """Estimate signed Banzhaf effects with full-rank centered regression.

    Uses the same interventions and coalition designs as Banzhaf. Full enumeration
    agrees with conditional-mean effects for any game; sampled estimates can differ
    because balanced mask columns need not be orthogonal.

    Parameters
    ----------
    model
        Model consuming already-encoded coefficients, optionally through a decoder.
    batch_size
        Maximum perturbations per inference call, default 32. None evaluates the
        complete design for each input in one call.
    operator
        Xplique fixed-target operator returning finite scores of shape (B,) or (B, 1).
    nb_samples
        Positive even evaluation budget, default 1024. Enumerate all coalitions
        when they fit; otherwise sample half the budget and append complements.
        In sampled mode, at least twice the active dimension is necessary, but
        not sufficient, for full rank. No resampling or regularization is used.
    seed
        Signed 64-bit stateless seed, default 0, folded with the input index.

    Raises
    ------
    ValueError
        If the centered design is rank deficient. Checks precede inference for
        each input; previous inputs in the same call may already have been evaluated.

    Notes
    -----
    Rank uses the threshold eps(float64) * max(Q, d) * largest singular value.
    Regression uses float64 SVD without an intercept, ridge, or minimum-norm
    fallback. Final effects are float32, broadcast to the coefficient shape.
    Empty support returns zeros without inference. Execution is eager, and all
    support, target, batching, and reproducibility conventions of Banzhaf apply.
    """

    def _sample_masks(self, nb_active: int, input_index: int) -> tf.Tensor:
        sampled = nb_active >= self.nb_samples.bit_length()
        if sampled and nb_active > self.nb_samples // 2:
            raise ValueError(
                f"Active dimension {nb_active} exceeds the antithetic rank bound "
                f"{self.nb_samples // 2} for nb_samples={self.nb_samples}. "
                "Increase nb_samples to allow full-rank regression."
            )
        masks = super()._sample_masks(nb_active, input_index)
        centered = tf.cast(masks, tf.float64) - 0.5
        singular_values = tf.linalg.svd(centered, compute_uv=False)
        tolerance = np.finfo(np.float64).eps * max(centered.shape) * singular_values[0]
        rank = int(tf.reduce_sum(tf.cast(singular_values > tolerance, tf.int32)))
        if rank < nb_active:
            raise ValueError(
                f"Centered mask design has rank {rank}, below active dimension {nb_active}, "
                f"with nb_samples={self.nb_samples}. Increase nb_samples; "
                "exhaustive enumeration guarantees full rank."
            )
        return masks

    def _estimate(self, masks: tf.Tensor, outputs: tf.Tensor) -> tf.Tensor:
        centered = tf.cast(masks, tf.float64) - 0.5
        scores = tf.cast(outputs, tf.float64)
        scores -= tf.reduce_mean(scores)
        # Recompute locally rather than caching per-input factors on the explainer.
        singular_values, left, right = tf.linalg.svd(centered, full_matrices=False)
        projected = tf.linalg.matvec(left, scores, transpose_a=True)
        effects = tf.linalg.matvec(right, projected / singular_values)
        return tf.cast(effects, tf.float32)
