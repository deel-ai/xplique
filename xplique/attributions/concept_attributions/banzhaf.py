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
