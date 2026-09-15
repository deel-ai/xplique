"""Total-order Sobol sensitivity for exact active concept channels."""

from numbers import Integral

import numpy as np
import tensorflow as tf

from ...types import Callable, OperatorSignature, Optional, Union
from ..global_sensitivity_analysis.replicated_designs import ReplicatedSampler
from ..global_sensitivity_analysis.sobol_estimators import JansenEstimator
from .base import _ConceptChannelExplainer


class SparseSobol(_ConceptChannelExplainer):
    """Estimate total-order sensitivity over exact active concept channels.

    Each sampled value attenuates one whole channel across all positions. Only
    channels with at least one exactly nonzero coefficient participate in the
    replicated design; estimated indices are scattered back to the input shape.

    Parameters
    ----------
    model
        Model consuming already-encoded coefficients, optionally through a decoder.
    batch_size
        Maximum perturbations evaluated together. None evaluates the complete
        replicated design for each input in one call and is not memory-bounded.
    operator
        Xplique fixed-target operator returning finite scores of shape (B,) or (B, 1).
    nb_design
        Number of rows in each base design A and B, default 32. Must be at least
        two. An input with d active channels requires nb_design * (d + 2) scores.
    mask_distribution
        Intervention distribution. ``"uniform"`` continuously attenuates channels
        with values in [0, 1); ``"bernoulli"`` retains or removes whole channels.
        These distributions define different sensitivity games.
    seed
        Signed 64-bit stateless seed, folded with the input index and A/B stream.

    Notes
    -----
    The existing Xplique Jansen estimator computes in float32, floors reference
    variance at 1e-12, and does not clip finite-sample indices to [0, 1]. Constant
    outputs produce zero indices. Returned values are unsigned total-order
    sensitivities, not signed Banzhaf effects or spatial attribution maps.
    """

    def __init__(
        self,
        model: Callable,
        batch_size: Optional[int] = 32,
        operator: Optional[Union[str, OperatorSignature]] = None,
        nb_design: int = 32,
        mask_distribution: str = "uniform",
        seed: int = 0,
    ):
        if (
            isinstance(nb_design, (bool, np.bool_))
            or not isinstance(nb_design, Integral)
            or nb_design < 2
        ):
            raise ValueError("nb_design must be an integer greater than or equal to two.")
        if mask_distribution not in ("uniform", "bernoulli"):
            raise ValueError("mask_distribution must be either 'uniform' or 'bernoulli'.")
        super().__init__(model, batch_size, operator, seed)
        self.nb_design = int(nb_design)
        self.mask_distribution = mask_distribution
        self.estimator = JansenEstimator()

    def _sample_masks(self, nb_active: int, input_index: int) -> tf.Tensor:
        input_seed = tf.random.experimental.stateless_fold_in(
            tf.constant([self.seed, 0], dtype=tf.int64), tf.cast(input_index, tf.int64)
        )
        seed_a = tf.random.experimental.stateless_fold_in(input_seed, tf.constant(0, tf.int64))
        seed_b = tf.random.experimental.stateless_fold_in(input_seed, tf.constant(1, tf.int64))
        shape = [self.nb_design, nb_active]
        sampling_a = tf.random.stateless_uniform(shape, seed=seed_a, dtype=tf.float32)
        sampling_b = tf.random.stateless_uniform(shape, seed=seed_b, dtype=tf.float32)
        if self.mask_distribution == "bernoulli":
            sampling_a = tf.cast(sampling_a < 0.5, tf.float32)
            sampling_b = tf.cast(sampling_b < 0.5, tf.float32)
        replicated_c = ReplicatedSampler.build_replicated_design(sampling_a, sampling_b)
        return tf.concat([sampling_a, sampling_b, replicated_c], axis=0)

    def _estimate(self, masks: tf.Tensor, outputs: tf.Tensor) -> tf.Tensor:
        return self.estimator(masks, outputs, self.nb_design)
