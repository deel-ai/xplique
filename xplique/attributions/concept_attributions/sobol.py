"""Total-order Sobol sensitivity for exact active concept channels."""

import tensorflow as tf

from ...types import Callable, OperatorSignature, Optional, Tuple, Union
from ..global_sensitivity_analysis.replicated_designs import ReplicatedSampler
from ..global_sensitivity_analysis.sobol_estimators import JansenEstimator
from .base import _check_integer, _ConceptChannelExplainer


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
    interaction_kind
        ``"pure"`` (default) reports isolated second-order variance indices from
        explain_interactions; ``"total"`` reports all variance components containing
        both channels. This does not affect explain() or its total-order singletons.

    Notes
    -----
    The existing Xplique Jansen estimator computes in float32, floors reference
    variance at 1e-12, and does not clip finite-sample indices to [0, 1]. Constant
    outputs produce zero indices. Returned values are unsigned total-order
    sensitivities, not signed Banzhaf effects or spatial attribution maps.
    Pair estimates use float64 covariance or squared mixed differences and
    add nb_design evaluations per requested active pair (except for two active
    channels, where the pair hybrid equals B). Pure finite-sample estimates
    may be negative; indices are not clipped.

    References
    ----------
    Sobol' (2001), "Global Sensitivity Indices for Nonlinear Mathematical Models
    and Their Monte Carlo Estimates", https://doi.org/10.1016/S0378-4754(00)00270-6.
    Jansen (1999), "Analysis of Variance Designs for Model Output",
    https://doi.org/10.1016/S0010-4655(98)00154-4.
    Fel et al. (2021), "Look at the Variance! Efficient Black-box Explanations
    with Sobol-based Sensitivity Analysis", https://arxiv.org/abs/2111.04138.
    Fel et al. (2023), "CRAFT: Concept Recursive Activation FacTorization for
    Explainability", https://arxiv.org/abs/2211.10154.
    """

    _supports_interactions = True

    def __init__(
        self,
        model: Callable,
        batch_size: Optional[int] = 32,
        operator: Optional[Union[str, OperatorSignature]] = None,
        nb_design: int = 32,
        mask_distribution: str = "uniform",
        seed: int = 0,
        *,
        interaction_kind: str = "pure",
    ):
        nb_design = _check_integer(
            nb_design, "nb_design must be an integer greater than or equal to two.", 2
        )
        if mask_distribution not in ("uniform", "bernoulli"):
            raise ValueError("mask_distribution must be either 'uniform' or 'bernoulli'.")
        if interaction_kind not in ("pure", "total"):
            raise ValueError("interaction_kind must be either 'pure' or 'total'.")
        super().__init__(model, batch_size, operator, seed)
        self.nb_design = nb_design
        self.mask_distribution = mask_distribution
        self.interaction_kind = interaction_kind
        self.estimator = JansenEstimator()

    def _sample_masks(self, nb_active: int, input_index: int) -> tf.Tensor:
        input_seed = self._input_seed(input_index)
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

    def _explain_interactions_input(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        single_input: tf.Tensor,
        single_target: tf.Tensor,
        input_index: int,
        active_ids: tf.Tensor,
        local_pairs: tf.Tensor,
        pair_batch_size: int,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Evaluate requested pair hybrids after the unchanged singleton design."""
        nb_active = int(tf.size(active_ids))
        masks = self._sample_masks(nb_active, input_index)
        outputs = self._evaluate_masks(single_input, single_target, active_ids, masks)
        main_effects = self._estimate(masks, outputs)
        if not int(tf.shape(local_pairs)[0]):
            return main_effects, tf.zeros([0], tf.float32)

        n = self.nb_design
        sampling_a = masks[:n]
        sampling_b = masks[n : 2 * n]
        scores_a = outputs[:n]
        scores_b = outputs[n : 2 * n]
        scores_c = tf.reshape(outputs[2 * n :], [nb_active, n])
        centered_a = scores_a - tf.reduce_mean(scores_a)
        variance = tf.maximum(
            tf.reduce_sum(tf.square(centered_a)) / (n - 1), tf.constant(1e-12, tf.float64)
        )
        centered_b = scores_b - tf.reduce_mean(scores_b)
        first_order = tf.reduce_sum(
            (scores_c - tf.reduce_mean(scores_c, axis=1, keepdims=True)) * centered_b[None],
            axis=1,
        ) / (n - 1)

        chunks = []
        for start in range(0, int(tf.shape(local_pairs)[0]), pair_batch_size):
            chunk = local_pairs[start : start + pair_batch_size]
            count = int(tf.shape(chunk)[0])
            if nb_active == 2:
                pair_outputs = tf.broadcast_to(scores_b[None], [count, n])
            else:
                selected = tf.reduce_sum(tf.one_hot(chunk, nb_active, dtype=tf.float32), axis=1)
                hybrids = sampling_a[None] + selected[:, None, :] * (sampling_b - sampling_a)[None]
                pair_outputs = self._evaluate_masks(
                    single_input,
                    single_target,
                    active_ids,
                    tf.reshape(hybrids, [count * n, nb_active]),
                )
                pair_outputs = tf.reshape(pair_outputs, [count, n])

            if self.interaction_kind == "pure":
                closed = tf.reduce_sum(
                    (pair_outputs - tf.reduce_mean(pair_outputs, axis=1, keepdims=True))
                    * centered_b[None],
                    axis=1,
                ) / (n - 1)
                pair_variance = closed - tf.gather(first_order, chunk[:, 0])
                pair_variance -= tf.gather(first_order, chunk[:, 1])
            else:
                mixed = scores_a[None] - tf.gather(scores_c, chunk[:, 0])
                mixed -= tf.gather(scores_c, chunk[:, 1])
                mixed += pair_outputs
                pair_variance = tf.reduce_mean(tf.square(mixed), axis=1) / 4.0
            chunks.append(tf.cast(pair_variance / variance, tf.float32))
        return main_effects, tf.concat(chunks, axis=0)
