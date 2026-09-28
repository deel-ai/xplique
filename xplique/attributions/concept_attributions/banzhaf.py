"""Signed Banzhaf effects for exact active concept channels."""

import numpy as np
import tensorflow as tf

from ...types import Callable, OperatorSignature, Optional, Tuple, Union
from .base import _bernoulli, _check_integer, _ConceptChannelExplainer


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
    explain_interactions reports signed Banzhaf mixed differences for each active
    pair. Enumeration gives exact uniform-coalition averages. In sampled mode,
    an unbiased covariance estimator treats each mask and its complement as one
    independent group; at least two groups are required for requested pairs.

    Execution is eager. Inputs are sanitized to float32 before exact support
    detection. Returned effects have the input shape and are broadcast across
    positions; average rather than sum positions to recover channel effects.

    References
    ----------
    Banzhaf (1965), "Weighted Voting Doesn't Work: A Mathematical Analysis";
    Dubey and Shapley (1979), "Mathematical Properties of the Banzhaf Power
    Index", https://doi.org/10.1287/moor.4.2.99.
    Wang and Jia (2023), "Data Banzhaf: A Robust Data Valuation Framework for
    Machine Learning", https://proceedings.mlr.press/v206/wang23e.html.
    Staudacher and Pollmann (2023), "Assessing Antithetic Sampling for
    Approximating Shapley, Banzhaf, and Owen Values",
    https://doi.org/10.3390/appliedmath3040049.
    """

    _supports_interactions = True

    def __init__(
        self,
        model: Callable,
        batch_size: Optional[int] = 32,
        operator: Optional[Union[str, OperatorSignature]] = None,
        nb_samples: int = 1024,
        seed: int = 0,
    ):
        nb_samples = _check_integer(
            nb_samples, "nb_samples must be a positive even integer.", 1, even=True
        )
        super().__init__(model, batch_size, operator, seed)
        self.nb_samples = nb_samples

    def _sample_masks(self, nb_active: int, input_index: int) -> tf.Tensor:
        # Compare bit lengths first to avoid constructing 2**d for large supports.
        if nb_active < self.nb_samples.bit_length():
            rows = tf.range(2**nb_active, dtype=tf.int64)[:, None]
            bits = tf.range(nb_active, dtype=tf.int64)[None, :]
            return tf.cast(
                tf.bitwise.bitwise_and(tf.bitwise.right_shift(rows, bits), 1), tf.float32
            )

        half = _bernoulli([self.nb_samples // 2, nb_active], self._input_seed(input_index))
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

    def _explain_interactions_input(  # pylint: disable=too-many-arguments
        self,
        single_input: tf.Tensor,
        single_target: tf.Tensor,
        input_index: int,
        active_ids: tf.Tensor,
        local_pairs: tf.Tensor,
        pair_batch_size: int,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Reject sampled pair estimates with fewer than two independent groups."""
        nb_active = int(tf.size(active_ids))
        if (
            int(tf.shape(local_pairs)[0])
            and nb_active >= self.nb_samples.bit_length()
            and self.nb_samples < 4
        ):
            raise ValueError("Sampled Banzhaf interactions require nb_samples >= 4.")
        return super()._explain_interactions_input(
            single_input, single_target, input_index, active_ids, local_pairs, pair_batch_size
        )

    def _prepare_interactions(self, masks, outputs, evaluate):
        """Share the evaluated design between singleton and pair estimates."""
        del evaluate
        return self._estimate(masks, outputs), (masks, outputs)

    def _estimate_pair_chunk(self, state, local_pairs: tf.Tensor) -> tf.Tensor:
        """Compute signed mixed effects using the current coalition evaluations."""
        masks, outputs = state
        signs = 2.0 * tf.cast(masks, tf.float64) - 1.0
        pair_signs = tf.gather(signs, local_pairs[:, 0], axis=1) * tf.gather(
            signs, local_pairs[:, 1], axis=1
        )
        scores = tf.cast(outputs, tf.float64)
        scores -= tf.reduce_mean(scores)
        if int(tf.shape(masks)[1]) < self.nb_samples.bit_length():
            effects = 4.0 * tf.reduce_mean(scores[:, None] * pair_signs, axis=0)
        else:
            half = self.nb_samples // 2
            group_scores = (scores[:half] + scores[half:]) / 2.0
            group_scores -= tf.reduce_mean(group_scores)
            features = pair_signs[:half]
            features -= tf.reduce_mean(features, axis=0, keepdims=True)
            effects = 4.0 * tf.reduce_sum(group_scores[:, None] * features, axis=0) / (half - 1)
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
    explain_interactions fits all centered quadratic pair features before selecting
    returned pairs. In sampled mode, the independent pair-feature design has
    nb_samples/2 rows and requires more rows than active pairs, plus full rank.
    A selected pair's coefficient does not depend on other requested pairs.

    References
    ----------
    Liu et al. (2025), "Kernel Banzhaf: A Fast and Robust Estimator for
    Banzhaf Values", https://arxiv.org/abs/2410.08336.
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

    def _explain_interactions_input(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        single_input: tf.Tensor,
        single_target: tf.Tensor,
        input_index: int,
        active_ids: tf.Tensor,
        local_pairs: tf.Tensor,
        pair_batch_size: int,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Fit all centered pair features, then report only requested coefficients."""
        nb_active = int(tf.size(active_ids))
        pair_count = nb_active * (nb_active - 1) // 2
        sampled = nb_active >= self.nb_samples.bit_length()
        if int(tf.shape(local_pairs)[0]) and sampled and self.nb_samples // 2 <= pair_count:
            raise ValueError(
                f"Quadratic pair design needs more than {pair_count} independent groups "
                f"for {nb_active} active channels; nb_samples={self.nb_samples} "
                "is insufficient. Increase nb_samples."
            )

        masks = self._sample_masks(nb_active, input_index)
        if not int(tf.shape(local_pairs)[0]):
            outputs = self._evaluate_masks(single_input, single_target, active_ids, masks)
            return self._estimate(masks, outputs), tf.zeros([0], tf.float32)

        all_pairs = np.stack(np.triu_indices(nb_active, 1), axis=1)
        centered = tf.cast(masks, tf.float64) - 0.5
        if sampled:
            centered = centered[: self.nb_samples // 2]
        features = tf.gather(centered, all_pairs[:, 0], axis=1) * tf.gather(
            centered, all_pairs[:, 1], axis=1
        )
        features -= tf.reduce_mean(features, axis=0, keepdims=True)
        singular_values, left, right = tf.linalg.svd(features, full_matrices=False)
        tolerance = np.finfo(np.float64).eps * max(features.shape) * singular_values[0]
        rank = int(tf.reduce_sum(tf.cast(singular_values > tolerance, tf.int32)))
        if rank < pair_count:
            raise ValueError(
                f"Centered quadratic pair design has rank {rank}, below {pair_count} "
                f"pairs for {nb_active} active channels with nb_samples={self.nb_samples}. "
                "Increase nb_samples; exhaustive enumeration guarantees full rank."
            )

        outputs = self._evaluate_masks(single_input, single_target, active_ids, masks)
        main_effects = self._estimate(masks, outputs)
        scores = tf.cast(outputs, tf.float64)
        if sampled:
            half = self.nb_samples // 2
            scores = (scores[:half] + scores[half:]) / 2.0
        scores -= tf.reduce_mean(scores)
        projected = tf.linalg.matvec(left, scores, transpose_a=True)
        coefficients = tf.cast(tf.linalg.matvec(right, projected / singular_values), tf.float32)

        chunks = []
        for start in range(0, int(tf.shape(local_pairs)[0]), pair_batch_size):
            chunk = local_pairs[start : start + pair_batch_size]
            left_ids, right_ids = chunk[:, 0], chunk[:, 1]
            offsets = left_ids * (2 * nb_active - left_ids - 1) // 2 + right_ids - left_ids - 1
            chunks.append(tf.gather(coefficients, offsets))
        return main_effects, tf.concat(chunks, axis=0)
