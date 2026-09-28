"""Signed Banzhaf effects for exact active concept channels."""

import numpy as np
import tensorflow as tf

from ...types import Callable, OperatorSignature, Optional, Tuple, Union
from .base import _bernoulli, _check_integer, _ConceptChannelExplainer


def _numerical_rank(singular_values: tf.Tensor, shape: tf.TensorShape) -> int:
    """Count singular values above eps(float64) * max(shape) * largest singular value."""
    tolerance = np.finfo(np.float64).eps * max(shape) * singular_values[0]
    return int(tf.reduce_sum(tf.cast(singular_values > tolerance, tf.int32)))


def _svd_solve(factors: Tuple[tf.Tensor, tf.Tensor, tf.Tensor], scores: tf.Tensor) -> tf.Tensor:
    """Least-squares coefficients V diag(1/s) U.T y of a full-rank factored design."""
    singular_values, left, right = factors
    projected = tf.linalg.matvec(left, scores, transpose_a=True)
    return tf.linalg.matvec(right, projected / singular_values)


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
    independent group; at least two groups are required for requested pairs, which
    is checked for every input before any inference.

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

    def _is_enumerated(self, nb_active: int) -> bool:
        """Whether all 2**d coalitions fit the budget, without constructing 2**d."""
        return nb_active < self.nb_samples.bit_length()

    def _sample_masks(self, nb_active: int, input_index: int) -> tf.Tensor:
        if self._is_enumerated(nb_active):
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

    def _check_interactions(self, nb_active: int) -> None:
        """Sampled pair covariances need at least two independent mask/complement groups."""
        if not self._is_enumerated(nb_active) and self.nb_samples < 4:
            raise ValueError("Sampled Banzhaf interactions require nb_samples >= 4.")

    def _prepare_interactions(
        self, masks: tf.Tensor, outputs: tf.Tensor, evaluate: Callable
    ) -> Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """Reduce every pair estimate to sign products weighted by centered scores."""
        del evaluate  # Pairs reuse the singleton design.
        signs = 2.0 * tf.cast(masks, tf.float64) - 1.0
        # Center before grouping: summing raw scores with a large offset loses precision.
        scores = tf.cast(outputs, tf.float64)
        scores -= tf.reduce_mean(scores)
        if self._is_enumerated(int(masks.shape[1])):
            # Exact coalition average 4 * mean((v - mean(v)) * s_i * s_j).
            weights = 4.0 * scores / float(masks.shape[0])
        else:
            # A mask and its complement share s_i * s_j: each pair is one independent
            # group. Recentering removes rounding residue; centered group scores
            # suffice for the unbiased covariance with the uncentered sign products.
            half = self.nb_samples // 2
            group_scores = (scores[:half] + scores[half:]) / 2.0
            weights = 4.0 * (group_scores - tf.reduce_mean(group_scores)) / (half - 1)
            signs = signs[:half]
        return self._estimate(masks, outputs), (signs, weights)

    def _estimate_pair_chunk(
        self, state: Tuple[tf.Tensor, tf.Tensor], local_pairs: tf.Tensor
    ) -> tf.Tensor:
        """Contract each pair's sign product with the shared score weights."""
        signs, weights = state
        products = tf.gather(signs, local_pairs[:, 0], axis=1) * tf.gather(
            signs, local_pairs[:, 1], axis=1
        )
        return tf.cast(tf.linalg.matvec(products, weights, transpose_a=True), tf.float32)


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
        If the centered design is rank deficient. Realized rank is checked before
        each input's inference; previous inputs in the same call may already have
        been evaluated. explain_interactions checks the pair budget for every input
        before any inference.

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
        if not self._is_enumerated(nb_active) and nb_active > self.nb_samples // 2:
            raise ValueError(
                f"Active dimension {nb_active} exceeds the antithetic rank bound "
                f"{self.nb_samples // 2} for nb_samples={self.nb_samples}. "
                "Increase nb_samples to allow full-rank regression."
            )
        masks = super()._sample_masks(nb_active, input_index)
        centered = tf.cast(masks, tf.float64) - 0.5
        rank = _numerical_rank(tf.linalg.svd(centered, compute_uv=False), centered.shape)
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
        factors = tf.linalg.svd(centered, full_matrices=False)
        return tf.cast(_svd_solve(factors, scores), tf.float32)

    def _check_interactions(self, nb_active: int) -> None:
        """Sampled quadratic regression needs more independent groups than active pairs.

        This bound subsumes the parent's two-group requirement whenever a pair exists.
        """
        pair_count = nb_active * (nb_active - 1) // 2
        if not self._is_enumerated(nb_active) and self.nb_samples // 2 <= pair_count:
            raise ValueError(
                f"Quadratic pair design needs more than {pair_count} independent groups "
                f"for {nb_active} active channels; nb_samples={self.nb_samples} "
                "is insufficient. Increase nb_samples."
            )

    def _pair_design_factors(self, masks: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Factor the centered quadratic design of every active pair, requiring full rank."""
        nb_active = int(masks.shape[1])
        left_ids, right_ids = np.triu_indices(nb_active, 1)
        centered = tf.cast(masks, tf.float64) - 0.5
        if not self._is_enumerated(nb_active):
            # A mask and its complement share every pair product: keep one row per group.
            centered = centered[: self.nb_samples // 2]
        features = tf.gather(centered, left_ids, axis=1) * tf.gather(centered, right_ids, axis=1)
        features -= tf.reduce_mean(features, axis=0, keepdims=True)
        factors = tf.linalg.svd(features, full_matrices=False)
        rank = _numerical_rank(factors[0], features.shape)
        if rank < len(left_ids):
            raise ValueError(
                f"Centered quadratic pair design has rank {rank}, below {len(left_ids)} "
                f"pairs for {nb_active} active channels with nb_samples={self.nb_samples}. "
                "Increase nb_samples; exhaustive enumeration guarantees full rank."
            )
        return factors

    def _explain_interactions_input(  # pylint: disable=too-many-arguments
        self,
        single_input: tf.Tensor,
        single_target: tf.Tensor,
        input_index: int,
        active_ids: tf.Tensor,
        local_pairs: tf.Tensor,
        pair_batch_size: int,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Fit all centered pair features, checking the realized rank before inference."""
        nb_active = int(tf.size(active_ids))
        masks = self._sample_masks(nb_active, input_index)
        factors = self._pair_design_factors(masks)
        outputs = self._evaluate_masks(single_input, single_target, active_ids, masks)
        scores = tf.cast(outputs, tf.float64)
        # Center before grouping to avoid cancellation, then remove rounding residue.
        scores -= tf.reduce_mean(scores)
        if not self._is_enumerated(nb_active):
            half = self.nb_samples // 2
            scores = (scores[:half] + scores[half:]) / 2.0
        coefficients = _svd_solve(factors, scores - tf.reduce_mean(scores))
        state = (tf.cast(coefficients, tf.float32), nb_active)
        return self._estimate(masks, outputs), self._estimate_pairs(
            state, local_pairs, pair_batch_size
        )

    def _estimate_pair_chunk(
        self, state: Tuple[tf.Tensor, int], local_pairs: tf.Tensor
    ) -> tf.Tensor:
        """Gather fitted coefficients at the row-major upper-triangular pair offsets."""
        coefficients, nb_active = state
        left_ids, right_ids = local_pairs[:, 0], local_pairs[:, 1]
        offsets = left_ids * (2 * nb_active - left_ids - 1) // 2 + right_ids - left_ids - 1
        return tf.gather(coefficients, offsets)
