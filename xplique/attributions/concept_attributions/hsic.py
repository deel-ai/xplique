"""Singleton and pairwise HSIC dependence for exact active concept channels."""

import tensorflow as tf

from ...types import Callable, OperatorSignature, Optional, Tuple, Union
from .base import _bernoulli, _check_integer, _ConceptChannelExplainer


def _median_or_one(values: tf.Tensor) -> tf.Tensor:
    """Return the median of values, or one when empty."""
    values = tf.sort(values)
    count = tf.size(values)

    def median():
        lower = values[(count - 1) // 2]
        upper = values[count // 2]
        midpoint = lower + (upper - lower) / 2.0
        return tf.where(tf.math.is_inf(upper), upper, midpoint)

    return tf.cond(count > 0, median, lambda: tf.constant(1.0, tf.float64))


def _median_positive_pairwise_distance(outputs: tf.Tensor) -> tf.Tensor:
    """Return the median positive pairwise distance, or one when none exists."""
    outputs = tf.cast(tf.reshape(outputs, [-1]), tf.float64)
    distances = tf.abs(outputs[:, None] - outputs[None, :])
    return _median_or_one(tf.boolean_mask(distances, distances > 0))


class SparseHSIC(_ConceptChannelExplainer):
    """Estimate singleton and pairwise dependence on active binary concept channels.

    Each IID mask retains or removes one whole channel across all positions. The
    estimator uses a binary equality input kernel and a scalar-output RBF kernel.
    Scores are unsigned marginal dependence values, not signed effects or
    total-order interaction indices.

    Parameters
    ----------
    model
        Model consuming already-encoded coefficients, optionally through a decoder.
    batch_size
        Maximum perturbations evaluated together. None evaluates all masks for one
        input in one call and is not memory-bounded.
    operator
        Xplique fixed-target operator returning finite scores of shape (B,) or (B, 1).
    nb_samples
        Number of IID Bernoulli masks per nonempty input, default 1024. Must be at
        least two; odd values are accepted. This is also the evaluation count.
    seed
        Signed 64-bit stateless seed folded with the original input index.

    Notes
    -----
    The output RBF bandwidth is the median strictly positive pairwise score
    distance. Constant outputs return exact zeros. The estimator uses biased
    HSIC normalization by nb_samples squared and float64 arithmetic, then casts
    results to float32. Its algebra requires O(nb_samples**2 + nb_samples*d)
    memory rather than constructing a (d, nb_samples, nb_samples) tensor.

    explain_interactions reuses the same masks and output Gram. Let s = 2M - 1
    and q_ij = H(s_i * s_j). The interaction component is
    q_ij.T @ L @ q_ij / (4*n**2), using the same uncentered output RBF Gram L
    as the singleton effects. The paper uses (n-1)**2 normalization; this
    method preserves the existing n**2 convention. Finite IID masks induce
    sampling noise. XOR can have zero singletons and a positive pair, while
    higher-order dependence is not exhausted by pairs. With an RBF output
    kernel, even additive scalar scores need not have zero pair components.
    Pair scores are unsigned dependence components, not signed synergy or
    second-order Sobol indices.

    References
    ----------
    Gretton et al. (2005), "Measuring Statistical Dependence with Hilbert-Schmidt
    Norms", https://doi.org/10.1007/11564089_7.
    Novello, Fel, and Vigouroux (2022), "Making Sense of Dependence: Efficient
    Black-box Explanations Using Dependence Measure",
    https://arxiv.org/abs/2206.06219.
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
            nb_samples, "nb_samples must be an integer greater than or equal to two.", 2
        )
        super().__init__(model, batch_size, operator, seed)
        self.nb_samples = nb_samples

    def _sample_masks(self, nb_active: int, input_index: int) -> tf.Tensor:
        return _bernoulli([self.nb_samples, nb_active], self._input_seed(input_index))

    def _estimate(self, masks: tf.Tensor, outputs: tf.Tensor) -> tf.Tensor:
        return self._singletons(tf.cast(masks, tf.float64), self._output_gram(outputs))

    @staticmethod
    def _output_gram(outputs: tf.Tensor) -> Optional[tf.Tensor]:
        """Build the uncentered float64 output RBF Gram, or None for constant scores."""
        outputs = tf.cast(tf.reshape(outputs, [-1]), tf.float64)
        if bool(tf.reduce_all(outputs == outputs[0])):
            return None

        distances = tf.abs(outputs[:, None] - outputs[None, :])
        bandwidth = _median_positive_pairwise_distance(outputs)
        if bool(tf.math.is_inf(bandwidth)):
            scaled_outputs = outputs / tf.reduce_max(tf.abs(outputs))
            scaled_distances = tf.abs(scaled_outputs[:, None] - scaled_outputs[None, :])
            bandwidth = _median_or_one(tf.boolean_mask(scaled_distances, distances > 0))
            distances = scaled_distances
        output_gram = tf.exp(-tf.square(distances / bandwidth) / 2.0)
        return output_gram

    @staticmethod
    def _centered_hsic(features: tf.Tensor, output_gram: tf.Tensor, scale: float) -> tf.Tensor:
        """Compute scale * diag(Z.T @ L @ Z) / n**2 for column-centered features Z."""
        centered = features - tf.reduce_mean(features, axis=0, keepdims=True)
        projected = tf.matmul(output_gram, centered)
        normalizer = tf.square(tf.cast(tf.shape(features)[0], tf.float64))
        scores = scale * tf.reduce_sum(centered * projected, axis=0) / normalizer
        return tf.cast(tf.maximum(scores, 0.0), tf.float32)

    @classmethod
    def _singletons(cls, masks: tf.Tensor, output_gram: Optional[tf.Tensor]) -> tf.Tensor:
        """Compute equality-kernel singleton HSIC for float64 masks and a shared output Gram."""
        if output_gram is None:
            return tf.zeros([tf.shape(masks)[1]], tf.float32)
        # The centered equality kernel of a binary column z is 2 * Hz (Hz).T.
        return cls._centered_hsic(masks, output_gram, 2.0)

    def _prepare_interactions(
        self, masks: tf.Tensor, outputs: tf.Tensor, evaluate: Callable
    ) -> Tuple[tf.Tensor, Tuple[tf.Tensor, Optional[tf.Tensor]]]:
        """Share float64 masks and one output Gram (None if constant) per input."""
        del evaluate  # Pairs reuse the singleton design.
        masks = tf.cast(masks, tf.float64)
        output_gram = self._output_gram(outputs)
        return self._singletons(masks, output_gram), (masks, output_gram)

    def _estimate_pair_chunk(
        self, state: Tuple[tf.Tensor, Optional[tf.Tensor]], local_pairs: tf.Tensor
    ) -> tf.Tensor:
        """Estimate at most one chunk of pairs using (n, B) centered sign products."""
        masks, output_gram = state
        if output_gram is None:
            return tf.zeros([tf.shape(local_pairs)[0]], tf.float32)
        left = 2.0 * tf.gather(masks, local_pairs[:, 0], axis=1) - 1.0
        right = 2.0 * tf.gather(masks, local_pairs[:, 1], axis=1) - 1.0
        # Center the sign products after multiplication, not the individual signs.
        return self._centered_hsic(left * right, output_gram, 0.25)
