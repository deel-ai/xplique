"""Singleton and pairwise HSIC dependence for exact active concept channels."""

from itertools import combinations, islice
from numbers import Integral
from typing import List

import numpy as np
import tensorflow as tf

from ...commons import tensor_sanitize
from ...types import Callable, OperatorSignature, Optional, Union
from .base import _ConceptChannelExplainer
from .interactions import ConceptInteractionResult


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

    References
    ----------
    Gretton et al. (2005), "Measuring Statistical Dependence with Hilbert-Schmidt
    Norms", https://doi.org/10.1007/11564089_7.
    Novello, Fel, and Vigouroux (2022), "Making Sense of Dependence: Efficient
    Black-box Explanations Using Dependence Measure",
    https://arxiv.org/abs/2206.06219.
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
            or nb_samples < 2
        ):
            raise ValueError("nb_samples must be an integer greater than or equal to two.")
        super().__init__(model, batch_size, operator, seed)
        self.nb_samples = int(nb_samples)

    def _sample_masks(self, nb_active: int, input_index: int) -> tf.Tensor:
        seed = tf.random.experimental.stateless_fold_in(
            tf.constant([self.seed, 0], dtype=tf.int64), tf.cast(input_index, tf.int64)
        )
        uniform = tf.random.stateless_uniform(
            [self.nb_samples, nb_active], seed=seed, dtype=tf.float32
        )
        return tf.cast(uniform < 0.5, tf.float32)

    def _estimate(self, masks: tf.Tensor, outputs: tf.Tensor) -> tf.Tensor:
        """Retain the original singleton estimator for input-shaped explanations."""
        return self._estimate_singletons(masks, self._output_gram(outputs))

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
    def _estimate_singletons(masks: tf.Tensor, output_gram: Optional[tf.Tensor]) -> tf.Tensor:
        """Compute equality-kernel singleton HSIC from the shared output Gram."""
        if output_gram is None:
            return tf.zeros([tf.shape(masks)[1]], tf.float32)
        masks = tf.cast(masks, tf.float64)
        centered_masks = masks - tf.reduce_mean(masks, axis=0, keepdims=True)
        projected = tf.matmul(output_gram, centered_masks)
        normalizer = tf.square(tf.cast(tf.shape(masks)[0], tf.float64))
        scores = 2.0 * tf.reduce_sum(centered_masks * projected, axis=0) / normalizer
        return tf.cast(tf.maximum(scores, 0.0), tf.float32)

    @staticmethod
    def _estimate_pair_chunk(
        signed_masks: tf.Tensor, local_pairs: tf.Tensor, output_gram: tf.Tensor
    ) -> tf.Tensor:
        """Estimate at most one chunk of pairs using (n, B) centered sign products."""
        left = tf.gather(signed_masks, local_pairs[:, 0], axis=1)
        right = tf.gather(signed_masks, local_pairs[:, 1], axis=1)
        products = left * right
        centered = products - tf.reduce_mean(products, axis=0, keepdims=True)
        projected = tf.matmul(output_gram, centered)
        normalizer = 4.0 * tf.square(tf.cast(tf.shape(signed_masks)[0], tf.float64))
        scores = tf.reduce_sum(centered * projected, axis=0) / normalizer
        return tf.cast(tf.maximum(scores, 0.0), tf.float32)

    @staticmethod
    def _validate_pairs(pairs, n_concepts: int) -> Optional[np.ndarray]:
        if pairs is None:
            return None
        requested = np.asarray(pairs)
        if requested.ndim != 2 or requested.shape[1] != 2:
            raise ValueError("pairs must be an integer array of shape (P, 2).")
        if not np.issubdtype(requested.dtype, np.integer) or np.issubdtype(
            requested.dtype, np.bool_
        ):
            raise ValueError("pairs must contain integer indices, excluding booleans.")
        if np.any(requested[:, 0] < 0) or np.any(requested[:, 1] >= n_concepts):
            raise ValueError("pairs must satisfy 0 <= i < j < n_concepts.")
        if np.any(requested[:, 0] >= requested[:, 1]):
            raise ValueError("pairs must satisfy 0 <= i < j < n_concepts.")
        if len(np.unique(requested, axis=0)) != len(requested):
            raise ValueError("pairs must not contain duplicates.")
        return requested.astype(np.int64)

    def explain_interactions(
        self,
        inputs,
        targets=None,
        *,
        pairs=None,
        pair_batch_size=256,
    ) -> List[ConceptInteractionResult]:
        """Estimate singleton and pairwise HSIC from one mask design per input.

        Parameters
        ----------
        inputs
            Dense channel-last coefficients (N, ..., K) or a paired dataset.
        targets
            Fixed targets for the inputs; omit only for a paired dataset.
        pairs
            Optional integer (P, 2) ambient concept indices, ordered i < j.
            None evaluates all distinct active pairs. Inactive requested pairs
            have exact zero scores; unrequested pairs are not evaluated.
        pair_batch_size
            Maximum number of pair features evaluated together, independently
            of the inference batch size.

        Returns
        -------
        results
            One indexed ConceptInteractionResult per input, with no
            spatial axes. Scores are unsigned dependence components, not signed
            synergy or second-order Sobol indices. An empty batch returns [].

        Raises
        ------
        ValueError
            If the sanitized inputs, targets, pairs or pair batch size are invalid.

        Notes
        -----
        Let s = 2M - 1 and q_ij = H(s_i * s_j). The interaction component is
        q_ij.T @ L @ q_ij / (4*n**2), using the same uncentered output RBF Gram L
        as the singleton effects. The paper uses (n-1)**2 normalization; this
        method preserves the existing n**2 convention. Finite IID masks induce
        sampling noise. XOR can have zero singletons and a positive pair, while
        higher-order dependence is not exhausted by pairs. With an RBF output
        kernel, even additive scalar scores need not have zero pair components.
        """
        if (
            isinstance(pair_batch_size, (bool, np.bool_))
            or not isinstance(pair_batch_size, Integral)
            or pair_batch_size <= 0
        ):
            raise ValueError("pair_batch_size must be a positive integer.")
        inputs, targets = tensor_sanitize(inputs, targets)
        self._validate_inputs_targets(inputs, targets)
        n_concepts = int(inputs.shape[-1])
        requested_pairs = self._validate_pairs(pairs, n_concepts)
        results = []
        for input_index, single_input in enumerate(inputs):
            results.append(
                self._interactions_for_input(
                    single_input,
                    targets[input_index : input_index + 1],
                    input_index,
                    n_concepts,
                    requested_pairs,
                    int(pair_batch_size),
                )
            )
        return results

    def _interactions_for_input(
        self,
        single_input,
        single_target,
        input_index,
        n_concepts,
        requested_pairs,
        pair_batch_size,
    ) -> ConceptInteractionResult:
        """Reuse one mask design and one output Gram for this input's scores."""
        active_ids = self._active_channel_ids(single_input)
        active = active_ids.numpy()
        output_gram = None
        signed_masks = None
        if len(active):
            masks = self._sample_masks(len(active), input_index)
            outputs = self._evaluate_masks(single_input, single_target, active_ids, masks)
            output_gram = self._output_gram(outputs)
            main_effects = self._estimate_singletons(masks, output_gram)
            if output_gram is not None:
                signed_masks = 2.0 * tf.cast(masks, tf.float64) - 1.0
        else:
            main_effects = tf.zeros([0], tf.float32)

        if requested_pairs is None:
            pair_indices, pair_scores = self._all_pair_scores(
                active, signed_masks, output_gram, pair_batch_size
            )
        else:
            pair_indices = tf.convert_to_tensor(requested_pairs, tf.int64)
            pair_scores = self._requested_pair_scores(
                requested_pairs,
                active,
                n_concepts,
                signed_masks,
                output_gram,
                pair_batch_size,
            )
        return ConceptInteractionResult(
            n_concepts, active_ids, main_effects, pair_indices, pair_scores
        )

    def _all_pair_scores(self, active, signed_masks, output_gram, pair_batch_size):
        """Stream automatic pairs lexicographically in bounded feature chunks."""
        pair_iter = combinations(range(len(active)), 2)
        pair_indices = []
        pair_scores = []
        chunk = list(islice(pair_iter, pair_batch_size))
        while chunk:
            local_chunk = np.asarray(chunk, np.int64)
            pair_indices.extend(active[local_chunk].tolist())
            if output_gram is None:
                pair_scores.extend([0.0] * len(chunk))
            else:
                pair_scores.extend(
                    self._estimate_pair_chunk(
                        signed_masks, tf.convert_to_tensor(local_chunk), output_gram
                    )
                    .numpy()
                    .tolist()
                )
            chunk = list(islice(pair_iter, pair_batch_size))
        return (
            tf.convert_to_tensor(np.asarray(pair_indices, np.int64).reshape(-1, 2)),
            tf.convert_to_tensor(pair_scores, tf.float32),
        )

    def _requested_pair_scores(
        self,
        requested_pairs,
        active,
        n_concepts,
        signed_masks,
        output_gram,
        pair_batch_size,
    ) -> tf.Tensor:
        """Score active requested pairs; leave inactive rows at exact zero."""
        scores = np.zeros(len(requested_pairs), np.float32)
        if output_gram is None:
            return tf.convert_to_tensor(scores)
        ambient_to_local = np.full(n_concepts, -1, np.int64)
        ambient_to_local[active] = np.arange(len(active))
        local_pairs = ambient_to_local[requested_pairs]
        valid_rows = np.flatnonzero(np.all(local_pairs >= 0, axis=1))
        for start in range(0, len(valid_rows), pair_batch_size):
            rows = valid_rows[start : start + pair_batch_size]
            scores[rows] = self._estimate_pair_chunk(
                signed_masks, tf.convert_to_tensor(local_pairs[rows]), output_gram
            ).numpy()
        return tf.convert_to_tensor(scores)
