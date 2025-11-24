"""
Attribution complexity metrics
Re-implementations in TensorFlow for efficiency, inspired by Quantus:
https://github.com/understandable-machine-intelligence-lab/Quantus/blob/main/quantus/metrics/complexity/complexity.py
"""
import numpy as np
import tensorflow as tf

from .base import BaseComplexityMetric


_EPS = 1e-8


class Complexity(BaseComplexityMetric):
    """
    Entropy-based complexity of attribution maps.

    The metric computes the Shannon entropy of the absolute attributions, after
    flattening (and averaging across channels if present). Intuitively, higher
    entropy indicates more diffuse/less concentrated explanations; lower entropy
    indicates concentrated/sparse explanations.

    Reference
    ---------
    Bhatt et al., "Evaluating and Aggregating Feature-based Model Explanations"
    (2020). https://arxiv.org/abs/2005.00631

    Notes
    -----
    - If explanations are 4D (B, H, W, C), we average across channels C before
      flattening, consistent with typical usage for saliency maps.
    """
    def detailed_evaluate(self, explanations: tf.Tensor) -> np.ndarray:
        """
        Per-sample entropy of explanations (no reduction).

        Returns
        -------
        entropies
            A numpy array of shape (B,) with entropy per sample.
        """
        x = tf.convert_to_tensor(explanations, dtype=tf.float32)

        # Channel handling: average across channels if present
        if x.shape.rank == 4:  # (B, H, W, C)
            x = tf.reduce_mean(x, axis=-1)

        # |attribution| and flatten
        x = tf.math.abs(x)
        b = tf.shape(x)[0]
        x = tf.reshape(x, (b, -1))

        # Normalize to a probability simplex
        norm = tf.reduce_sum(x, axis=-1, keepdims=True)
        p = x / (norm + _EPS)

        # Entropy: -sum p log p
        entropy = -tf.reduce_sum(p * tf.math.log(p + _EPS), axis=-1)

        return entropy.numpy()


class Sparseness(BaseComplexityMetric):
    """
    Gini-index-based sparseness of attribution maps.

    The metric computes the Gini coefficient of the absolute attributions after
    L1-normalization (and channel-averaging if needed). Higher values indicate
    sparser/more concentrated explanations.

    Reference
    ---------
    Chalasani et al., "Concise Explanations of Neural Networks using Adversarial
    Training" (ICML 2020). https://proceedings.mlr.press/v119/chalasani20a.html

    Notes
    -----
    - Gini(x) = (2 * sum_i i x_(i)) / (n * sum_i x_(i)) - (n + 1) / n,
      where x_(i) is the i-th smallest component and n is the number of features.
    - We compute on |attribution| with L1 normalization for scale invariance.
    - If explanations are 4D (B, H, W, C), we average across channels C.
    """
    def detailed_evaluate(self, explanations: tf.Tensor) -> np.ndarray:
        """
        Per-sample Gini index of explanations (no reduction).

        Returns
        -------
        ginis
            A numpy array of shape (B,) with Gini per sample.
        """
        x = tf.convert_to_tensor(explanations, dtype=tf.float32)

        # Channel handling: average across channels if present
        if x.shape.rank == 4:  # (B, H, W, C)
            x = tf.reduce_mean(x, axis=-1)

        # |attribution| and flatten
        x = tf.math.abs(x)
        b = tf.shape(x)[0]
        x = tf.reshape(x, (b, -1))  # (B, n)

        # L1-normalize for scale invariance
        l1 = tf.reduce_sum(x, axis=-1, keepdims=True)
        x = x / (l1 + _EPS)

        # Ascending sort for Gini
        x_sorted = tf.sort(x, axis=-1)

        # n can be dynamic; build 1..n vector and broadcast
        n = tf.cast(tf.shape(x_sorted)[1], tf.float32)
        idx = tf.cast(tf.range(1, tf.shape(x_sorted)[1] + 1), tf.float32)  # (n,)
        # (B, n) via broadcasting
        weighted = x_sorted * idx

        # Gini formula: (2 * sum_i i x_i) / (n * sum_i x_i) - (n+1)/n
        # sum_i x_i == 1 due to L1-normalization, but keep general form for robustness
        num = 2.0 * tf.reduce_sum(weighted, axis=-1)                 # (B,)
        den = n * tf.reduce_sum(x_sorted, axis=-1) + _EPS            # (B,)
        gini = num / den - (n + 1.0) / n                             # (B,)

        return gini.numpy()
