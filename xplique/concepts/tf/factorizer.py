"""
TensorFlow-specific factorizer implementations
"""

import numpy as np
import tensorflow as tf

from ..factorizer import SklearnNMFFactorizer


class TfSklearnNMFFactorizer(SklearnNMFFactorizer):
    """
    TensorFlow-compatible sklearn NMF factorizer with differentiable encoding.
    """

    def encode_differentiable(self, activations: tf.Tensor) -> tf.Tensor:
        """
        Encode activations with a differentiable non-negative solver.

        Solves the fixed-dictionary NMF subproblem with projected proximal
        gradient descent so gradients still flow through ``activations``.

        Parameters
        ----------
        activations : tf.Tensor
            Activations to encode, shape (n_samples, n_features)

        Returns
        -------
        tf.Tensor
            Coefficients, shape (n_samples, n_concepts)
        """
        if self._concept_bank_w is None:
            raise ValueError("Factorizer must be fitted before encoding")

        if not isinstance(activations, tf.Tensor):
            activations = tf.convert_to_tensor(activations)

        beta_loss = self.nmf_kwargs.get("beta_loss", "frobenius")
        if beta_loss != "frobenius":
            raise NotImplementedError(
                "TensorFlow differentiable NMF encoding only supports beta_loss='frobenius'"
            )

        tf.debugging.assert_non_negative(
            activations, message="NMF requires non-negative activations"
        )

        concept_bank_tensor = tf.constant(self._concept_bank_w, dtype=activations.dtype)
        dtype = activations.dtype
        eps = tf.cast(1e-8, dtype)

        alpha_w = tf.cast(self.nmf_kwargs.get("alpha_W", 0.0), dtype)
        l1_ratio = tf.cast(self.nmf_kwargs.get("l1_ratio", 0.0), dtype)
        n_features = tf.cast(tf.shape(activations)[1], dtype)
        l1_reg = n_features * alpha_w * l1_ratio
        l2_reg = n_features * alpha_w * (1.0 - l1_ratio)

        max_iter = int(self.nmf_kwargs.get("max_iter", 200))
        tol = float(self.nmf_kwargs.get("tol", 1e-4))

        gram = concept_bank_tensor @ tf.transpose(concept_bank_tensor)
        cross = activations @ tf.transpose(concept_bank_tensor)
        diagonal = tf.linalg.diag_part(gram)
        coefficients = tf.nn.relu(cross / (diagonal[tf.newaxis, :] + l2_reg + eps))

        lipschitz = tf.norm(gram, ord="fro", axis=(0, 1)) + l2_reg + eps
        step = tf.cast(1.0, dtype) / lipschitz

        if tol > 0.0:
            max_iter_tensor = tf.constant(max_iter)
            tol_tensor = tf.cast(tol, dtype)
            delta = tf.constant(float("inf"), dtype=dtype)

            def cond(iteration, coeffs, coeff_delta):
                return tf.logical_and(iteration < max_iter_tensor, coeff_delta > tol_tensor)

            def body(iteration, coeffs, _):
                gradient = coeffs @ gram - cross + l2_reg * coeffs
                updated = tf.nn.relu(coeffs - step * gradient - step * l1_reg)
                coeff_delta = tf.norm(updated - coeffs) / (tf.norm(updated) + eps)
                return iteration + 1, updated, coeff_delta

            _, coefficients, _ = tf.while_loop(
                cond,
                body,
                (tf.constant(0), coefficients, delta),
                parallel_iterations=1,
            )
        else:
            for _ in range(max_iter):
                gradient = coefficients @ gram - cross + l2_reg * coefficients
                coefficients = tf.nn.relu(coefficients - step * gradient - step * l1_reg)

        return coefficients

    def decode(self, coefficients: tf.Tensor) -> tf.Tensor:
        """
        Decode coefficients to activations via matrix multiplication.

        Parameters
        ----------
        coefficients : tf.Tensor
            Coefficients to decode, shape (n_samples, n_concepts)

        Returns
        -------
        tf.Tensor
            Reconstructed activations, shape (n_samples, n_features)
        """
        if self._concept_bank_w is None:
            raise ValueError("Factorizer must be fitted before decoding")

        if isinstance(coefficients, np.ndarray):
            return coefficients @ self._concept_bank_w

        concept_bank_tensor = tf.constant(self._concept_bank_w, dtype=coefficients.dtype)

        return coefficients @ concept_bank_tensor
