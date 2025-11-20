"""
TensorFlow-specific factorizer implementations
"""
import tensorflow as tf
import numpy as np

from ..factorizer import SklearnNMFFactorizer


class TfSklearnNMFFactorizer(SklearnNMFFactorizer):
    """
    TensorFlow-compatible sklearn NMF factorizer with differentiable encoding.
    """
    
    def encode_differentiable(self, activations: tf.Tensor) -> tf.Tensor:
        """
        Encode activations to coefficients using differentiable least squares.
        
        Uses tf.linalg.lstsq to solve: activations ≈ coefficients @ concept_bank_w
        
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
        
        concept_bank_tensor = tf.constant(
            self._concept_bank_w,
            dtype=activations.dtype
        )
        
        coefficients = tf.linalg.lstsq(
            tf.transpose(concept_bank_tensor),
            tf.transpose(activations)
        )
        coefficients = tf.transpose(coefficients)
        
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
        
        concept_bank_tensor = tf.constant(
            self._concept_bank_w,
            dtype=coefficients.dtype
        )
        
        return coefficients @ concept_bank_tensor
