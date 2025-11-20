"""
PyTorch-specific factorizer implementations
"""
import torch
import numpy as np

from ..factorizer import SklearnNMFFactorizer


class TorchSklearnNMFFactorizer(SklearnNMFFactorizer):
    """
    PyTorch-compatible sklearn NMF factorizer with differentiable encoding.
    """
    
    def encode_differentiable(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Encode activations to coefficients using differentiable least squares.
        
        Uses torch.linalg.lstsq to solve: activations ≈ coefficients @ concept_bank_w
        
        Parameters
        ----------
        activations : torch.Tensor
            Activations to encode, shape (n_samples, n_features)
        
        Returns
        -------
        torch.Tensor
            Coefficients, shape (n_samples, n_concepts)
        """
        if self._concept_bank_w is None:
            raise ValueError("Factorizer must be fitted before encoding")
        
        concept_bank_tensor = torch.tensor(
            self._concept_bank_w,
            dtype=activations.dtype,
            device=activations.device
        )
        
        coefficients = torch.linalg.lstsq(
            concept_bank_tensor.T,
            activations.T
        ).solution.T
        
        return coefficients
    
    def decode(self, coefficients: torch.Tensor) -> torch.Tensor:
        """
        Decode coefficients to activations via matrix multiplication.
        
        Parameters
        ----------
        coefficients : torch.Tensor
            Coefficients to decode, shape (n_samples, n_concepts)
        
        Returns
        -------
        torch.Tensor
            Reconstructed activations, shape (n_samples, n_features)
        """
        if self._concept_bank_w is None:
            raise ValueError("Factorizer must be fitted before decoding")
        
        if isinstance(coefficients, np.ndarray):
            return coefficients @ self._concept_bank_w
        
        concept_bank_tensor = torch.tensor(
            self._concept_bank_w,
            dtype=coefficients.dtype,
            device=coefficients.device
        )
        
        return coefficients @ concept_bank_tensor
