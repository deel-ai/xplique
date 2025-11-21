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


class OvercompleteFactorizer:

    def __init__(self, optimizer_class, nb_concepts, device='cuda', **kwargs):
        self._nmf = optimizer_class(nb_concepts=nb_concepts, device=device, **kwargs)
        self.device = device

    def fit(self, activations: np.ndarray):
        activations_torch = torch.tensor(activations, device=self.device)
        result = self._nmf.fit(activations_torch)
        # Handle both tuple return (Z, D) and single tensor return
        if isinstance(result, tuple):
            Z_torch, D_torch = result[0], result[1]
        else:
            Z_torch = result
            D_torch = self._nmf.get_dictionary()

        concept_bank_w = D_torch.detach().cpu().numpy()
        coeffs_u = Z_torch.detach().cpu().numpy()
        return concept_bank_w, coeffs_u

    def encode(self, activations: np.ndarray) -> np.ndarray:
        activations_torch = torch.tensor(activations, device=self.device)
        result = self._nmf.encode(activations_torch)
        return result.detach().cpu().numpy()

    def encode_differentiable(self, activations: torch.Tensor) -> torch.Tensor:
        return self._nmf.encode(activations)

    def decode(self, coefficients):
        if isinstance(coefficients, np.ndarray):
            coefficients_torch = torch.tensor(coefficients, device=self.device)
            result = self._nmf.decode(coefficients_torch)
            return result.detach().cpu().numpy()
        else:
            return self._nmf.decode(coefficients)

    def get_concept_bank(self) -> np.ndarray:
        dictionary = self._nmf.get_dictionary()
        return dictionary.detach().cpu().numpy()

    @property
    def is_fitted(self) -> bool:
        return self._nmf.fitted

    @property
    def requires_positive_activations(self) -> bool:
        from overcomplete.optimization import SemiNMF
        return not isinstance(self._nmf, SemiNMF)
