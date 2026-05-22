"""
PyTorch-specific factorizer implementations
"""

import numpy as np
import torch

from ..factorizer import ConceptFactorizer, SklearnNMFFactorizer


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
            self._concept_bank_w, dtype=activations.dtype, device=activations.device
        )

        # pylint: disable=not-callable
        coefficients = torch.linalg.lstsq(concept_bank_tensor.T, activations.T).solution.T

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
            self._concept_bank_w, dtype=coefficients.dtype, device=coefficients.device
        )

        return coefficients @ concept_bank_tensor


class OvercompleteFactorizer(ConceptFactorizer):
    """
    Factorizer wrapper for overcomplete optimization methods.
    """

    def __init__(self, optimizer_class, nb_concepts, device="cuda", **kwargs):
        """
        Initialize the overcomplete factorizer.

        Parameters
        ----------
        optimizer_class : class
            The NMF Optimizer class to use (e.g., SemiNMF)
        nb_concepts : int
            Number of concepts to extract
        device : str
            Device to use for computation ('cuda' or 'cpu')
        **kwargs
            Additional arguments passed to the optimizer
        """
        self.concept_model = optimizer_class(nb_concepts=nb_concepts, device=device, **kwargs)
        self.device = device

    def fit(self, activations: np.ndarray):
        """
        Fit the factorizer on activations.

        Parameters
        ----------
        activations : np.ndarray
            Activations to factorize

        Returns
        -------
        tuple
            Concept bank and coefficients
        """
        activations_torch = torch.tensor(activations, device=self.device)
        result = self.concept_model.fit(activations_torch)
        # Handle both tuple return (Z, D) and single tensor return
        if isinstance(result, tuple):
            coeffs_torch, dictionary_torch = result[0], result[1]
        else:
            coeffs_torch = result
            dictionary_torch = self.concept_model.get_dictionary()

        concept_bank_w = dictionary_torch.detach().cpu().numpy()
        coeffs_u = coeffs_torch.detach().cpu().numpy()
        return concept_bank_w, coeffs_u

    def encode(self, activations: np.ndarray) -> np.ndarray:
        """
        Encode activations to coefficients.

        Parameters
        ----------
        activations : np.ndarray
            Activations to encode

        Returns
        -------
        np.ndarray
            Coefficients
        """
        activations_torch = torch.tensor(activations, device=self.device)
        result = self.concept_model.encode(activations_torch)
        return result.detach().cpu().numpy()

    def encode_differentiable(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Encode activations using differentiable operations.

        Parameters
        ----------
        activations : torch.Tensor
            Activations to encode

        Returns
        -------
        torch.Tensor
            Coefficients
        """
        return self.concept_model.encode(activations)

    def decode(self, coefficients):
        """
        Decode coefficients to activations.

        Parameters
        ----------
        coefficients : np.ndarray or torch.Tensor
            Coefficients to decode

        Returns
        -------
        np.ndarray or torch.Tensor
            Reconstructed activations
        """
        if isinstance(coefficients, np.ndarray):
            coefficients_torch = torch.tensor(coefficients, device=self.device)
            result = self.concept_model.decode(coefficients_torch)
            return result.detach().cpu().numpy()

        return self.concept_model.decode(coefficients)

    def get_concept_bank(self) -> np.ndarray:
        """
        Get the concept bank (dictionary).

        Returns
        -------
        np.ndarray
            Concept bank
        """
        dictionary = self.concept_model.get_dictionary()
        return dictionary.detach().cpu().numpy()

    @property
    def is_fitted(self) -> bool:
        """
        Check if the factorizer has been fitted.

        Returns
        -------
        bool
            True if fitted, False otherwise
        """
        return self.concept_model.fitted

    @property
    def requires_positive_activations(self) -> bool:
        """
        Check if positive activations are required.

        Returns
        -------
        bool
            True if positive activations are required
        """
        # pylint: disable=import-outside-toplevel
        from overcomplete.optimization import SemiNMF

        return not isinstance(self.concept_model, SemiNMF)
