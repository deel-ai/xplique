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
        Encode activations with a differentiable non-negative solver.

        Solves the fixed-dictionary NMF subproblem with restartable FISTA so
        gradients still flow through ``activations``.

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

        if not isinstance(activations, torch.Tensor):
            activations = torch.as_tensor(activations)

        beta_loss = self.nmf_kwargs.get("beta_loss", "frobenius")
        if beta_loss != "frobenius":
            raise NotImplementedError(
                "PyTorch differentiable NMF encoding only supports beta_loss='frobenius'"
            )

        if torch.any(activations < 0).item():
            raise ValueError("NMF requires non-negative activations")

        concept_bank_tensor = torch.tensor(
            self._concept_bank_w, dtype=activations.dtype, device=activations.device
        )
        dtype = activations.dtype
        device = activations.device
        eps = torch.tensor(1e-8, dtype=dtype, device=device)
        one = torch.tensor(1.0, dtype=dtype, device=device)
        four = torch.tensor(4.0, dtype=dtype, device=device)

        alpha_w = torch.tensor(self.nmf_kwargs.get("alpha_W", 0.0), dtype=dtype, device=device)
        l1_ratio = torch.tensor(self.nmf_kwargs.get("l1_ratio", 0.0), dtype=dtype, device=device)
        n_features = torch.tensor(activations.shape[1], dtype=dtype, device=device)
        l1_reg = n_features * alpha_w * l1_ratio
        l2_reg = n_features * alpha_w * (1.0 - l1_ratio)

        max_iter = int(self.nmf_kwargs.get("max_iter", 200))
        tol = float(self.nmf_kwargs.get("tol", 1e-4))

        gram = concept_bank_tensor @ concept_bank_tensor.T
        cross = activations @ concept_bank_tensor.T
        diagonal = torch.diagonal(gram)
        coefficients = torch.relu(cross / (diagonal.unsqueeze(0) + l2_reg + eps))

        lipschitz = torch.linalg.norm(gram, ord="fro") + l2_reg + eps
        step = one / lipschitz

        def fista_step(coeffs, extrapolated_coeffs, momentum):
            gradient = extrapolated_coeffs @ gram - cross + l2_reg * extrapolated_coeffs
            updated = torch.relu(extrapolated_coeffs - step * gradient - step * l1_reg)
            coeff_delta = torch.linalg.norm(updated - coeffs) / (torch.linalg.norm(updated) + eps)

            next_momentum = (one + torch.sqrt(one + four * torch.square(momentum))) / 2.0
            accelerated = updated + ((momentum - one) / next_momentum) * (updated - coeffs)
            restart = torch.sum((extrapolated_coeffs - updated) * (updated - coeffs)) > 0

            next_extrapolated = torch.where(restart, updated, accelerated)
            next_momentum = torch.where(restart, one, next_momentum)
            return updated, next_extrapolated, next_momentum, coeff_delta

        extrapolated_coeffs = coefficients
        momentum = one

        if tol > 0.0:
            for _ in range(max_iter):
                coefficients, extrapolated_coeffs, momentum, delta = fista_step(
                    coefficients, extrapolated_coeffs, momentum
                )
                if delta.item() <= tol:
                    break
        else:
            for _ in range(max_iter):
                coefficients, extrapolated_coeffs, momentum, _ = fista_step(
                    coefficients, extrapolated_coeffs, momentum
                )

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
