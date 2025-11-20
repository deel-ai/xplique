"""
Factorizer protocol and base implementations for concept extraction

Note: All factorizers expect activations in shape (spatial_flattened, channels),
where spatial_flattened = N*H*W for batched 2D feature maps.
"""
from typing import Protocol, Tuple, Union
import numpy as np
from sklearn.decomposition import NMF


class ConceptFactorizer(Protocol):
    """
    Protocol for concept factorization methods used in CRAFT.
    """
    
    def fit(self, activations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the factorizer on activations.
        
        Parameters
        ----------
        activations : np.ndarray
            Activations to factorize, shape (n_samples, n_features)
        
        Returns
        -------
        concept_bank_w : np.ndarray
            Concept bank (dictionary), shape (n_concepts, n_features)
        coeffs_u : np.ndarray
            Coefficients for the input activations, shape (n_samples, n_concepts)
        """
        ...
    
    def encode(self, activations: np.ndarray) -> np.ndarray:
        """
        Encode activations to coefficients (non-differentiable).
        
        Parameters
        ----------
        activations : np.ndarray
            Activations to encode, shape (n_samples, n_features)
        
        Returns
        -------
        np.ndarray
            Coefficients, shape (n_samples, n_concepts)
        """
        ...
    
    def encode_differentiable(self, activations) -> Union[np.ndarray, "torch.Tensor", "tf.Tensor"]:
        """
        Encode activations to coefficients (differentiable, framework-specific).
        
        Parameters
        ----------
        activations : Tensor
            Activations to encode (torch.Tensor or tf.Tensor)
        
        Returns
        -------
        Tensor
            Coefficients (same framework as input)
        """
        ...
    
    def decode(self, coefficients) -> Union[np.ndarray, "torch.Tensor", "tf.Tensor"]:
        """
        Decode coefficients back to activations (naturally differentiable).
        
        Parameters
        ----------
        coefficients : array or Tensor
            Coefficients to decode, shape (n_samples, n_concepts)
        
        Returns
        -------
        array or Tensor
            Reconstructed activations, shape (n_samples, n_features)
        """
        ...
    
    def get_concept_bank(self) -> np.ndarray:
        """
        Get the concept bank (W matrix).
        
        Returns
        -------
        np.ndarray
            Concept bank, shape (n_concepts, n_features)
        """
        ...
    
    @property
    def is_fitted(self) -> bool:
        """
        Check if the factorizer has been fitted.
        
        Returns
        -------
        bool
            True if fitted, False otherwise
        """
        ...
    
    @property
    def requires_positive_activations(self) -> bool:
        """
        Whether this factorizer requires positive activations.
        
        Returns
        -------
        bool
            True if activations must be non-negative
        """
        ...


class SklearnNMFFactorizer:
    """
    Base sklearn NMF factorizer (framework-agnostic).
    """
    
    def __init__(self, n_components: int = 20, **nmf_kwargs):
        """
        Parameters
        ----------
        n_components : int
            Number of concepts to extract
        **nmf_kwargs
            Additional arguments passed to sklearn.decomposition.NMF
        """
        self.n_components = n_components
        self.nmf_kwargs = nmf_kwargs
        self._decomposer = None
        self._concept_bank_w = None
    
    def fit(self, activations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit NMF on activations.
        
        Parameters
        ----------
        activations : np.ndarray
            Activations to factorize, shape (n_samples, n_features)
        
        Returns
        -------
        concept_bank_w : np.ndarray
            Concept bank (dictionary), shape (n_concepts, n_features)
        coeffs_u : np.ndarray
            Coefficients for the input activations, shape (n_samples, n_concepts)
        """
        self._decomposer = NMF(n_components=self.n_components, **self.nmf_kwargs)
        coeffs_u = self._decomposer.fit_transform(activations)
        self._concept_bank_w = self._decomposer.components_
        return self._concept_bank_w, coeffs_u
    
    def encode(self, activations: np.ndarray) -> np.ndarray:
        """
        Encode activations to coefficients using sklearn NMF transform.
        
        Parameters
        ----------
        activations : np.ndarray
            Activations to encode, shape (n_samples, n_features)
        
        Returns
        -------
        np.ndarray
            Coefficients, shape (n_samples, n_concepts)
        """
        if self._decomposer is None:
            raise ValueError("Factorizer must be fitted before encoding")
        
        # Cast activations to concept bank dtype for consistency
        if activations.dtype != self._concept_bank_w.dtype:
            activations = activations.astype(self._concept_bank_w.dtype)
        
        return self._decomposer.transform(activations)
    
    def encode_differentiable(self, activations):
        """
        Differentiable encoding (not implemented in base class).
        
        This method should be overridden by framework-specific subclasses.
        
        Parameters
        ----------
        activations : Tensor
            Activations to encode
        
        Returns
        -------
        Tensor
            Coefficients
        
        Raises
        ------
        NotImplementedError
            Always raised, must be implemented by subclasses
        """
        raise NotImplementedError(
            "encode_differentiable must be implemented by framework-specific subclasses"
        )
    
    def decode(self, coefficients) -> Union[np.ndarray, "torch.Tensor", "tf.Tensor"]:
        """
        Decode coefficients to activations via matrix multiplication.
        
        This operation is naturally differentiable when using tensors.
        
        Parameters
        ----------
        coefficients : array or Tensor
            Coefficients to decode, shape (n_samples, n_concepts)
        
        Returns
        -------
        array or Tensor
            Reconstructed activations, shape (n_samples, n_features)
        """
        if self._concept_bank_w is None:
            raise ValueError("Factorizer must be fitted before decoding")
        
        if isinstance(coefficients, np.ndarray):
            return coefficients @ self._concept_bank_w
        else:
            concept_bank_tensor = type(coefficients)(self._concept_bank_w)
            return coefficients @ concept_bank_tensor
    
    def get_concept_bank(self) -> np.ndarray:
        """
        Get the concept bank (W matrix from NMF).
        
        Returns
        -------
        np.ndarray
            Concept bank, shape (n_concepts, n_features)
        """
        if self._concept_bank_w is None:
            raise ValueError("Factorizer must be fitted before getting concept bank")
        return self._concept_bank_w
    
    @property
    def is_fitted(self) -> bool:
        """
        Check if the factorizer has been fitted.
        
        Returns
        -------
        bool
            True if fitted, False otherwise
        """
        return self._decomposer is not None and self._concept_bank_w is not None
    
    @property
    def requires_positive_activations(self) -> bool:
        """
        NMF requires non-negative activations.
        
        Returns
        -------
        bool
            True
        """
        return True
