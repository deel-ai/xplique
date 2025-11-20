"""PyTorch-specific wrapper for HolisticCraft."""

from typing import Any, Optional, Union

import numpy as np
import torch
from torch import nn

from xplique.wrappers import TorchWrapper

from ..holistic_craft import HolisticCraft
from ..latent_extractor import LatentData
from .factorizer import TorchSklearnNMFFactorizer
from .latent_extractor import TorchLatentExtractor as LatentExtractor


class HolisticCraftTorch(HolisticCraft):
    """
    PyTorch-specific implementation of CRAFT for holistic model explanations.

    This class is a thin wrapper around the framework-agnostic base class.
    All core functionality is inherited from HolisticCraft.

        Parameters
        ----------
        latent_extractor
            PyTorch latent extractor for the model
        number_of_concepts
            Number of concepts to extract (default: 20)
        device
            PyTorch device ('cuda' or 'cpu') (default: 'cuda')
        factorizer
            Optional factorizer instance. If None, creates a TorchSklearnNMFFactorizer
            with alpha_W=1e-2 and max_iter=200
    """

    def __init__(
        self,
        latent_extractor: LatentExtractor,
        number_of_concepts: int = 20,
        device: str = "cuda",
        factorizer: Optional[Any] = None,
    ) -> None:
        """
        Initialize the PyTorch CRAFT wrapper.

        Parameters
        ----------
        latent_extractor
            PyTorch latent extractor for the model
        number_of_concepts
            Number of concepts to extract (default: 20)
        device
            PyTorch device ('cuda' or 'cpu') (default: 'cuda')
        factorizer
            Optional factorizer instance. If None, creates a TorchSklearnNMFFactorizer
            with alpha_W=1e-2 and max_iter=200
        """
        # Create PyTorch-specific factorizer if none provided
        if factorizer is None:
            factorizer = TorchSklearnNMFFactorizer(
                n_components=number_of_concepts,
                alpha_W=1e-2,
                max_iter=200
            )

        super().__init__(latent_extractor, number_of_concepts, device, factorizer)
        self.framework = 'torch'
        self._framework_module = torch

    def transform_latent_differentiable(self, latent_data: LatentData) -> torch.Tensor:
        """
        Transform latent data to concept coefficients with gradient preservation.

        PyTorch-specific implementation using differentiable least squares solver.
        Ensures gradients flow through the concept projection for attribution methods.

        Parameters
        ----------
        latent_data
            Single image's latent representation containing activations

        Returns
        -------
        coeffs_u
            Concept coefficients as PyTorch tensor with gradients preserved

        Raises
        ------
        ValueError
            If latent_data is not a single LatentData instance
        NotFittedError
            If fit() has not been called yet
        """
        if not isinstance(latent_data, LatentData):
            raise ValueError(
                f"transform_latent_differentiable() only accepts a single "
                f"LatentData as input, got {type(latent_data)}")
        self.check_if_fitted()

        # Get activations as tensors with gradients preserved
        activations = latent_data.get_activations(as_numpy=False, keep_gradients=True)

        # Ensure we have PyTorch tensors with gradients enabled
        if not isinstance(activations, torch.Tensor):
            activations = torch.tensor(activations, requires_grad=True)
        elif not activations.requires_grad:
            activations = activations.clone().detach().requires_grad_(True)

        activations_original_shape = activations.shape[:-1]
        activations_flat = activations.reshape(-1, activations.shape[-1])

        # Use factorizer's differentiable encoding
        coeffs_u = self.factorizer.encode_differentiable(activations_flat)

        coeffs_u = coeffs_u.reshape(*activations_original_shape, -1)
        return coeffs_u

    def _to_numpy(self, tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Convert PyTorch tensor to numpy array.

        Parameters
        ----------
        tensor
            PyTorch tensor or numpy array

        Returns
        -------
        array
            Numpy array detached from computational graph
        """
        if isinstance(tensor, np.ndarray):
            return tensor
        return tensor.detach().cpu().numpy()

    def _to_tensor(self, array: np.ndarray, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Convert numpy array to PyTorch tensor.

        Parameters
        ----------
        array
            Numpy array to convert
        dtype
            Target PyTorch dtype (e.g., torch.float32)

        Returns
        -------
        tensor
            PyTorch tensor on the specified device
        """
        kwargs = {'device': self.device}
        if dtype is not None:
            kwargs['dtype'] = dtype
        return torch.tensor(array, **kwargs)

    def make_concept_decoder(self, latent_data: LatentData) -> TorchWrapper:
        """
        Creates a PyTorch concept decoder for gradient-based attribution.

        The decoder is a PyTorch nn.Module that accepts concept coefficients and
        returns detection predictions. It maintains a reference to the latent_data
        to reconstruct activations during the forward pass.

        Parameters
        ----------
        latent_data
            Single image's latent representation

        Returns
        -------
        decoder
            ConceptDecoderTorch instance (nn.Module) with forward method, wrapped in TorchWrapper
        """
        parent_craft = self

        class ConceptDecoderTorch(nn.Module):
            """
            PyTorch concept decoder module.

            Converts concept coefficients back to object detection predictions by
            reconstructing activations and passing them through the decoder network.

            Parameters
            ----------
            latent_data
                Image-specific latent representation to use for decoding
            """

            def __init__(self, latent_data: LatentData) -> None:
                super().__init__()
                self.latent_data = latent_data

            def set_latent_data(self, latent_data: LatentData) -> None:
                """
                Update the latent data for this decoder.

                Parameters
                ----------
                latent_data
                    New latent representation to use
                """
                self.latent_data = latent_data

            def forward(self, coeffs_u: torch.Tensor) -> torch.Tensor:
                """
                Decode concept coefficients to predictions.

                Parameters
                ----------
                coeffs_u
                    Concept coefficients with batch size 1

                Returns
                -------
                logits
                    Detection predictions as batched tensor

                Raises
                ------
                ValueError
                    If coeffs_u batch size is not 1
                """
                if coeffs_u.shape[0] != 1:
                    raise ValueError(
                        f"ConceptDecoder.forward() only accepts coeffs_u with "
                        f"batch size 1, got {coeffs_u.shape}")
                nbc_tensor = parent_craft.decode(self.latent_data, coeffs_u)
                logits = nbc_tensor.to_batched_tensor()
                return logits

        torch_decoder = ConceptDecoderTorch(latent_data)
        wrapped_decoder = TorchWrapper(torch_decoder.eval(),
                                       device=self.device,
                                       is_channel_first=False)
        return wrapped_decoder
