import contextlib
from abc import ABC, abstractmethod
from typing import Callable, NamedTuple, Optional, Union

import numpy as np

from xplique.utils_functions.object_detection.base.box_formatter import (
    BaseBoxFormatter,
)


class LatentData(ABC):
    """Abstract base class for storing latent representations and activations.

    This class provides an interface for managing intermediate activations
    extracted from a model's latent space. Subclasses must implement methods
    to get and set activations in a framework-specific manner.
    """

    @abstractmethod
    def get_activations(self, as_numpy: bool = True, keep_gradients: bool = False):
        """Retrieve the activations.

        Parameters
        ----------
        as_numpy
            If True, return activations as a NumPy array. If False, return them as a
            framework-native tensor (e.g., TensorFlow or PyTorch). Default is True.
        keep_gradients
            If True, preserve gradients for backpropagation. Default is False.

        Returns
        -------
        activations
            The activations following TensorFlow ordering (batch, height, width, channels).
        """
        raise NotImplementedError("get_activations method must be implemented by subclasses")

    @abstractmethod
    def set_activations(self, values: np.ndarray):
        """Set the activations.

        Parameters
        ----------
        values
            A NumPy array containing the activations, following TensorFlow ordering
            (batch, height, width, channels).
        """
        raise NotImplementedError("set_activations method must be implemented by subclasses")


class EncodedData(NamedTuple):
    """Encoded representation containing latent data and concept coefficients.

    This named tuple is returned by the encode() method and contains both the
    intermediate latent representation and the concept coefficients from NMF
    factorization.

    Attributes
    ----------
    latent_data
        Intermediate latent representation from the model, containing activations
        and metadata needed for decoding back to the original space.
    coeffs_u
        Concept coefficients (U matrix from NMF factorization).
        - When differentiable=False: NumPy array
        - When differentiable=True: Framework-specific tensor (torch.Tensor or
          tf.Tensor) with gradients preserved for backpropagation.
    """
    latent_data: 'LatentData'
    coeffs_u: Union[np.ndarray, 'torch.Tensor', 'tf.Tensor']


class LatentExtractor:
    """Extracts and manages latent representations from models.

    This class provides functionality to split a model into two parts:
    - input_to_latent: extracts intermediate activations
    - latent_to_logit: processes activations to final predictions

    This splitting enables concept-based explanations by allowing manipulation
    of the latent space between the two model parts.

    Parameters
    ----------
    model
        The full model to be split
    input_to_latent_model
        Model segment that processes inputs to latent activations
    latent_to_logit_model
        Model segment that processes latent activations to final outputs
    latent_data_class
        Class for storing latent data (default: LatentData)
    output_formatter
        Optional formatter for model outputs
    batch_size
        Batch size for processing (default: 8)
    """

    def __init__(self, model: Callable,
                 input_to_latent_model: Callable,
                 latent_to_logit_model: Callable,
                 latent_data_class=LatentData,
                 output_formatter: Optional[BaseBoxFormatter] = None,
                 batch_size: int = 8):
        self.model = model
        self.input_to_latent_model = input_to_latent_model
        self.latent_to_logit_model = latent_to_logit_model
        self.latent_data_class = latent_data_class
        self.output_formatter = output_formatter
        self.batch_size = batch_size

    def __call__(self, *args, **kwargs):
        """Make the extractor callable, forwarding to forward method.

        Returns
        -------
        predictions
            Model predictions
        """
        return self.forward(*args, **kwargs)

    def input_to_latent(self, inputs) -> LatentData:
        """Extract latent representations from inputs.

        Parameters
        ----------
        inputs
            Input data to process

        Returns
        -------
        latent_data
            Latent representation containing activations

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def latent_to_logit(self, latent_data: LatentData):
        """Convert latent representations to final predictions.

        Parameters
        ----------
        latent_data
            Latent representation containing activations

        Returns
        -------
        predictions
            Model output predictions

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def forward(self, samples):
        """Forward pass through the full model via latent space.

        Parameters
        ----------
        samples
            Input samples to process

        Returns
        -------
        predictions
            Model predictions
        """
        latent_data = self.input_to_latent(samples)
        return self.latent_to_logit(latent_data)

    @contextlib.contextmanager
    def temporary_force_batch_size(self, batch_size: int):
        """Context manager to temporarily set the batch size.

        This is used during encoding to process one sample at a time, when we need
        to compute explanations per sample.

        Parameters
        ----------
        batch_size
            Temporary batch size to use within the context

        Yields
        ------
        None
            Context for processing with temporary batch size
        """
        old_batch_size = self.batch_size
        self.batch_size = batch_size
        try:
            yield
        finally:
            self.batch_size = old_batch_size


class LatentExtractorBuilder(ABC):
    """Abstract base class ensuring all builders return LatentExtractor instances.

    This class provides a common interface for building LatentExtractor objects
    in a framework-specific manner. Subclasses implement the build method to
    construct appropriate extractors for different model architectures.
    """

    @classmethod
    @abstractmethod
    def build(cls, **kwargs) -> 'LatentExtractor':
        """Build and return a LatentExtractor.

        Parameters
        ----------
        **kwargs
            Framework-specific arguments for building the extractor

        Returns
        -------
        extractor
            Configured LatentExtractor instance

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses
        """
        raise NotImplementedError("build method must be implemented by subclasses")
