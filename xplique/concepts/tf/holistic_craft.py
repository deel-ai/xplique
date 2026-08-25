"""TensorFlow-specific wrapper for HolisticCraft."""

from typing import Any, Optional, Union

import numpy as np
import tensorflow as tf

from xplique.concepts.factorizer import ConceptFactorizer
from xplique.utils_functions.object_detection.tf.box_model_wrapper import (
    _pad_and_stack_box_predictions,
)

from ..holistic_craft import ConceptDecoder, ConceptLocalizer, HolisticCraft
from ..latent_extractor import LatentData
from .factorizer import TfSklearnNMFFactorizer
from .latent_extractor import TfLatentExtractor as LatentExtractor


class HolisticCraftTf(HolisticCraft):
    """
    TensorFlow-specific implementation of CRAFT for holistic model explanations.

    This class is a thin wrapper around the framework-agnostic base class.
    All core functionality is inherited from HolisticCraft.

    Parameters
    ----------
    latent_extractor
        TensorFlow latent extractor for the model.
    number_of_concepts
        Number of concepts to extract, by default 20.
    factorizer
        Optional factorizer instance. If None, creates a TfSklearnNMFFactorizer
        with alpha_W=1e-2 and max_iter=200
    """

    def __init__(
        self,
        latent_extractor: LatentExtractor,
        number_of_concepts: int = 20,
        factorizer: Optional[ConceptFactorizer] = None,
    ) -> None:
        """
        Initialize the TensorFlow CRAFT wrapper.

        Parameters
        ----------
        latent_extractor
            TensorFlow latent extractor for the model
        number_of_concepts
            Number of concepts to extract (default: 20)
        factorizer
            Optional factorizer instance. If None, creates a TfSklearnNMFFactorizer
            with alpha_W=1e-2 and max_iter=200
        """

        # Create TensorFlow-specific factorizer if none provided
        if factorizer is None:
            factorizer = TfSklearnNMFFactorizer(
                n_components=number_of_concepts, alpha_W=1e-2, max_iter=200
            )

        super().__init__(latent_extractor, number_of_concepts, device=None, factorizer=factorizer)
        self.framework = "tf"
        self._framework_module = tf

    def latent_to_concept_differentiable(self, latent_data: LatentData) -> tf.Tensor:
        """
        Transform latent data to concept coefficients with gradient preservation.

        TensorFlow-specific implementation using a differentiable non-negative
        optimization solver. Maintains the gradient tape for computing
        attributions with respect to concepts.

        Parameters
        ----------
        latent_data
            Single image's latent representation containing activations

        Returns
        -------
        coeffs_u
            Concept coefficients as TensorFlow tensor with gradients preserved

        Raises
        ------
        ValueError
            If latent_data is not a single LatentData instance
        NotFittedError
            If fit() has not been called yet
        """
        if not isinstance(latent_data, LatentData):
            raise ValueError(
                f"latent_to_concept_differentiable() only accepts a single "
                f"LatentData as input, got {type(latent_data)}"
            )
        self.check_if_fitted()

        # Get activations as tensors with gradients preserved
        activations = latent_data.get_activations(as_numpy=False, keep_gradients=True)

        # Ensure we have TensorFlow tensors
        if not isinstance(activations, tf.Tensor):
            activations = tf.convert_to_tensor(activations)

        activations_original_shape = activations.shape[:-1]
        activations_flat = tf.reshape(activations, (-1, activations.shape[-1]))

        # Use factorizer's differentiable encoding
        coeffs_u = self.factorizer.encode_differentiable(activations_flat)

        # Reshape back to original dimensions
        coeffs_u = tf.reshape(coeffs_u, tf.concat([activations_original_shape, [-1]], axis=0))
        return coeffs_u

    def _to_numpy(self, tensor: Union[tf.Tensor, np.ndarray]) -> np.ndarray:
        """
        Convert TensorFlow tensor to numpy array.

        Parameters
        ----------
        tensor
            TensorFlow tensor or numpy array

        Returns
        -------
        array
            Numpy array
        """
        if isinstance(tensor, np.ndarray):
            return tensor
        return tensor.numpy()

    def _to_tensor(self, array: np.ndarray, dtype: Optional[tf.DType] = None) -> tf.Tensor:
        """
        Convert numpy array to TensorFlow tensor.

        Parameters
        ----------
        array
            Numpy array to convert
        dtype
            Target TensorFlow dtype (e.g., tf.float32)

        Returns
        -------
        tensor
            TensorFlow tensor
        """
        kwargs = {}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return tf.convert_to_tensor(array, **kwargs)

    def make_concept_decoder(self, latent_data: LatentData) -> tf.keras.layers.Layer:
        """
        Creates a TensorFlow concept decoder for gradient-based attribution.

        The decoder is a Keras Layer that accepts concept coefficients and returns
        detection predictions. It maintains a reference to the latent_data to
        reconstruct activations during the call.

        Parameters
        ----------
        latent_data
            Single image's latent representation

        Returns
        -------
        decoder
            ConceptDecoderTf instance (Keras Layer) with call method
        """

        return ConceptDecoderTf(self, latent_data)

    def make_concept_localizer(
        self,
        concept_reducer: Union[str, Any] = "mean",
    ) -> tf.keras.layers.Layer:
        """Create a TensorFlow concept localizer for black-box attribution.

        Parameters
        ----------
        concept_reducer
            Reduction from coefficient maps to one scalar score per concept.

        Returns
        -------
        localizer
            Keras layer returning a tensor with shape ``(batch_size, K)``.
        """
        return ConceptLocalizerTf(self, concept_reducer)


class ConceptLocalizerTf(tf.keras.layers.Layer, ConceptLocalizer):
    """TensorFlow concept localizer layer."""

    def __init__(
        self,
        parent_craft: HolisticCraft,
        concept_reducer: Union[str, Any] = "mean",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        ConceptLocalizer.__init__(self, parent_craft, concept_reducer)

    def call(self, inputs: Any) -> tf.Tensor:
        """Return reduced concept scores for a batch of inputs."""
        scores = self._compute_scores(inputs)
        return tf.convert_to_tensor(scores, dtype=tf.float32)


class ConceptDecoderTf(tf.keras.layers.Layer, ConceptDecoder):
    """
    TensorFlow concept decoder layer.

    Converts concept coefficients back to object detection predictions by
    reconstructing activations and passing them through the decoder network.

    Parameters
    ----------
    latent_data
        Image-specific latent representation to use for decoding
    **kwargs
        Additional keyword arguments for Keras Layer
    """

    def __init__(self, parent_craft: HolisticCraft, latent_data: LatentData, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.parent_craft = parent_craft
        self.latent_data = latent_data

    def call(self, coeffs_u: tf.Tensor) -> tf.Tensor:
        """
        Decode concept coefficients to predictions.

        Parameters
        ----------
        coeffs_u
            Batched concept coefficients.

        Returns
        -------
        logits
            Predictions as a dense batched tensor. Object detections are zero-padded
            to the largest number of boxes in the batch.
        """

        return self._decode(coeffs_u)

    def _predictions_to_tensor(self, predictions) -> tf.Tensor:
        if isinstance(predictions, (list, tuple)):
            return _pad_and_stack_box_predictions(predictions)
        if hasattr(predictions, "to_batched_tensor"):
            return predictions.to_batched_tensor()
        return predictions
