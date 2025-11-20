"""TensorFlow-specific wrapper for HolisticCraft."""

from typing import Any, Optional, Union

import numpy as np
import tensorflow as tf

from ..holistic_craft import HolisticCraft
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
        factorizer: Optional[Any] = None,
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
                n_components=number_of_concepts,
                alpha_W=1e-2,
                max_iter=200
            )

        super().__init__(latent_extractor, number_of_concepts, device=None, factorizer=factorizer)
        self.framework = 'tf'
        self._framework_module = tf

    def transform_latent_differentiable(self, latent_data: LatentData) -> tf.Tensor:
        """
        Transform latent data to concept coefficients with gradient preservation.

        TensorFlow-specific implementation using differentiable least squares solver.
        Maintains the gradient tape for computing attributions with respect to concepts.

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
                f"transform_latent_differentiable() only accepts a single "
                f"LatentData as input, got {type(latent_data)}")
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
            kwargs['dtype'] = dtype
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
        parent_craft = self

        class ConceptDecoderTf(tf.keras.layers.Layer):
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

            def __init__(self, latent_data: LatentData, **kwargs: Any) -> None:
                super().__init__(**kwargs)
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

            def call(self, coeffs_u: tf.Tensor) -> tf.Tensor:
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
                        f"ConceptDecoder.call() only accepts coeffs_u with "
                        f"batch size 1, got {coeffs_u.shape}")
                nbc_tensor = parent_craft.decode(self.latent_data, coeffs_u)
                logits = nbc_tensor.to_batched_tensor()
                return logits

        return ConceptDecoderTf(latent_data)
