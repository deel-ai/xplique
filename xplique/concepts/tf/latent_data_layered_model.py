"""
TensorFlow latent data and extractor builder for layered models.
"""
from typing import Union

import tensorflow as tf
import numpy as np

from xplique.utils_functions.classification.tf import TfClassifierFormatter
from ..latent_extractor import LatentData, LatentExtractorBuilder
from .latent_extractor import TfLatentExtractor

class LatentDataLayered(LatentData):
    """
    Stores latent representations (activations) from a layered TensorFlow model.

    This class encapsulates intermediate activations from any layered model
    (ResNet, VGG, DenseNet, etc.) used for classification tasks. It stores
    activations from a single intermediate layer of interest.

    Attributes
    ----------
    activations
        Tensor of intermediate activations from the model. Expected shape depends on
        the extraction layer (e.g., (batch, height, width, channels) for conv layers,
        or (batch, features) for fully connected layers).
    """

    def __init__(self, activations: tf.Tensor):
        """
        Initialize layered model latent data with activations.

        Parameters
        ----------
        activations
            Intermediate activations tensor from the model.
        """
        self.activations = activations

    def __len__(self) -> int:
        """
        Return the batch size from the activations.

        Returns
        -------
        batch_size
            Number of samples in the batch.
        """
        return self.activations.shape[0]

    def __getitem__(self, indices: Union[int, slice]) -> 'LatentDataLayered':
        """
        Get a subset of the latent data by indexing.

        Parameters
        ----------
        indices
            Indices or slice to extract from the batch.

        Returns
        -------
        latent_data
            New LatentDataLayered instance with selected samples.
        """
        return LatentDataLayered(self.activations[indices])

    def get_activations(
            self, as_numpy: bool = True, keep_gradients: bool = False
        ) -> Union[np.ndarray, tf.Tensor]:
        """
        Extract activations as a numpy array or tensor.

        Parameters
        ----------
        as_numpy
            If True, convert tensors to numpy arrays. Default is True.
        keep_gradients
            If True, preserve gradient information. Default is False.

        Returns
        -------
        activations
            Activations as numpy array or TensorFlow tensor.
        """
        activations = self.activations

        if as_numpy:
            activations = (
                activations.numpy()
                if hasattr(activations, 'numpy')
                else np.array(activations)
            )

        return activations

    def set_activations(self, values: Union[tf.Tensor, np.ndarray]) -> None:
        """
        Update activations with new values.

        Parameters
        ----------
        values
            New activation tensor values as tf.Tensor or numpy array.
        """
        if isinstance(values, tf.Tensor):
            self.activations = values
        else:
            # Convert from numpy
            self.activations = tf.constant(values)


class LayeredModelExtractorBuilder(LatentExtractorBuilder):
    """
    Builder for creating LatentExtractor instances for generic layered TensorFlow models.

    This class provides methods to construct a TfLatentExtractor for any layered
    model (ResNet, VGG, DenseNet, etc.) by specifying a split layer. It automatically
    splits the model's forward pass into feature extraction (g) and classification (h).
    """

    # pylint: disable=arguments-differ
    @classmethod
    def build(
            cls,
            model: tf.keras.Model,
            split_layer: int,
            batch_size: int = 1,
            **kwargs) -> 'TfLatentExtractor':
        """
        Build a LatentExtractor for a generic layered classifier model.

        This method creates custom g and h functions that split the model's forward pass
        at a specified layer: g extracts features up to and including the split layer,
        and h processes them through the remaining layers to produce predictions.

        Parameters
        ----------
        model
            TensorFlow/Keras model instance with sequential layers.
        split_layer
            Integer index of the layer to split at. Supports negative indexing
            (e.g., -1 for the last layer, -2 for the second-to-last). The split
            targets the layer at this index, and h processes the remaining layers.
        batch_size
            Batch size for processing. Default is 1.
        **kwargs
            Additional keyword arguments (ignored, for compatibility).

        Returns
        -------
        latent_extractor
            Configured TfLatentExtractor instance for the model.

        Raises
        ------
        ValueError
            If split_layer is not found in the model or is an invalid type.
        """
        # Get the split layer (supports negative indexing)
        split_layer_obj = model.layers[split_layer]

        # Create sub-model from input to split layer output
        g_model = tf.keras.Model(inputs=model.input, outputs=split_layer_obj.output)

        # For h, we need to create a model from split layer's output to final output
        # We'll create an input layer matching the split layer's output shape
        h_input_shape = split_layer_obj.output.shape[1:]  # Remove batch dimension
        h_input = tf.keras.Input(shape=h_input_shape)

        # Get layers after the split
        layers_after_split = model.layers[model.layers.index(split_layer_obj) + 1:]

        # Build h_model by connecting layers sequentially
        x = h_input
        for layer in layers_after_split:
            x = layer(x)

        h_model = tf.keras.Model(inputs=h_input, outputs=x)

        def g(images: tf.Tensor) -> LatentDataLayered:
            """
            Extract activations from the split layer (bottleneck features).

            Parameters
            ----------
            images
                Input images tensor of shape (batch, height, width, 3).

            Returns
            -------
            latent_data
                LatentDataLayered containing split layer activations.
            """
            activations = g_model(images)
            return LatentDataLayered(activations)

        def h(latent_data: LatentDataLayered) -> tf.Tensor:
            """
            Process latent activations through remaining layers to get logits.

            Parameters
            ----------
            latent_data
                LatentDataLayered containing split layer activations.

            Returns
            -------
            logits
                Classification logits tensor of shape (batch, num_classes).
            """
            return h_model(latent_data.activations)

        latent_extractor = TfLatentExtractor(
            model,
            g,
            h,
            latent_data_class=LatentDataLayered,
            output_formatter=TfClassifierFormatter(),
            batch_size=batch_size)

        return latent_extractor
