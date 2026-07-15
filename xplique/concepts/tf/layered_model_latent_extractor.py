"""
TensorFlow latent data and extractor builder for layered models.
"""

from typing import Union

import numpy as np
import tensorflow as tf

from xplique.utils_functions.classification.tf import TfClassifierTensor

from ..latent_extractor import LatentData, LatentExtractorBuilder
from .latent_extractor import TfLatentExtractor


class LayeredLatentData(LatentData):
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

    def __getitem__(self, indices: Union[int, slice]) -> "LayeredLatentData":
        """
        Get a subset of the latent data by indexing.

        Parameters
        ----------
        indices
            Indices or slice to extract from the batch.

        Returns
        -------
        latent_data
            New LayeredLatentData instance with selected samples.
        """
        return LayeredLatentData(self.activations[indices])

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
                activations.numpy() if hasattr(activations, "numpy") else np.array(activations)
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
        cls, model: tf.keras.Model, split_layer: int, batch_size: int = 1, **kwargs
    ) -> "TfLatentExtractor":
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
            If the model cannot be split at split_layer or uses unsupported input/output shapes.
        """
        if not isinstance(model, tf.keras.Model):
            raise ValueError("model must be a tf.keras.Model")
        if not isinstance(split_layer, int) or isinstance(split_layer, bool):
            raise ValueError("split_layer must be an integer layer index")

        layers = list(model.layers)
        if not layers:
            raise ValueError("model must contain at least one layer")
        if not -len(layers) <= split_layer < len(layers):
            raise ValueError(
                f"split_layer must be between {-len(layers)} and {len(layers) - 1}, "
                f"got {split_layer}"
            )

        try:
            model_inputs = tf.nest.flatten(model.inputs)
            model_outputs = tf.nest.flatten(model.outputs)
        except (AttributeError, ValueError) as error:
            raise ValueError("model must be built before creating a latent extractor") from error
        if len(model_inputs) != 1 or len(model_outputs) != 1:
            raise ValueError(
                "LayeredModelExtractorBuilder only supports single-input, single-output models"
            )

        split_layer_obj = layers[split_layer]
        split_outputs = tf.nest.flatten(split_layer_obj.output)
        if len(split_outputs) != 1:
            raise ValueError("split_layer must produce exactly one tensor")
        split_output = split_outputs[0]

        # Every path to the output must pass through split_output. Keras otherwise
        # accepts a bypassed branch and silently reconnects it to the new h input.
        pending_tensors = [model_outputs[0]]
        visited_tensors = set()
        while pending_tensors:
            tensor = pending_tensors.pop()
            if tensor is split_output:
                continue
            if id(tensor) in visited_tensors:
                continue
            visited_tensors.add(id(tensor))

            history = getattr(tensor, "_keras_history", None)
            if history is None:
                continue
            operation = history.operation
            node = operation._inbound_nodes[history.node_index]
            parent_tensors = tf.nest.flatten(node.input_tensors)
            if not parent_tensors:
                raise ValueError(
                    "split_layer must form a graph cut: model outputs must depend only on "
                    "the selected layer output"
                )
            pending_tensors.extend(parent_tensors)

        try:
            # Reuse the Functional graph rather than replaying layers in sequence. This
            # preserves branches and merge layers such as residual Add connections.
            g_model = tf.keras.Model(inputs=model_inputs[0], outputs=split_output)
            h_model = tf.keras.Model(inputs=split_output, outputs=model_outputs[0])
        except ValueError as error:
            raise ValueError(
                "split_layer must form a graph cut: model outputs must depend only on "
                "the selected layer output"
            ) from error

        def g(images: tf.Tensor) -> LayeredLatentData:
            """
            Extract activations from the split layer (bottleneck features).

            Parameters
            ----------
            images
                Input images tensor of shape (batch, height, width, 3).

            Returns
            -------
            latent_data
                LayeredLatentData containing split layer activations.
            """
            activations = g_model(images)
            return LayeredLatentData(activations)

        def h(latent_data: LayeredLatentData) -> tf.Tensor:
            """
            Process latent activations through remaining layers to get logits.

            Parameters
            ----------
            latent_data
                LayeredLatentData containing split layer activations.

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
            latent_data_class=LayeredLatentData,
            output_formatter=TfClassifierTensor.from_predictions,
            batch_size=batch_size,
        )

        return latent_extractor
