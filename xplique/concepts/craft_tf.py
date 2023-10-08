
"""
CRAFT Module for Tensorflow
"""

from typing import Callable, Tuple, Optional
import tensorflow as tf
import numpy as np

from .craft import BaseCraft
from .craft_manager import BaseCraftManager

class CraftTf(BaseCraft):

    """
    Class implementing the CRAFT Concept Extraction Mechanism on Tensorflow.

    Parameters
    ----------
    input_to_latent_model
        The first part of the model taking an input and returning
        positive activations, g(.) in the original paper.
        Must be a Tensorflow model (tf.keras.engine.base_layer.Layer) accepting
        data of shape (n_samples, height, width, channels).
    latent_to_logit_model
        The second part of the model taking activation and returning
        logits, h(.) in the original paper.
        Must be a Tensorflow model (tf.keras.engine.base_layer.Layer).
    number_of_concepts
        The number of concepts to extract. Default is 20.
    batch_size
        The batch size to use during training and prediction. Default is 64.
    patch_size
        The size of the patches to extract from the input data. Default is 64.
    """
    def __init__(self, input_to_latent_model : Callable,
                       latent_to_logit_model : Callable,
                       number_of_concepts: int = 20,
                       batch_size: int = 64,
                       patch_size: int = 64):
        super().__init__(input_to_latent_model,
                         latent_to_logit_model,
                         number_of_concepts,
                         batch_size)
        self.patch_size = patch_size

        # Check model type
        keras_base_layer = tf.keras.Model

        is_tf_model = issubclass(type(input_to_latent_model), keras_base_layer) & \
                      issubclass(type(latent_to_logit_model), keras_base_layer)
        if not is_tf_model:
            raise TypeError('input_to_latent_model and latent_to_logit_model are not '\
                            'Tensorflow models')

    def _latent_predict(self, inputs: tf.Tensor):
        """
        Compute the embedding space using the 1st model `input_to_latent_model`.

        Parameters
        ----------
        inputs
            Input data of shape (n_samples, height, width, channels).

        Returns
        -------
        activations
            The latent activations of shape (n_samples, height, width, channels)
        """
        if len(inputs.shape) == 3:
            # add an extra dim in case we get only 1 image to predict
            inputs = tf.expand_dims(inputs, 0)
        return self.input_to_latent_model.predict(inputs, batch_size=self.batch_size, verbose=False)

    def _logit_predict(self, activations: np.ndarray):
        """
        Compute logits from activations using the 2nd model `latent_to_logit_model`.

        Parameters
        ----------
        activations
            Activations produced by the 1st model `input_to_latent_model`,
            of shape (n_samples, height, width, channels).

        Returns
        -------
        logits
            The logits of shape (n_samples, n_classes)
        """
        return self.latent_to_logit_model.predict(activations,
                                                  batch_size=self.batch_size,
                                                  verbose=False)

    def _extract_patches(self, inputs: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Extract patches (crops) from the input images, and compute their embeddings.

        Parameters
        ----------
        inputs
            Input images (n_samples, height, width, channels).

        Returns
        -------
        patches
            The patches (n_patches, height, width, channels).
        activations
            The patches activations (n_patches, channels).
        """

        strides = int(self.patch_size * 0.80)
        patches = tf.image.extract_patches(images=inputs,
                                           sizes=[1, self.patch_size, self.patch_size, 1],
                                           strides=[1, strides, strides, 1],
                                           rates=[1, 1, 1, 1],
                                           padding='VALID')
        patches = tf.reshape(patches, (-1, self.patch_size, self.patch_size, inputs.shape[-1]))

        # encode the patches and obtain the activations
        input_width, input_height = inputs.shape[1], inputs.shape[2]
        activations = self._latent_predict(tf.image.resize(patches,
                                                           (input_width, input_height),
                                                           method="bicubic"))
        assert np.min(activations) >= 0.0, "Activations must be positive."

        # if the activations have shape (n_samples, height, width, n_channels),
        # apply average pooling
        if len(activations.shape) == 4:
            # activations: (N, H, W, C)
            activations = tf.reduce_mean(activations, axis=(1, 2))

        return patches, activations

    def _to_np_array(self, inputs: tf.Tensor, dtype: type):
        """
        Converts a Tensorflow tensor into a numpy array.
        """
        return np.array(inputs, dtype)


class CraftManagerTf(BaseCraftManager):
    """
    Class implementing the CraftManager on Tensorflow.
    This manager creates one CraftTf instance per class to explain.

    Parameters
    ----------
    input_to_latent_model
        The first part of the model taking an input and returning
        positive activations, g(.) in the original paper.
        Must return positive activations.
    latent_to_logit_model
        The second part of the model taking activation and returning
        logits, h(.) in the original paper.
    inputs
        Input data of shape (n_samples, height, width, channels).
        (x1, x2, ..., xn) in the paper.
    labels
        Labels of the inputs of shape (n_samples, class_id)
    list_of_class_of_interest
        A list of the classes id to explain. The manager will instanciate one
        CraftTf object per element of this list.
    number_of_concepts
        The number of concepts to extract. Default is 20.
    batch_size
        The batch size to use during training and prediction. Default is 64.
    patch_size
        The size of the patches (crops) to extract from the input data. Default is 64.
    """
    def __init__(self, input_to_latent_model : Callable,
                    latent_to_logit_model : Callable,
                    inputs : np.ndarray,
                    labels : np.ndarray,
                    list_of_class_of_interest : Optional[list] = None,
                    number_of_concepts: int = 20,
                    batch_size: int = 64,
                    patch_size: int = 64):

        super().__init__(input_to_latent_model, latent_to_logit_model,
                         inputs, labels, list_of_class_of_interest)

        self.craft_instances = {}
        for class_of_interest in self.list_of_class_of_interest:
            craft = CraftTf(input_to_latent_model, latent_to_logit_model,
                            number_of_concepts, batch_size, patch_size)
            self.craft_instances[class_of_interest] = craft

    def compute_predictions(self):
        """
        Compute the predictions for the current dataset, using the 2 models
        input_to_latent_model and latent_to_logit_model chained.

        Returns
        -------
        y_preds
            the predictions
        """
        y_preds = np.array(tf.argmax(self.latent_to_logit_model.predict(
                            self.input_to_latent_model.predict(self.inputs)), 1))
        return y_preds
