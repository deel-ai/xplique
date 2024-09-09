import tensorflow as tf
import numpy as np

from .base import WhiteBoxExplainer, sanitize_input_output
from ..commons import Tasks, find_layer
from ..types import Union, Optional, OperatorSignature


_normalization_dict = {
    'max_min': lambda x: (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x)),
    'sigmoid': lambda x: tf.nn.sigmoid(x),
    'max': lambda x: x / tf.reduce_max(x),
}


_loss_dict = {
    'mse': lambda l, p: tf.reduce_mean(tf.square(l - p)),
    'l1': lambda l, p: tf.reduce_mean(tf.abs(l - p)),
}


class OptiCAM(WhiteBoxExplainer):
    """
    Used to compute the Opti-CAM visualization method.

    Only for Convolutional Networks.

    Ref. Zhang & al., Opti-CAM: Optimizing saliency maps for interpretability (2024).
    https://www.sciencedirect.com/science/article/pii/S1077314224001826?via%3Dihub

    Parameters
    ----------
    model
        The model from which we want to obtain explanations
    output_layer
        Layer to target for the outputs (e.g logits or after softmax).
        If an `int` is provided it will be interpreted as a layer index.
        If a `string` is provided it will look for the layer name.

        Default to the last layer.
        It is recommended to use the layer before Softmax.
    batch_size
        Number of inputs to explain at once, if None compute all at once.
    operator
        Function g to explain, g take 3 parameters (f, x, y) and should return a scalar,
        with f the model, x the inputs and y the targets. If None, use the standard
        operator g(f, x, y) = f(x)[y].
    conv_layer
        Layer to target for Grad-CAM algorithm.
        If an int is provided it will be interpreted as a layer index.
        If a string is provided it will look for the layer name.
    n_iters
        Number of iterations to optimize the CAM.
    """
    def __init__(self,
                 model: tf.keras.Model,
                 output_layer: Optional[Union[str, int]] = None,
                 batch_size: Optional[int] = 32,
                 operator: Optional[Union[Tasks, str, OperatorSignature]] = None,
                 conv_layer: Optional[Union[str, int]] = None,
                 loss_type: str = "l1",
                 n_iters: int = 100,
                 normalization: Optional[str] = 'max_min'):
        assert n_iters > 0, "n_iters should be a positive integer"
        if normalization not in _normalization_dict:
            raise ValueError(f"normalization should be in {_normalization_dict.keys()}")
        if loss_type not in _loss_dict:
            raise ValueError(f"loss_type should be in {_loss_dict.keys()}")
        super().__init__(model, output_layer, batch_size, operator)
        self.n_iters = n_iters
        self.normalize = _normalization_dict[normalization]
        self.loss_fn = _loss_dict[loss_type]

        # find the layer to apply opti-cam
        if conv_layer is not None:
            self.conv_layer = find_layer(model, conv_layer)
        else:
            # no conv_layer specified, assuming default procedure : the last conv layer
            self.conv_layer = next(
                layer for layer in model.layers[::-1] if hasattr(layer, 'filters'))

        # create a model that outputs the feature maps of the conv layer (+ relu)
        relu = tf.keras.layers.ReLU()
        self.model = tf.keras.Model(model.input, [relu(self.conv_layer.output), self.model.output])

    @sanitize_input_output
    def explain(self,
                inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                targets: Optional[Union[tf.Tensor, np.ndarray]] = None) -> tf.Tensor:
        """
        Compute and resize explanations to match inputs shape.
        Accept Tensor, numpy array or tf.data.Dataset (in that case targets is None)

        Parameters
        ----------
        inputs
            Dataset, Tensor or Array. Input samples to be explained.
            If Dataset, targets should not be provided (included in Dataset).
            Expected shape among (N, W), (N, T, W), (N, H, W, C).
            More information in the documentation.
        targets
            Tensor or Array. One-hot encoding of the model's output from which an explanation
            is desired. One encoding per input and only one output at a time. Therefore,
            the expected shape is (N, output_size).
            More information in the documentation.

        Returns
        -------
        opti_cams
            Opti-CAM explanations, same shape as the inputs except for the channels.
        """
        # pylint: disable=E1101
        batch_size = self.batch_size if self.batch_size is not None else len(inputs)

        # optimization loop is done for each batch separately
        opti_cams = None
        for x_batch, y_batch in tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(
                batch_size):
            # initialize weights and optimization elements
            current_batch_size = x_batch.shape[0]
            weights = tf.Variable(0.5 * tf.ones((current_batch_size, 1, 1, self.conv_layer.output.shape[-1])),
                                  trainable=True, dtype=tf.float32)
            optimizer = tf.keras.optimizers.Adam(0.05)
            for _ in range(self.n_iters):
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(weights)
                    feature_maps, logits = self.model(x_batch)
                    explanations = self._one_step_explanation(x_batch, feature_maps, weights)
                    explanations = tf.map_fn(self.normalize, explanations)
                    x_perturbed = tf.multiply(tf.tile(explanations, [1, 1, 1, x_batch.shape[-1]]), x_batch)
                    _, logits_perturbed = self.model(x_perturbed)
                    logits, logits_perturbed = tf.reduce_sum(logits * y_batch, axis=1), tf.reduce_sum(logits_perturbed * y_batch, axis=1)
                    score = self.loss_fn(logits, logits_perturbed)
                grads = tape.gradient(score, weights)
                optimizer.apply_gradients([(grads, weights)])

            # Generate the final explanations for the batch of images
            explanations = self._one_step_explanation(x_batch, feature_maps, weights)
            explanations = tf.map_fn(self.normalize, explanations)
            opti_cams = explanations if opti_cams is None else tf.concat([opti_cams, explanations], axis=0)

        return opti_cams

    @staticmethod
    @tf.function
    def _one_step_explanation(images: tf.Tensor, feature_maps: tf.Tensor, weights: tf.Tensor):
        """
        Generate the explanation for a single step of the optimization.

        Parameters
        ----------
        images
            Original images that we wish to explain.
        feature_maps
            Feature maps of the convolutional layer for the images.
        weights
            Weights of the feature maps (aka the CAM explanation) for the images.

        Returns
        -------
        overlayed_images
            Images with the explanation overlayed.
        """
        alpha = tf.nn.softmax(weights, axis=-1)
        weighted_explanation = tf.reshape(
            tf.reduce_sum(
                tf.tile(alpha, [1, feature_maps.shape[1], feature_maps.shape[2], 1]) * feature_maps,
                axis=-1
            ),
            (-1, feature_maps.shape[1], feature_maps.shape[2])
        )
        weighted_explanation = tf.expand_dims(weighted_explanation, axis=-1)
        weighted_explanation = tf.image.resize(weighted_explanation, images.shape[1:-1])

        return weighted_explanation
