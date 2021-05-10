"""
Module related to DeconvNet method
"""

from .base import BaseExplanation
from ..utils import override_relu_gradient, deconv_relu, sanitize_input_output

class DeconvNet(BaseExplanation):
    """
    Used to compute the DeconvNet method, which modifies the classic Saliency procedure on
    ReLU's non linearities, allowing only the positive gradients (even from negative inputs) to
    pass through.

    Ref. Zeiler & al., Visualizing and Understanding Convolutional Networks (2013).
    https://arxiv.org/abs/1311.2901

    Parameters
    ----------
    model : tf.keras.Model
        Model used for computing explanations.
    output_layer_index : int, optional
        Index of the output layer, default to the last layer, it is recommended to use the layer
        before Softmax (often '-2').
    batch_size : int, optional
        Number of samples to explain at once, if None compute all at once.
    """

    def __init__(self, model, output_layer_index=-1, batch_size=32):
        super().__init__(model, output_layer_index, batch_size)
        self.model = override_relu_gradient(self.model, deconv_relu)

    @sanitize_input_output
    def explain(self, inputs, labels):
        """
        Compute Guided Backpropagation a batch of samples.

        Parameters
        ----------
        inputs : ndarray (N, W, H, C)
            Input samples, with N number of samples, W & H the sample dimensions, and C the
            number of channels.
        labels : ndarray (N, L)
            One hot encoded labels to compute for each sample, with N the number of samples, and L
            the number of classes.

        Returns
        -------
        explanations : ndarray (N, W, H)
            Guided Backpropagation maps.
        """
        gradients = BaseExplanation._batch_gradient(self.model, inputs, labels, self.batch_size)
        return gradients
