"""
Module related to abstract explainer
"""

import warnings

import tensorflow as tf
from tensorflow.keras.activations import softmax  # pylint: disable=import-error
from tensorflow.keras.models import Model  # pylint: disable=import-error


class BaseExplanation:
    """
    Base class for explainers.

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

    # in order to avoid re-tracing at each tf.function call,
    # share the reconfigured models between the methods if possible
    _cache_models = {}

    def __init__(self, model, output_layer_index=-1, batch_size=32):
        self.target_layer = model.layers[output_layer_index]
        # sanity check, output layer before softmax
        try:
            if self.target_layer.activation.__name__ == softmax.__name__:
                warnings.warn("Output is after softmax, it is recommended to "
                              "use the layer before.")
        except AttributeError:
            pass

        if (id(model), output_layer_index) not in BaseExplanation._cache_models:
            # reconfigure the model to use the specified output, and store it
            BaseExplanation._cache_models[(id(model), output_layer_index)] \
                = Model(model.input, self.target_layer.output)

        self.model = BaseExplanation._cache_models[(id(model), output_layer_index)]
        self.batch_size = batch_size

    def explain(self, inputs, labels):
        """
        Compute the explanations of the given samples, take care of sanitizing inputs, returning
        a ndarray, and splits the calculation into several batches if necessary.

        Parameters
        ----------
        inputs : ndarray (N, W, H, C)
            Input samples, with N number of samples, W & H the sample dimensions, and C the
            number of channels.
        labels : ndarray(N, L)
            One hot encoded labels to compute for each sample, with N the number of samples, and L
            the number of classes.

        Returns
        -------
        grad_cam : ndarray (N, W, H)
            Explanations computed, with the same shape as the inputs except for the channels.
        """
        raise NotImplementedError()

    def __call__(self, inputs, labels):
        """Explain alias"""
        return self.explain(inputs, labels)

    @staticmethod
    def compute(model, inputs, labels, batch_size, *params):
        """
        Compute the explanation on the given samples.

        Parameters
        ----------
        model : tf.keras.Model
            Model used for computing explanations.
        inputs : ndarray (N, W, H, C)
            Batch of input samples , with N number of samples, W & H the sample dimensions,
            and C the number of channels.
        labels : ndarray (N, L)
            One hot encoded labels to compute for each sample, with N the number of samples, and L
            the number of classes.
        batch_size : int, optional
            Number of samples to explain at once, if None compute all at once.
        params : optional
            Each method has its own specific parameters that will be expected as arguments.

        Returns
        -------
        explanations : tf.Tensor (N, W, H)
            Explanations computed.
        """
        raise NotImplementedError()

    @staticmethod
    @tf.function
    def _gradient(model, inputs, labels):
        """
        Compute gradients for a batch of samples.

        Parameters
        ----------
        model : tf.keras.Model
            Model used for computing gradient.
        inputs : ndarray (N, W, H, C)
            Input samples, with N number of samples, W & H the sample dimensions, and C the
            number of channels.
        labels : ndarray(N, L)
            One hot encoded labels to compute for each sample, with N the number of samples, and L
            the number of classes.

        Returns
        -------
        gradients : tensor (N, W, H, C)
            Gradients computed, with the same shape as the inputs.
        """
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(inputs)
            score = tf.reduce_sum(tf.multiply(model(inputs), labels), axis=1)
        return tape.gradient(score, inputs)

    @staticmethod
    def _batch_gradient(model, inputs, labels, batch_size):
        """
        Compute the gradients of the sample passed, take care of splitting the samples in
        multiple batches if batch_size is specified.

        Parameters
        ----------
        model : tf.keras.Model
            Model used for computing gradient.
        inputs : ndarray (N, W, H, C)
            Input samples, with N number of samples, W & H the sample dimensions, and C the
            number of channels.
        labels : ndarray(N, L)
            One hot encoded labels to compute for each sample, with N the number of samples, and L
            the number of classes.
        batch_size : int, optional
            Number of samples to explain at once, if None compute all at once.

        Returns
        -------
        gradients : tensor (N, W, H, C)
            Gradients computed, with the same shape as the inputs.
        """

        if batch_size is not None:
            dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
            gradients = tf.concat([
                BaseExplanation._gradient(model, x, y)
                for x, y in dataset.batch(batch_size)
            ], axis=0)
        else:
            gradients = BaseExplanation._gradient(model, inputs, labels)

        return gradients
