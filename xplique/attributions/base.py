"""
Module related to abstract explainer
"""

from abc import ABC, abstractmethod
import warnings

import tensorflow as tf


class BaseExplainer(ABC):
    """
    Base class for explainers.

    Parameters
    ----------
    model : tf.keras.Model
        Model used for computing explanations.
    batch_size : int, optional
        Number of samples to explain at once, if None compute all at once.
    """

    # in order to avoid re-tracing at each tf.function call,
    # share the reconfigured models between the methods if possible
    _cache_models = {}

    def __init__(self, model, batch_size=64):
        model_key = (id(model.input), id(model.output))
        if model_key not in BaseExplainer._cache_models:
            BaseExplainer._cache_models[model_key] = model

        self.model = BaseExplainer._cache_models[model_key]
        self.batch_size = batch_size

    @abstractmethod
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
        explanations : ndarray (N, W, H)
            Explanations computed, with the same shape as the inputs except for the channels.
        """
        raise NotImplementedError()

    def __call__(self, inputs, labels):
        """Explain alias"""
        return self.explain(inputs, labels)


class BlackBoxExplainer(BaseExplainer, ABC):
    """
    Base class for Black-Box explainers.

    Parameters
    ----------
    model : tf.keras.Model
        Model used for computing explanations.
    batch_size : int, optional
        Number of samples to explain at once, if None compute all at once.
    """

    @staticmethod
    @tf.function
    def _predictions(model, inputs, labels):
        """
        Compute predictions scores, only for the label class, for a batch of samples.

        Parameters
        ----------
        model : tf.keras.Model
            Model used for computing predictions.
        inputs : tf.tensor (N, W, H, C)
            Input samples, with N number of samples, W & H the sample dimensions, and C the
            number of channels.
        labels : tf.tensor (N, L)
            One hot encoded labels to compute for each sample, with N the number of samples, and L
            the number of classes.

        Returns
        -------
        scores : tf.tensor (N)
            Predictions scores computed, only for the label class.
        """
        scores = tf.reduce_sum(model(inputs) * labels, axis=-1)
        return scores

    @staticmethod
    def _batch_predictions(model, inputs, labels, batch_size):
        """
        Compute predictions scores, only for the label class, for the samples passed. Take care
        of splitting in multiple batches if batch_size is specified.

        Parameters
        ----------
        model : tf.keras.Model
            Model used for computing predictions score.
        inputs : tf.tensor (N, W, H, C)
            Input samples, with N number of samples, W & H the sample dimensions, and C the
            number of channels.
        labels : tf.tensor (N, L)
            One hot encoded labels to compute for each sample, with N the number of samples, and L
            the number of classes.
        batch_size : int, optional
            Number of samples to predict at once, if None compute all at once.

        Returns
        -------
        scores : tf.tensor (N)
            Predictions scores computed, only for the label class.
        """
        if batch_size is not None:
            dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
            scores = tf.concat([
                BlackBoxExplainer._predictions(model, x_batch, y_batch)
                for x_batch, y_batch in dataset.batch(batch_size)
            ], axis=0)
        else:
            scores = BlackBoxExplainer._predictions(model, inputs, labels)

        return scores


class WhiteBoxExplainer(BlackBoxExplainer, ABC):
    """
    Base class for White-Box explainers.

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

    def __init__(self, model, output_layer_index, batch_size=64):
        # reconfigure the model
        target_layer = model.layers[output_layer_index]
        model = tf.keras.Model(model.input, target_layer.output)

        # sanity check, output layer before softmax
        try:
            if target_layer.activation.__name__ == tf.keras.activations.softmax.__name__:
                warnings.warn("Output is after softmax, it is recommended to "
                              "use the layer before.")
        except AttributeError:
            pass

        super().__init__(model, batch_size)

    @staticmethod
    @tf.function
    def _gradient(model, inputs, labels):
        """
        Compute gradients for a batch of samples.

        Parameters
        ----------
        model : tf.keras.Model
            Model used for computing gradient.
        inputs : tf.tensor (N, W, H, C)
            Input samples, with N number of samples, W & H the sample dimensions, and C the
            number of channels.
        labels : tf.tensor (N, L)
            One hot encoded labels to compute for each sample, with N the number of samples, and L
            the number of classes.

        Returns
        -------
        gradients : tf.tensor (N, W, H, C)
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
        inputs : tf.tensor (N, W, H, C)
            Input samples, with N number of samples, W & H the sample dimensions, and C the
            number of channels.
        labels : tf.tensor (N, L)
            One hot encoded labels to compute for each sample, with N the number of samples, and L
            the number of classes.
        batch_size : int, optional
            Number of samples to explain at once, if None compute all at once.

        Returns
        -------
        gradients : tf.tensor (N, W, H, C)
            Gradients computed, with the same shape as the inputs.
        """
        if batch_size is not None:
            dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
            gradients = tf.concat([
                WhiteBoxExplainer._gradient(model, x, y)
                for x, y in dataset.batch(batch_size)
            ], axis=0)
        else:
            gradients = WhiteBoxExplainer._gradient(model, inputs, labels)

        return gradients
