"""
Module related to abstract explainer
"""

from abc import ABC, abstractmethod
import warnings

import tensorflow as tf
import numpy as np

from ..types import Callable, Dict, Tuple, Union, Optional, OperatorSignature
from ..commons import Tasks
from ..commons import (find_layer, tensor_sanitize, get_inference_function,
                      get_gradient_functions, no_gradients_available)
from ..wrappers import TorchWrapper


def sanitize_input_output(explanation_method: Callable):
    """
    Wrap a method explanation function to ensure tf.Tensor as inputs,
    and as output

    explanation_method
        Function to wrap, should return an tf.tensor.
    """
    def sanitize(self, inputs: Union[tf.data.Dataset, tf.Tensor, np.array],
                 targets: Optional[Union[tf.Tensor, np.array]],
                 *args):
        # ensure we have tf.tensor
        inputs, targets = tensor_sanitize(inputs, targets)
        # then enter the explanation function
        return explanation_method(self, inputs, targets, *args)

    return sanitize


class BlackBoxExplainer(ABC):
    """
    Base class for Black-Box explainers.

    Parameters
    ----------
    model
        The model from which we want to obtain explanations
    batch_size
        Number of pertubed samples to explain at once, if None compute all at once.
    operator
        Function g to explain, g take 3 parameters (f, x, y) and should return a scalar,
        with f the model, x the inputs and y the targets. If None, use the standard
        operator g(f, x, y) = f(x)[y].
    """

    # in order to avoid re-tracing at each tf.function call,
    # share the reconfigured models between the methods if possible
    _cache_models: Dict[Tuple[int, int], tf.keras.Model] = {}

    def __init__(self, model: Callable, batch_size: Optional[int] = 64,
                operator: Optional[Union[Tasks, str, OperatorSignature]] = None):

        if isinstance(model, TorchWrapper):
            self.model = model
        elif isinstance(model, tf.keras.Model):
            model_key = (id(model.input), id(model.output))
            if model_key not in BlackBoxExplainer._cache_models:
                BlackBoxExplainer._cache_models[model_key] = model
            self.model = BlackBoxExplainer._cache_models[model_key]
        else:
            self.model = model

        self.batch_size = batch_size

        # define the inference function according to the model type
        self.inference_function, self.batch_inference_function = \
            get_inference_function(model, operator)

        # black box method don't have access to the model's gradients
        self.gradient = no_gradients_available
        self.batch_gradient = no_gradients_available


    @abstractmethod
    def explain(self,
                inputs: Union[tf.data.Dataset, tf.Tensor, np.array],
                targets: Optional[Union[tf.Tensor, np.array]] = None) -> tf.Tensor:
        """
        Compute the explanations of the given inputs.
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
        explanations
            Explanation generated by the method.
        """
        raise NotImplementedError()

    def __call__(self,
                 inputs: tf.Tensor,
                 labels: tf.Tensor) -> tf.Tensor:
        """Explain alias"""
        return self.explain(inputs, labels)


class WhiteBoxExplainer(BlackBoxExplainer, ABC):
    """
    Base class for White-Box explainers.

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
        Operator to use to compute the explanation, if None use standard predictions.
    reducer
        String, name of the reducer to use. Either "min", "mean", "max" or "sum".
    """

    def __init__(self,
                model: tf.keras.Model,
                output_layer: Optional[Union[str, int]] = None,
                batch_size: Optional[int] = 64,
                operator: Optional[OperatorSignature] = None,
                reducer: Optional[str] = "mean",):

        super().__init__(model, batch_size, operator)

        if output_layer is not None:
            # reconfigure the model (e.g skip softmax to target logits)
            target_layer = find_layer(model, output_layer)
            model = tf.keras.Model(model.input, target_layer.output)

            # sanity check, output layer before softmax
            try:
                if target_layer.activation.__name__ == tf.keras.activations.softmax.__name__:
                    warnings.warn("Output is after softmax, it is recommended to " +\
                                  "use the layer before.")
            except AttributeError:
                pass

        # check and get gradient function from model and operator
        self.gradient, self.batch_gradient = get_gradient_functions(model, operator)

        self._set_channel_reducer(reducer)

    def _set_channel_reducer(self, reducer: Optional[str]):
        """
        Set the channel reducer to use for the explanations.

        Parameters
        ----------
        reducer
            String, name of the reducer to use. Either "min", "mean", "max" or "sum".
            It can also be None, in that case, no reduction is applied.
        """
        if reducer is None:
            self.reduce = lambda x, axis, keepdims: x
        else:
            self.reduce = getattr(tf, "reduce_" + reducer)

    @staticmethod
    def _harmonize_channel_dimension(explain_method: Callable):
        """
        Makes sure explanations for images have the shape (n, h, w, 1)

        explain_method
            Function to wrap, should return an tf.tensor.
            Explain method from WhiteBoxExplainers.
        """
        def explain(self,
                    inputs: Union[tf.data.Dataset, tf.Tensor, np.array],
                    targets: Optional[Union[tf.Tensor, np.array]] = None) -> tf.Tensor:
            """
            Compute the explanations of the given inputs.
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
            explanations
                Explanation generated by the method.
            """
            explanations = explain_method(self, inputs, targets)

            if len(explanations.shape) == 3 and len(inputs.shape) == 4:
                explanations = tf.expand_dims(explanations, axis=-1)

            if len(explanations.shape) == 4 and explanations.shape[-1] != 1:
                explanations = self.reduce(explanations, axis=-1, keepdims=True)

            return explanations

        return explain
