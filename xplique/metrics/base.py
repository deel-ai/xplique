"""
Module related to abstract attribution metric
"""

from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np

from ..commons import numpy_sanitize, get_inference_function
from ..types import Callable, Optional, Union


class BaseAttributionMetric(ABC):
    """
    Base class for Attribution Metric.

    Parameters
    ----------
    model
        Model used for computing explanations.
    inputs
        Input samples to be explained.
    targets
        One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.
    batch_size
        Number of samples to evaluate at once, if None compute all at once.
    """

    def __init__(self,
                 model: Callable,
                 inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
                 batch_size: Optional[int] = 64):
        self.model = model
        self.inputs, self.targets = numpy_sanitize(inputs, targets)
        self.batch_size = batch_size


class ExplainerMetric(BaseAttributionMetric, ABC):
    """
    Base class for Attribution Metric that require explainer.

    Parameters
    ----------
    model
        Model used for computing explanations.
    inputs
        Input samples to be explained.
    targets
        One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.
    batch_size
        Number of samples to evaluate at once, if None compute all at once.
    """

    @abstractmethod
    def evaluate(self,
                 explainer: Callable) -> float:
        """
        Compute the score of the given explainer.

        Parameters
        ----------
        explainer
            Explainer to call to get explanation for an input and a label.

        Returns
        -------
        score
            Score of the explainer on the inputs.
        """
        raise NotImplementedError()

    def __call__(self,
                 explainer: Callable) -> float:
        """Evaluate alias"""
        return self.evaluate(explainer)


class ExplanationMetric(BaseAttributionMetric, ABC):
    """
    Base class for Attribution Metric that require explanations.

    Parameters
    ----------
    model
        Model used for computing explanations.
    inputs
        Input samples to be explained.
    targets
        One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.
    batch_size
        Number of samples to evaluate at once, if None compute all at once.
    operator
        Function g to explain, g take 3 parameters (f, x, y) and should return a scalar,
        with f the model, x the inputs and y the targets. If None, use the standard
        operator g(f, x, y) = f(x)[y].
    """

    def __init__(self,
                 model: tf.keras.Model,
                 inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
                 batch_size: Optional[int] = 64,
                 operator: Optional[Callable] = None,
                 ):
        # pylint: disable=R0913
        super().__init__(model, inputs, targets, batch_size)

        # define the inference function according to the model type
        self.inference_function, self.batch_inference_function = \
            get_inference_function(model, operator)

    @abstractmethod
    def evaluate(self,
                 explanations: Union[tf.Tensor, np.array]) -> float:
        """
        Compute the score of the given explanations.

        Parameters
        ----------
        explanations
            Explanation for the inputs, labels to evaluate.

        Returns
        -------
        score
            Score of the explanations.
        """
        raise NotImplementedError()

    def __call__(self,
                 explanations: Union[tf.Tensor, np.array]) -> float:
        """Evaluate alias"""
        return self.evaluate(explanations)
