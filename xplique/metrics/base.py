"""
Module related to abstract attribution metric
"""

from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np

from ..commons import numpy_sanitize
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
    """

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
