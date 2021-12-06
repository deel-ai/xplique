# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Module related to abstract attribution metric
"""
from abc import ABC
from abc import abstractmethod

import numpy as np
import tensorflow as tf

from ..commons import numpy_sanitize
from ..types import Callable
from ..types import Optional
from ..types import Union


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

    def __init__(
        self,
        model: Callable,
        inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
        batch_size: Optional[int] = 64,
    ):
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
    def evaluate(self, explainer: Callable) -> float:
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

    def __call__(self, explainer: Callable) -> float:
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
    def evaluate(self, explanations: Union[tf.Tensor, np.array]) -> float:
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

    def __call__(self, explanations: Union[tf.Tensor, np.array]) -> float:
        """Evaluate alias"""
        return self.evaluate(explanations)
