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
Stability (or Sensitivity) metrics
"""
import numpy as np
import tensorflow as tf

from ..types import Callable
from ..types import Optional
from ..types import Union
from .base import ExplainerMetric


class AverageStability(ExplainerMetric):
    """
    Used to compute the average sensitivity metric (or stability). This metric ensure that close
    inputs with similar predictions yields similar explanations. For each inputs we randomly
    sample noise to add to the inputs and compute the explanation for the noisy inputs. We then
    get the average distance between the original explanations and the noisy explanations.

    Ref. Bhatt & al., Evaluating and Aggregating Feature-based Model Explanations (2020).
    https://arxiv.org/abs/2005.00631 (def. 2)

    Parameters
    ----------
    model
        Model used for computing metric.
    inputs
        Input samples under study.
    targets
        One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.
    batch_size
        Number of samples to explain at once, if None compute all at once.
    radius
        Radius defining the neighborhood of the inputs with respect to l1 distance.
    distance
        Distance metric between the explanations.
    nb_samples
        Number of different neighbors points to try on each input to measure the stability.
    """

    def __init__(
        self,
        model: Callable,
        inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
        batch_size: Optional[int] = 64,
        radius: float = 0.1,
        distance: Union[str, Callable] = "l2",
        nb_samples: int = 20,
    ):
        # pylint: disable=R0913
        super().__init__(model, inputs, targets, batch_size)
        self.radius = radius
        self.nb_samples = nb_samples

        if distance == "l1":
            self.distance = lambda x, y: tf.reduce_sum(tf.abs(x - y))
        elif distance == "l2":
            self.distance = lambda x, y: tf.reduce_sum((x - y) ** 2.0)
        elif hasattr(distance, "__call__"):
            self.distance = distance
        else:
            raise ValueError(f"{distance} is not a valid distance.")

        # prepare the noisy masks that will be used to generate the neighbors
        nb_variables = np.prod(inputs.shape[1:])
        self.noisy_masks = tf.random.uniform(
            (nb_samples, *inputs.shape[1:]), 0, 1.0 / nb_variables
        )

    def evaluate(
        self,
        explainer: Callable,
        base_explanations: Optional[Union[tf.Tensor, np.ndarray]] = None,
    ) -> float:
        # pylint: disable=W0221
        """
        Evaluate the fidelity score.

        Parameters
        ----------
        explainer
            Explainer or Explanations associated to each inputs.
        base_explanations
            Explanation for the inputs under study. Calculates them automatically if they are
            not provided.

        Returns
        -------
        stability_score
            Average distance between the explanations
        """
        if base_explanations is None:
            base_explanations = np.array(explainer(self.inputs, self.targets))

        distances = []
        for inp, label, phi in zip(self.inputs, self.targets, base_explanations):
            label = tf.repeat(label[None, :], self.nb_samples, 0)
            neighbors = inp + self.noisy_masks
            phis_neighbors = np.array(explainer(neighbors, label))

            # compute the distances between each new explanations
            avg_dist = np.mean([self.distance(phi_n, phi) for phi_n in phis_neighbors])
            distances.append(avg_dist)

        stability_score = np.mean(distances)

        return float(stability_score)
