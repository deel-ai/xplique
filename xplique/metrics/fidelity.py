"""
Fidelity (or Faithfulness) metrics
"""

from inspect import isfunction

import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr

from .base import ExplanationMetric
from ..commons import batch_predictions_one_hot
from ..types import Union, Callable, Optional


class MuFidelity(ExplanationMetric):
    """
    Used to compute the fidelity correlation metric. This metric ensure there is a correlation
    between a random subset of pixels and their attribution score. For each random subset
    created, we set the pixels of the subset at a baseline state and obtain the prediction score.
    This metric measures the correlation between the drop in the score and the importance of the
    explanation.

    Ref. Bhatt & al., Evaluating and Aggregating Feature-based Model Explanations (2020).
    https://arxiv.org/abs/2005.00631 (def. 3)

    Notes
    -----
    As noted in the original article, the default operation selects pixel-wise subsets
    independently. However, when using medium or high dimensional images, it is recommended to
    select super-pixels, see the grid_size parameter.

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
    grid_size
        If none, compute the original metric, else cut the image in (grid_size, grid_size) and
        each element of the subset will be a super pixel representing one element of the grid.
        You should use this when dealing with medium / large size images.
    subset_percent
        Percent of the image that will be set to baseline.
    baseline_mode
        Value of the baseline state, will be called with the a single input if it is a function.
    nb_samples
        Number of different subsets to try on each input to measure the correlation.
    """

    def __init__(self,
                 model: Callable,
                 inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
                 batch_size: Optional[int] = 64,
                 grid_size: Optional[int] = 9,
                 subset_percent: float = 0.2,
                 baseline_mode: Union[Callable, float] = 0.0,
                 nb_samples: int = 200):
        # pylint: disable=R0913
        super().__init__(model, inputs, targets, batch_size)
        self.grid_size = grid_size
        self.subset_percent = subset_percent
        self.baseline_mode = baseline_mode
        self.nb_samples = nb_samples

        # if unspecified use the original equation (pixel-wise modification)
        self.grid_size = grid_size or inputs.shape[1]
        # cardinal of subset (|S| in the equation)
        self.subset_size = int(self.grid_size ** 2 * self.subset_percent)

        # prepare the random masks that will designate the modified subset (S in original equation)
        # we ensure the masks have exactly `subset_size` pixels set to baseline
        subset_masks = np.random.rand(self.nb_samples, self.grid_size ** 2).argsort(axis=-1) > \
                       self.subset_size
        # and interpolate them if needed
        subset_masks = subset_masks.astype(np.float32).reshape(
            (self.nb_samples, self.grid_size, self.grid_size, 1))
        self.subset_masks = tf.image.resize(subset_masks, inputs.shape[1:-1], method="nearest")

        self.base_predictions = batch_predictions_one_hot(self.model, inputs,
                                                          targets, self.batch_size)

    def evaluate(self,
                 explanations: Union[tf.Tensor, np.ndarray]) -> float:
        """
        Evaluate the fidelity score.

        Parameters
        ----------
        explanations
            Explanation for the inputs, labels to evaluate.

        Returns
        -------
        fidelity_score
            Metric score, average correlation between the drop in score when variables are set
            to a baseline state and the importance of these variables according to the
            explanations.
        """
        explanations = np.array(explanations)
        assert len(explanations) == len(self.inputs), "The number of explanations must be the " \
                                                      "same as the number of inputs"

        correlations = []
        for inp, label, phi, base in zip(self.inputs, self.targets, explanations,
                                         self.base_predictions):
            label = tf.repeat(label[None, :], self.nb_samples, 0)
            baseline = self.baseline_mode(inp) if isfunction(self.baseline_mode) else \
                self.baseline_mode
            # use the masks to set the selected subsets to baseline state
            degraded_inputs = inp * self.subset_masks + (1.0 - self.subset_masks) * baseline
            # measure the two terms that should be correlated
            preds = base - batch_predictions_one_hot(self.model, degraded_inputs,
                                                     label, self.batch_size)

            attrs = tf.reduce_sum(phi * (1.0 - self.subset_masks), (1, 2, 3))
            corr_score = spearmanr(preds, attrs)[0]

            # sanity check: if the model predictions are the same, no variation
            if np.isnan(corr_score):
                corr_score = 0.0

            correlations.append(corr_score)

        fidelity_score = np.mean(correlations)

        return float(fidelity_score)


class CausalFidelity(ExplanationMetric):
    """
    Used to compute the insertion and deletion metrics.

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
    causal_mode
        If 'insertion', the path is baseline to original image, for 'deletion' the path is original
        image to baseline.
    baseline_mode
        Value of the baseline state, will be called with the inputs if it is a function.
    steps
        Number of steps between the start and the end state.
    """

    def __init__(self,
                 model: tf.keras.Model,
                 inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
                 batch_size: Optional[int] = 64,
                 causal_mode: str = "deletion",
                 baseline_mode: Union[float, Callable] = 0.0,
                 steps: int = 10,
                 ):
        super().__init__(model, inputs, targets, batch_size)
        self.causal_mode = causal_mode
        self.baseline_mode = baseline_mode
        self.steps = steps

        self.nb_features = np.prod(inputs.shape[1:-1])
        self.inputs_flatten = inputs.reshape((len(inputs), self.nb_features, inputs.shape[-1]))

    def evaluate(self,
                 explanations: Union[tf.Tensor, np.ndarray]) -> float:
        """
        Evaluate the causal score.

        Parameters
        ----------
        explanations
            Explanation for the inputs, labels to evaluate.

        Returns
        -------
        causal_score
            Metric score, area over the deletion (lower is better) or insertion (higher is
            better) curve.
        """
        explanations = np.array(explanations)
        assert len(explanations) == len(self.inputs), "The number of explanations must be the " \
                                                      "same as the number of inputs"
        # the reference does not specify how to manage the channels of the explanations
        if len(explanations.shape) == 4:
            explanations = np.mean(explanations, -1)

        explanations_flatten = explanations.reshape((len(explanations), -1))

        # for each sample, sort by most important features according to the explanation
        most_important_features = np.argsort(explanations_flatten, axis=-1)[:, ::-1]

        baselines = self.baseline_mode(self.inputs) if isfunction(self.baseline_mode) else \
            np.ones_like(self.inputs, dtype=np.float32) * self.baseline_mode
        baselines_flatten = baselines.reshape(self.inputs_flatten.shape)

        steps = np.linspace(0, self.nb_features, self.steps, dtype=np.int32)

        scores = []
        if self.causal_mode == "deletion":
            start = self.inputs_flatten
            end = baselines_flatten
        elif self.causal_mode == "insertion":
            start = baselines_flatten
            end = self.inputs_flatten
        else:
            raise NotImplementedError(f'Unknown causal mode `{self.causal_mode}`.')

        for step in steps:
            ids_to_flip = most_important_features[:, :step]
            batch_inputs = start.copy()

            for i, ids in enumerate(ids_to_flip):
                batch_inputs[i, ids] = end[i, ids]

            batch_inputs = batch_inputs.reshape((-1, *self.inputs.shape[1:]))

            predictions = batch_predictions_one_hot(self.model, batch_inputs,
                                                    self.targets, self.batch_size)
            scores.append(predictions)

        auc = np.trapz(np.mean(scores, -1), steps / self.nb_features)

        return auc


class Deletion(CausalFidelity):
    """
    The deletion metric measures the drop in the probability of a class as important pixels (given
    by the saliency map) are gradually removed from the image. A sharp drop, and thus a small
    area under the probability curve, are indicative of a good explanation.

    Ref. Petsiuk & al., RISE: Randomized Input Sampling for Explanation of Black-box Models (2018).
    https://arxiv.org/pdf/1806.07421.pdf

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
    baseline_mode
        Value of the baseline state, will be called with the inputs if it is a function.
    steps
        Number of steps between the start and the end state.
    """

    def __init__(self,
                 model: tf.keras.Model,
                 inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
                 batch_size: Optional[int] = 64,
                 baseline_mode: Union[float, Callable] = 0.0,
                 steps: int = 10,
                 ):
        super().__init__(model, inputs, targets, batch_size, "deletion", baseline_mode, steps)


class Insertion(CausalFidelity):
    """
    The insertion metric, on the other hand, captures the importance of the pixels in terms of
    their ability to synthesize an image and is measured by the rise in the probability of the
    class of interest as pixels are added according to the generated importance map.

    Ref. Petsiuk & al., RISE: Randomized Input Sampling for Explanation of Black-box Models (2018).
    https://arxiv.org/pdf/1806.07421.pdf

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
    baseline_mode
        Value of the baseline state, will be called with the inputs if it is a function.
    steps
        Number of steps between the start and the end state.
    """

    def __init__(self,
                 model: tf.keras.Model,
                 inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
                 batch_size: Optional[int] = 64,
                 baseline_mode: Union[float, Callable] = 0.0,
                 steps: int = 10,
                 ):
        super().__init__(model, inputs, targets, batch_size, "insertion", baseline_mode, steps)
