"""
Fidelity (or Faithfulness) metrics
"""

from inspect import isfunction
from math import floor

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
    max_percentage_perturbed
        Maximum percentage of the input perturbed.
    """

    def __init__(self,
                 model: tf.keras.Model,
                 inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
                 batch_size: Optional[int] = 64,
                 causal_mode: str = "deletion",
                 baseline_mode: Union[float, Callable] = 0.0,
                 steps: int = 10,
                 max_percentage_perturbed: float = 1.0,
                 ): # pylint: disable=R0913
        super().__init__(model, inputs, targets, batch_size)
        self.causal_mode = causal_mode
        self.baseline_mode = baseline_mode
        self.steps = steps

        self.nb_features = np.prod(inputs.shape[1:-1])
        self.inputs_flatten = inputs.reshape((len(inputs), self.nb_features, inputs.shape[-1]))

        assert 0.0 < max_percentage_perturbed <= 1.0, "`max_percentage_perturbed` must be" \
                                                      "in ]O, 1]."
        self.max_percentage_perturbed = max_percentage_perturbed

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

        steps = np.linspace(0, self.nb_features * self.max_percentage_perturbed,self.steps,
                            dtype=np.int32)

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

        # compute auc using trapezoidal rule (the steps are equally reparted)
        avg_scores = np.mean(scores, -1)
        auc = np.mean(avg_scores[:-1] + avg_scores[1:]) * 0.5

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
    max_percentage_perturbed
        Maximum percentage of the input perturbed.
    """

    def __init__(self,
                 model: tf.keras.Model,
                 inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
                 batch_size: Optional[int] = 64,
                 baseline_mode: Union[float, Callable] = 0.0,
                 steps: int = 10,
                 max_percentage_perturbed: float = 1.0
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
    max_percentage_perturbed
        Maximum percentage of the input perturbed.
    """

    def __init__(self,
                 model: tf.keras.Model,
                 inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
                 batch_size: Optional[int] = 64,
                 baseline_mode: Union[float, Callable] = 0.0,
                 steps: int = 10,
                 max_percentage_perturbed: float = 1.0
                 ):
        super().__init__(model, inputs, targets, batch_size, "insertion", baseline_mode, steps)


class CausalFidelityTS(ExplanationMetric):
    """
    Used to compute the insertion and deletion metrics for Time Series explanations.

    Adaptation of the insertion and deletion principle based on the perturbations suggested by:
        Schlegel et al. in 2019 in their paper: Towards a Rigorous Evaluation of XAI Methods on...
    4 baseline mode are supported:
        float values: set the baseline to a fixed values
        "zero": set the baseline to zero
        "inverse": set the baseline to the maximum for each feature minus the input value
        "negative": set the baseline by taking the inverse of the input values

    Parameters
    ----------
    model
        Model used for computing metric.
    inputs
        Input samples under study. (n*t*d)
    targets
        One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.
    metric
        The metric used to evaluate the model performance. One of the model metric keys when calling
        the evaluate function (e.g 'loss', 'accuracy'...). Default to loss.
    batch_size
        Number of samples to explain at once, if None compute all at once.
    causal_mode
        If 'insertion', the path is baseline to original time series,
        for 'deletion' the path is original time series to baseline.
    baseline_mode
        Value of the baseline state, associated perturbation for strings.
    steps
        Number of steps between the start and the end state.
        Can be set to -1 for all possible steps to be computed.
    max_percentage_perturbed
        Maximum percentage of the input perturbed.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self,
                 model: tf.keras.Model,
                 inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
                 metric: str = "loss",
                 batch_size: Optional[int] = 64,
                 causal_mode: str = "deletion",
                 baseline_mode: Union[float, str] = 0.0,
                 steps: int = 10,
                 max_percentage_perturbed: float = 1.0,
                 ):  # pylint: disable=R0913
        super().__init__(model, inputs, targets, batch_size)
        self.baseline_mode = baseline_mode
        self.causal_mode = causal_mode
        assert metric == "loss" or metric in self.model.metrics_names
        self.metric = metric

        self.nb_samples = inputs.shape[0]
        self.nb_features = np.prod(inputs.shape[1:])
        self.inputs_flatten = inputs.reshape(
            (self.nb_samples, self.nb_features, 1)
        )

        assert 0 < max_percentage_perturbed <= 1, \
            "max_percentage_perturbed should be between 0 and 1"
        max_nb_perturbation = floor(max_percentage_perturbed*self.nb_features)

        if steps == -1:
            steps = max_nb_perturbation
        assert 1 < steps <= max_nb_perturbation, \
            f"The number of steps must be between 2 and {max_nb_perturbation} (or '-1')."
        self.steps = int(steps)
        self.max_nb_perturbation = max_nb_perturbation
        self.model_baseline = self.model.evaluate(self.inputs, self.targets,
                                                  verbose=0, return_dict=True)[self.metric]

    def evaluate(self,
                 explanations: Union[tf.Tensor, np.ndarray]) -> float:
        """
        Evaluate the causal score for time series explanations.

        Parameters
        ----------
        explanations
            Explanation for the inputs, labels to evaluate.
        Returns
        -------
        causal_score
            Metric score, area over the deletion (lower is better) or insertion (higher is
            better) curve.
            The area represent the mean score perturbation compared to model_baseline
                (model score on non-perturbed inputs)
        """
        explanations = np.array(explanations)
        assert explanations.shape == self.inputs.shape, "The number of explanations must be the " \
                                                        "same as the number of inputs"

        explanations_flatten = explanations.reshape((len(explanations), -1))

        # for each sample, sort by most important features according to the explanation
        most_important_features = np.argsort(explanations_flatten, axis=-1)[:, ::-1]

        if isinstance(self.baseline_mode, float):
            baselines = np.full(self.inputs.shape, self.baseline_mode, dtype=np.float32)
        elif self.baseline_mode == "zero":
            baselines = np.zeros(self.inputs.shape)
        elif self.baseline_mode == "inverse":
            time_ax = 1
            maximums = self.inputs.max(axis=time_ax)
            maximums = np.expand_dims(maximums, axis=time_ax)
            maximums = np.repeat(maximums, self.inputs.shape[time_ax], axis=time_ax)
            baselines = maximums - self.inputs
        elif self.baseline_mode == "negative":
            baselines = -self.inputs
        else:
            raise NotImplementedError(f'Unknown perturbation type `{self.baseline_mode}`.')

        baselines_flatten = baselines.reshape(self.inputs_flatten.shape)

        steps = np.linspace(1, self.max_nb_perturbation, self.steps, dtype=np.int32)

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
            perturbed_inputs = start.copy()

            for i, ids in enumerate(ids_to_flip):
                perturbed_inputs[i, ids] = end[i, ids]

            perturbed_inputs = perturbed_inputs.reshape((-1, *self.inputs.shape[1:]))

            score = self.model.evaluate(perturbed_inputs, self.targets,
                                        self.batch_size, verbose=0,
                                        return_dict=True)
            scores.append(score[self.metric])

        auc = abs(np.mean(scores) - self.model_baseline)

        return auc


class DeletionTS(CausalFidelityTS):
    """
    Adaptation of the deletion metric for time series.

    The deletion metric measures the drop in the probability of a class
    as the input is gradually perturbed. The feature - time-steps pairs
    are perturbed in order of importance given by the explanation.
    A sharp drop, and thus a small area under the probability curve,
    are indicative of a good explanation.

    Parameters
    ----------
    model
        Model used for computing metric.
    inputs
        Input samples under study.
    targets
        One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.
    metric
        The metric used to evaluate the model performance. One of the model metric keys when calling
        the evaluate function (e.g 'loss', 'accuracy'...). Default to loss.
    batch_size
        Number of samples to explain at once, if None compute all at once.
    baseline_mode
        Value of the baseline state, associated perturbation for strings.
    steps
        Number of steps between the start and the end state.
        Can be set to -1 for all possible steps to be computed.
    max_nb_perturbation
        Maximum percentage of the input perturbed.
    """

    def __init__(self,
                 model: tf.keras.Model,
                 inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
                 metric: str = "loss",
                 batch_size: Optional[int] = 64,
                 baseline_mode: Union[float, str] = 0.0,
                 steps: int = 10,
                 max_percentage_perturbed: float = 1.0,
                 ):  # pylint: disable=R0913
        super().__init__(model, inputs, targets, metric, batch_size,
                         "deletion", baseline_mode, steps, max_percentage_perturbed)


class InsertionTS(CausalFidelityTS):
    """
    Adaptation of the insertion metric for time series.

    The insertion metric, on the other hand, captures the importance of
    the feature - time-steps pairs in terms of their ability to synthesize
    a time series and is measured by the rise in the probability of the
    class of interest as feature - time-steps pairs are added according to
    the generated importance map.

    Parameters
    ----------
    model
        Model used for computing metric.
    inputs
        Input samples under study.
    targets
        One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.
    metric
        The metric used to evaluate the model performance. One of the model metric keys when calling
        the evaluate function (e.g 'loss', 'accuracy'...). Default to loss.
    batch_size
        Number of samples to explain at once, if None compute all at once.
    baseline_mode
        Value of the baseline state, associated perturbation for strings.
    steps
        Number of steps between the start and the end state.
        Can be set to -1 for all possible steps to be computed.
    max_nb_perturbation
        Maximum percentage of the input perturbed.
    """

    def __init__(self,
                 model: tf.keras.Model,
                 inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
                 metric: str = "loss",
                 batch_size: Optional[int] = 64,
                 baseline_mode: Union[float, str] = 0.0,
                 steps: int = 10,
                 max_percentage_perturbed: float = 1.0,
                 ):  # pylint: disable=R0913
        super().__init__(model, inputs, targets, metric, batch_size,
                         "insertion", baseline_mode, steps, max_percentage_perturbed)
