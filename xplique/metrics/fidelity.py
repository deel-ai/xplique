"""
Fidelity (or Faithfulness) metrics
"""

from inspect import isfunction

import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr

from .base import ExplanationMetric
from ..commons import batch_predictions_one_hot
from ..types import Union, Callable, Optional, Dict


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
        subset_masks = np.random.rand(self.nb_samples, self.grid_size ** 2)
        subset_masks = subset_masks.argsort(axis=-1) > self.subset_size

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
                                            f"same as the number of inputs: {len(explanations)}"\
                                            f" vs {len(self.inputs)}"

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
        Can be set to -1 for all possible steps to be computed.
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
                 ):
        # pylint: disable=R0913
        super().__init__(model, inputs, targets, batch_size)
        self.causal_mode = causal_mode
        self.baseline_mode = baseline_mode

        self.nb_features = np.prod(inputs.shape[1:-1])
        self.inputs_flatten = inputs.reshape((len(inputs), self.nb_features, inputs.shape[-1]))

        assert 0.0 < max_percentage_perturbed <= 1.0, "`max_percentage_perturbed` must be" \
                                                      "in ]0, 1]."
        self.max_nb_perturbed = tf.math.floor(self.nb_features * max_percentage_perturbed)

        if steps == -1:
            steps = self.max_nb_perturbed
        self.steps = steps

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
        scores_dict = self.detailed_evaluate(explanations)

        # compute auc using trapezoidal rule (the steps are equally distributed)
        np_scores = np.array(list(scores_dict.values()))
        auc = np.mean(np_scores[:-1] + np_scores[1:]) * 0.5

        return auc

    def detailed_evaluate(self,
                          explanations: Union[tf.Tensor, np.ndarray]) -> Dict[int, float]:
        """
        Evaluate model performance for successive perturbations of an input.
        Used to compute causal score.

        The successive perturbations in the Insertion and Deletion metrics create a list of scores.
        This list of scores make a score evolution curve.
        The AUC of such curve is used as an explanation metric.
        However, the curve in itself is rich in information,
        its visualization and interpretation can bring further comprehension
        on the explanation and the model.
        Therefore this method was added so that it is possible to construct such curves.

        Parameters
        ----------
        explanations
            Explanation for the inputs, labels to evaluate.

        Returns
        -------
        causal_score_dict
            Dictionary of scores obtain for different perturbations
            Keys are the steps, i.e the number of features perturbed
            Values are the scores, the score of the model
                on the inputs with the corresponding number of features perturbed
        """
        explanations = np.array(explanations)
        assert len(explanations) == len(self.inputs), "The number of explanations must be the " \
                                            f"same as the number of inputs: {len(explanations)}"\
                                            f"vs {len(self.inputs)}"

        # the reference does not specify how to manage the channels of the explanations
        if len(explanations.shape) == 4:
            explanations = np.mean(explanations, -1)

        explanations_flatten = explanations.reshape((len(explanations), -1))

        # for each sample, sort by most important features according to the explanation
        most_important_features = np.argsort(explanations_flatten, axis=-1)[:, ::-1]

        baselines = self.baseline_mode(self.inputs) if isfunction(self.baseline_mode) else \
            np.ones_like(self.inputs, dtype=np.float32) * self.baseline_mode
        baselines_flatten = baselines.reshape(self.inputs_flatten.shape)

        steps = np.linspace(0, self.max_nb_perturbed, self.steps+1, dtype=np.int32)

        if self.causal_mode == "deletion":
            start = self.inputs_flatten
            end = baselines_flatten
        elif self.causal_mode == "insertion":
            start = baselines_flatten
            end = self.inputs_flatten
        else:
            raise NotImplementedError(f'Unknown causal mode `{self.causal_mode}`.')

        scores_dict = {}
        for step in steps:
            ids_to_flip = most_important_features[:, :step]
            batch_inputs = start.copy()

            for i, ids in enumerate(ids_to_flip):
                batch_inputs[i, ids] = end[i, ids]

            batch_inputs = batch_inputs.reshape((-1, *self.inputs.shape[1:]))

            predictions = batch_predictions_one_hot(self.model, batch_inputs,
                                                    self.targets, self.batch_size)

            scores_dict[step] = np.mean(predictions)

        return scores_dict


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
        Can be set to -1 for all possible steps to be computed.
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
        super().__init__(model, inputs, targets, batch_size, "deletion",
                         baseline_mode, steps, max_percentage_perturbed)


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
        Can be set to -1 for all possible steps to be computed.
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
        super().__init__(model, inputs, targets, batch_size, "insertion",
                         baseline_mode, steps, max_percentage_perturbed)


class CausalFidelityTS(ExplanationMetric):
    """
    Used to compute the insertion and deletion metrics for Time Series explanations.

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
                 baseline_mode: Union[float, Callable] = 0.0,
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
        self.max_nb_perturbed = int(self.nb_features * max_percentage_perturbed)

        if steps == -1:
            steps = self.max_nb_perturbed
        self.steps = steps

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
            Metric score (for interpretation, see score interpretation in the documentation).
        """
        scores_dict = self.detailed_evaluate(explanations)

        # compute auc with trapeze
        np_scores = np.array(list(scores_dict.values()))
        auc = np.mean(np_scores[:-1] + np_scores[1:]) * 0.5

        return auc

    def detailed_evaluate(self,
                          explanations: Union[tf.Tensor, np.ndarray]) -> Dict[int, float]:
        """
        Evaluate model performance for successive perturbations of an input.
        Used to compute causal score for time series explanations.

        The successive perturbations in the Insertion and Deletion metrics create a list of scores.
        This list of scores make a score evolution curve.
        The AUC of such curve is used as an explanation metric.
        However, the curve in itself is rich in information,
        its visualization and interpretation can bring further comprehension
        on the explanation and the model.
        Therefore this method was added so that it is possible to construct such curves.

        Parameters
        ----------
        explanations
            Explanation for the inputs, labels to evaluate.

        Returns
        -------
        causal_score_dict
            Dictionary of scores obtain for different perturbations
            Keys are the steps, i.e the number of features perturbed
            Values are the scores, the score of the model
                on the inputs with the corresponding number of features perturbed
        """
        explanations = np.array(explanations)
        assert explanations.shape == self.inputs.shape, "The number of explanations must be the " \
                                                        "same as the number of inputs"

        explanations_flatten = explanations.reshape((len(explanations), -1))

        # for each sample, sort by most important features according to the explanation
        most_important_features = np.argsort(explanations_flatten, axis=-1)[:, ::-1]

        baselines = self.baseline_mode(self.inputs) if isfunction(self.baseline_mode) else \
            np.full(self.inputs.shape, self.baseline_mode, dtype=np.float32)
        baselines_flatten = baselines.reshape(self.inputs_flatten.shape)

        steps = np.linspace(0, self.max_nb_perturbed, self.steps+1, dtype=np.int32)

        if self.causal_mode == "deletion":
            start = self.inputs_flatten
            end = baselines_flatten
        elif self.causal_mode == "insertion":
            start = baselines_flatten
            end = self.inputs_flatten
        else:
            raise NotImplementedError(f'Unknown causal mode `{self.causal_mode}`.')

        scores_dict = {}
        for step in steps:
            ids_to_flip = most_important_features[:, :step]
            perturbed_inputs = start.copy()

            for i, ids in enumerate(ids_to_flip):
                perturbed_inputs[i, ids] = end[i, ids]

            perturbed_inputs = perturbed_inputs.reshape((-1, *self.inputs.shape[1:]))

            score = self.model.evaluate(perturbed_inputs, self.targets,
                                        self.batch_size, verbose=0,
                                        return_dict=True)
            scores_dict[step] = score[self.metric]

        return scores_dict


class DeletionTS(CausalFidelityTS):
    """
    Adaptation of the deletion metric for time series.

    The deletion metric measures the drop in the probability of a class as the input is
    gradually perturbed. The feature - time-steps pairs are perturbed in order of importance
    given by the explanation. A sharp drop, and thus a small area under the probability curve,
    are indicative of a good explanation.

    Ref. Schlegel et al., Towards a Rigorous Evaluation of XAI Methods (2019).

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
        Value of the baseline state or the associated perturbation functions.
        A float value will fix the baseline to that value, "zero" set the baseline to zero,
        "inverse" set the baseline to the maximum for each feature minus the input value and
        "negative" set the baseline by taking the inverse of input values.
    steps
        Number of steps between the start and the end state.
        Can be set to -1 for all possible steps to be computed.
    max_percentage_perturbed
        Maximum percentage of the input perturbed.
    """

    def __init__(self,
                 model: tf.keras.Model,
                 inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
                 metric: str = "loss",
                 batch_size: Optional[int] = 64,
                 baseline_mode: Union[float, Callable] = 0.0,
                 steps: int = 10,
                 max_percentage_perturbed: float = 1.0,
                 ):  # pylint: disable=R0913
        super().__init__(model, inputs, targets, metric, batch_size,
                         "deletion", baseline_mode, steps, max_percentage_perturbed)


class InsertionTS(CausalFidelityTS):
    """
    Adaptation of the insertion metric for time series.

    The insertion metric, captures the importance of the feature - time-steps pairs in terms of
    their ability to synthesize a time series.

    Ref. Schlegel et al., Towards a Rigorous Evaluation of XAI Methods (2019).

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
        Value of the baseline state or the associated perturbation functions.
        A float value will fix the baseline to that value, "zero" set the baseline to zero,
        "inverse" set the baseline to the maximum for each feature minus the input value and
        "negative" set the baseline by taking the inverse of input values.
    steps
        Number of steps between the start and the end state.
        Can be set to -1 for all possible steps to be computed.
    max_percentage_perturbed
        Maximum percentage of the input perturbed.
    """

    def __init__(self,
                 model: tf.keras.Model,
                 inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
                 metric: str = "loss",
                 batch_size: Optional[int] = 64,
                 baseline_mode: Union[float, Callable] = 0.0,
                 steps: int = 10,
                 max_percentage_perturbed: float = 1.0,
                 ):  # pylint: disable=R0913
        super().__init__(model, inputs, targets, metric, batch_size,
                         "insertion", baseline_mode, steps, max_percentage_perturbed)


class DeletionTab(CausalFidelityTS):
    """
    Adaptation of the deletion metric for tabular data.

    Ref. Petsiuk & al., RISE: Randomized Input Sampling for Explanation of Black-box Models (2018).
    https://arxiv.org/pdf/1806.07421.pdf
    Ref. Schlegel et al., Towards a Rigorous Evaluation of XAI Methods (2019).

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
        Value of the baseline state, will be called with the inputs if it is a function.
    steps
        Number of steps between the start and the end state.
        Can be set to -1 for all possible steps to be computed.
    max_percentage_perturbed
        Maximum percentage of the input perturbed.
    """ # pylint: disable=R0913

    def __init__(self,
                 model: tf.keras.Model,
                 inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
                 metric: str = "loss",
                 batch_size: Optional[int] = 64,
                 baseline_mode: Union[float, Callable] = 0.0,
                 steps: int = 10,
                 max_percentage_perturbed: float = 1.0,
                 ):
        super().__init__(model, inputs, targets, metric, batch_size,
                         "deletion", baseline_mode, steps, max_percentage_perturbed)


class InsertionTab(CausalFidelityTS):
    """
    Adaptation of the insertion metric for tabular data.

    Ref. Petsiuk & al., RISE: Randomized Input Sampling for Explanation of Black-box Models (2018).
    https://arxiv.org/pdf/1806.07421.pdf
    Ref. Schlegel et al., Towards a Rigorous Evaluation of XAI Methods (2019).

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
        Value of the baseline state, will be called with the inputs if it is a function.
    steps
        Number of steps between the start and the end state.
        Can be set to -1 for all possible steps to be computed.
    max_percentage_perturbed
        Maximum percentage of the input perturbed.
    """ # pylint: disable=R0913

    def __init__(self,
                 model: tf.keras.Model,
                 inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
                 metric: str = "loss",
                 batch_size: Optional[int] = 64,
                 baseline_mode: Union[float, Callable] = 0.0,
                 steps: int = 10,
                 max_percentage_perturbed: float = 1.0,
                 ):
        super().__init__(model, inputs, targets, metric, batch_size,
                         "insertion", baseline_mode, steps, max_percentage_perturbed)
