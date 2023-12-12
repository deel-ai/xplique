"""
Fidelity (or Faithfulness) metrics
"""

from inspect import isfunction

import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr

from .base import ExplanationMetric
from ..types import Union, Callable, Optional, Dict
from ..commons import batch_tensor


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
    operator
        Function g to explain, g take 3 parameters (f, x, y) and should return a scalar,
        with f the model, x the inputs and y the targets. If None, use the standard
        operator g(f, x, y) = f(x)[y].
    activation
        A string that belongs to [None, 'sigmoid', 'softmax']. Specify if we should add
        an activation layer once the model has been called. It is useful, for instance
        if you want to measure a 'drop of probability' by adding a sigmoid or softmax
        after getting your logits. If None does not add a layer to your model.
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self,
                 model: Callable,
                 inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
                 batch_size: Optional[int] = 64,
                 grid_size: Optional[int] = 9,
                 subset_percent: float = 0.2,
                 baseline_mode: Union[Callable, float] = 0.0,
                 nb_samples: int = 200,
                 operator: Optional[Callable] = None,
                 activation: Optional[str] = None):
        # pylint: disable=too-many-arguments
        super().__init__(model, inputs, targets, batch_size, operator, activation)
        self.grid_size = grid_size
        self.subset_percent = subset_percent
        self.baseline_mode = baseline_mode
        self.nb_samples = nb_samples

        # set batch_size for inputs and perturbations
        self.batch_size = self.batch_size or (len(self.inputs) * self.nb_samples)
        self.perturbation_batch_size = min(self.batch_size, self.nb_samples)
        self.inputs_batch_size = max(1, self.batch_size // self.perturbation_batch_size)

        # if unspecified use the original equation (pixel-wise modification)
        self.grid_size = grid_size or self.inputs.shape[1]

        self.base_predictions = self.batch_inference_function(self.model, self.inputs,
                                                              self.targets, self.batch_size)
        self.base_predictions = tf.expand_dims(self.base_predictions, axis=1)

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
        for inp, label, phi, base in batch_tensor((self.inputs, self.targets,
                                                   explanations, self.base_predictions),
                                                  self.inputs_batch_size):
            # reshape the explanations to align with future mask multiplications
            if len(inp.shape) > len(phi.shape):
                phi = tf.expand_dims(phi, axis=-1)
            phi = tf.expand_dims(phi, axis=1)

            total_perturbed_samples = 0
            preds, attrs = None, None
            # loop over perturbations (a single pass if batch_size > nb_samples, batched otherwise)
            while total_perturbed_samples < self.nb_samples:
                nb_perturbations = min(self.perturbation_batch_size,
                                       self.nb_samples - total_perturbed_samples)
                total_perturbed_samples += nb_perturbations

                degraded_inputs, subset_masks = self._perturb_samples(inp, nb_perturbations)
                repeated_label = tf.repeat(label, nb_perturbations, 0)

                # compute the predictions for a batch of perturbed inputs
                perturbed_predictions = self.batch_inference_function(
                    self.model, degraded_inputs, repeated_label, self.batch_size)

                # reshape the predictions to align with the inputs
                perturbed_predictions = tf.reshape(perturbed_predictions,
                           (inp.shape[0], nb_perturbations))


                # measure the two terms that should be correlated
                pred = base - perturbed_predictions
                preds = pred if preds is None else tf.concat([preds, pred], axis=1)

                attr = tf.reduce_sum(phi * (1.0 - subset_masks),
                                     axis=list(range(2, len(subset_masks.shape))))
                attrs = attr if attrs is None else tf.concat([attrs, attr], axis=1)

            # iterate over samples of the batch
            batch_correlations = []
            for pred, attr in zip(preds, attrs):
                # compute correlation
                corr_score = spearmanr(pred, attr)[0]

                # sanity check: if the model predictions are the same, no variation
                if np.isnan(corr_score):
                    corr_score = 0.0
                batch_correlations.append(corr_score)
            correlations += batch_correlations

        fidelity_score = np.mean(correlations)

        return float(fidelity_score)

    @tf.function
    def _perturb_samples(self,
                         inputs: tf.Tensor,
                         nb_perturbations: int) -> tf.Tensor:
        """
        Duplicate the samples and apply a noisy mask to each of them.

        Parameters
        ----------
        inputs
            Input samples to be explained. (n, ...)
        nb_perturbations
            Number of perturbations to apply for each input.

        Returns
        -------
        perturbed_inputs
            Duplicated inputs perturbed with random noise. (n * nb_perturbations, ...)
        """
        # (n, nb_perturbations, ...)
        perturbed_inputs = tf.repeat(inputs[:, tf.newaxis], repeats=nb_perturbations, axis=1)

        # prepare the random masks that will designate the modified subset (S in original equation)
        # we ensure the masks have exactly `subset_size` pixels set to baseline
        # and interpolate them if needed
        # the masks format depend on the data type
        if len(inputs.shape) == 2:  # tabular data, grid size is ignored
            # prepare the random masks
            subset_masks = tf.random.uniform((nb_perturbations, inputs.shape[1]), 0, 1, tf.float32)
            subset_masks = subset_masks > self.subset_percent
            subset_masks = tf.cast(subset_masks, tf.float32)

        elif len(inputs.shape) == 3:  # time series
            # prepare the random masks
            subset_masks = tf.random.uniform((nb_perturbations, self.grid_size * inputs.shape[2]),
                                             minval=0, maxval=1, dtype=tf.float32)
            subset_masks = subset_masks > self.subset_percent

            # and interpolate them if needed
            subset_masks = tf.reshape(tf.cast(subset_masks, tf.float32),
                                      (nb_perturbations, self.grid_size, inputs.shape[2], 1))
            subset_masks = tf.image.resize(subset_masks, self.inputs.shape[1:], method="nearest")
            subset_masks = tf.squeeze(subset_masks, axis=-1)

        elif len(inputs.shape) == 4:  # image data
            # prepare the random masks
            subset_masks = tf.random.uniform(shape=(nb_perturbations, self.grid_size ** 2),
                                             minval=0, maxval=1, dtype=tf.float32)
            subset_masks = subset_masks > self.subset_percent

            # and interpolate them if needed
            subset_masks = tf.reshape(tf.cast(subset_masks, tf.float32),
                                      (nb_perturbations, self.grid_size, self.grid_size, 1))
            subset_masks = tf.image.resize(subset_masks, self.inputs.shape[1:-1], method="nearest")

        # (n, nb_perturbations, ...)
        subset_masks = tf.repeat(subset_masks[tf.newaxis],
                                 repeats=perturbed_inputs.shape[0], axis=0)

        baseline = self.baseline_mode(perturbed_inputs) if isfunction(self.baseline_mode) else \
                self.baseline_mode

        perturbed_inputs = perturbed_inputs * subset_masks + (1.0 - subset_masks) * baseline

        perturbed_inputs = tf.reshape(perturbed_inputs, (-1, *self.inputs.shape[1:]))

        return perturbed_inputs, subset_masks


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
    operator
        Function g to explain, g take 3 parameters (f, x, y) and should return a scalar,
        with f the model, x the inputs and y the targets. If None, use the standard
        operator g(f, x, y) = f(x)[y].
    activation
        A string that belongs to [None, 'sigmoid', 'softmax']. Specify if we should add
        an activation layer once the model has been called. It is useful, for instance
        if you want to measure a 'drop of probability' by adding a sigmoid or softmax
        after getting your logits. If None does not add a layer to your model.
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
                 operator: Optional[Callable] = None,
                 activation: Optional[str] = None
                 ):
        # pylint: disable=R0913
        super().__init__(model, inputs, targets, batch_size, operator, activation)
        self.causal_mode = causal_mode
        self.baseline_mode = baseline_mode

        # If the input has channels (colored image), they are all occluded at the same time
        self.has_channels = len(self.inputs.shape) > 3

        if self.has_channels:
            self.nb_features = np.prod(self.inputs.shape[1:-1])
            self.inputs_flatten = self.inputs.reshape(
                (len(self.inputs), self.nb_features, self.inputs.shape[-1]))
        else:
            self.nb_features = np.prod(self.inputs.shape[1:])
            self.inputs_flatten = self.inputs.reshape((len(self.inputs), self.nb_features, 1))

        assert 0.0 < max_percentage_perturbed <= 1.0, \
            "`max_percentage_perturbed` must be in ]0, 1]."
        self.max_nb_perturbed = int(np.floor(self.nb_features * max_percentage_perturbed))

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

        steps = np.linspace(0, self.max_nb_perturbed, self.steps + 1, dtype=np.int32)

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

            predictions = self.batch_inference_function(self.model, batch_inputs,
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
    operator
        Function g to explain, g take 3 parameters (f, x, y) and should return a scalar,
        with f the model, x the inputs and y the targets. If None, use the standard
        operator g(f, x, y) = f(x)[y].
    activation
        A string that belongs to [None, 'sigmoid', 'softmax']. Specify if we should add
        an activation layer once the model has been called. It is useful, for instance
        if you want to measure a 'drop of probability' by adding a sigmoid or softmax
        after getting your logits. If None does not add a layer to your model.
    """

    def __init__(self,
                 model: tf.keras.Model,
                 inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
                 batch_size: Optional[int] = 64,
                 baseline_mode: Union[float, Callable] = 0.0,
                 steps: int = 10,
                 max_percentage_perturbed: float = 1.0,
                 operator: Optional[Callable] = None,
                 activation: Optional[str] = None
                 ):
        super().__init__(model, inputs, targets, batch_size, "deletion",
                         baseline_mode, steps, max_percentage_perturbed,
                         operator, activation)


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
    operator
        Function g to explain, g take 3 parameters (f, x, y) and should return a scalar,
        with f the model, x the inputs and y the targets. If None, use the standard
        operator g(f, x, y) = f(x)[y].
    activation
        A string that belongs to [None, 'sigmoid', 'softmax']. Specify if we should add
        an activation layer once the model has been called. It is useful, for instance
        if you want to measure a 'drop of probability' by adding a sigmoid or softmax
        after getting your logits. If None does not add a layer to your model.
    """

    def __init__(self,
                 model: tf.keras.Model,
                 inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
                 batch_size: Optional[int] = 64,
                 baseline_mode: Union[float, Callable] = 0.0,
                 steps: int = 10,
                 max_percentage_perturbed: float = 1.0,
                 operator: Optional[Callable] = None,
                 activation: Optional[str] = None
                 ):
        super().__init__(model, inputs, targets, batch_size, "insertion",
                         baseline_mode, steps, max_percentage_perturbed,
                         operator, activation)
