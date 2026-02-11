"""
Fidelity (or Faithfulness) metrics
"""
from abc import ABC, abstractmethod
from inspect import isfunction

import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr

from .base import ExplanationMetric
from ..types import Union, Callable, Optional, Dict
from ..commons import batch_tensor


_EPS = 1e-8


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


class BaseAverageXMetric(ExplanationMetric, ABC):
    """
    Base class for Average Drop / Increase / Gain metrics.

    Parameters
    ----------
    model
        Model used for computing metric.
    inputs
        Input samples under study.
    targets
        One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.
    batch_size
        Number of samples to process at once, if None compute all at once.
    operator
        Function g to explain. It should take 3 parameters (f, x, y) and return a scalar,
        with f the model, x the inputs and y the targets. If None, use the standard
        operator g(f, x, y) = f(x)[y].
    activation
        A string that belongs to [None, 'sigmoid', 'softmax']. Specify if we should add
        an activation layer once the model has been called.

    Notes
    -----
    Subclasses implement:
    - `evaluate(explanations) -> float`
    - `detailed_evaluate(inputs, targets, explanations) -> np.ndarray`
    """
    def __init__(self,
                 model: tf.keras.Model,
                 inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
                 batch_size: Optional[int] = 64,
                 operator: Optional[Union[str, Callable]] = None,
                 activation: Optional[str] = None):
        super().__init__(model, inputs, targets, batch_size, operator, activation)

    @staticmethod
    def _perturb_with_mask(inputs: tf.Tensor, explanations: tf.Tensor) -> tf.Tensor:
        """
        Build a normalized mask from explanations and apply
        element-wise multiplication to inputs.

        This method processes explanations into a [0, 1] mask by:
        1. Taking the absolute value of explanation scores
        2. Averaging across channels if explanations are multi-channel (4D)
        3. Applying per-sample min-max normalization to [0, 1]
        4. Broadcasting the mask to match input dimensions
        5. Performing element-wise multiplication: x ⊙ mask

        The resulting perturbed inputs retain features with high attribution scores
        while suppressing features with low scores.

        Parameters
        ----------
        inputs : tf.Tensor
            Input samples to perturb. Shape can be:
            - (B, H, W, C) for images
            - (B, T, F) for time series
            - (B, ...) for other data types
        explanations : tf.Tensor
            Attribution scores for each input. Shape can be:
            - (B, H, W, C) or (B, H, W) for images
            - (B, T, F) or (B, T) for time series
            Must have same batch size B as inputs.

        Returns
        -------
        perturbed_inputs : tf.Tensor
            Element-wise product of inputs and normalized mask, same shape as inputs.
            Values range from 0 (fully masked) to original input value (unmasked).

        Notes
        -----
        - Multi-channel explanations are averaged to a single channel before normalization
        - Min-max normalization is applied independently per sample to ensure each mask
          spans the full [0, 1] range
        - The mask is broadcast to match input shape, preserving spatial/temporal structure
        """
        # Apply average if multi-channel explanations and get absolute value
        inp = tf.convert_to_tensor(inputs, dtype=tf.float32)
        exp = tf.convert_to_tensor(explanations, dtype=tf.float32)

        # Absolute value and channel averaging
        mask = tf.math.abs(exp)
        if mask.shape.rank == 4:
            mask = tf.reduce_mean(mask, axis=-1)  # (B,H,W)

        # Apply min-max normalization per sample
        axes = tf.range(1, tf.rank(mask))
        mask_min = tf.reduce_min(mask, axis=axes, keepdims=True)
        mask_max = tf.reduce_max(mask, axis=axes, keepdims=True)
        mask = (mask - mask_min) / (mask_max - mask_min + _EPS)

        # Broadcast mask to inputs
        if inp.shape.rank == 4 and mask.shape.rank == 3:
            mask = mask[..., tf.newaxis]  # (B,H,W,1)
        if inp.shape.rank == 3 and mask.shape.rank == 2:
            mask = mask[..., tf.newaxis]  # (B,T,1)

        return tf.multiply(inp, mask)

    def evaluate(self, explanations: Union[tf.Tensor, np.ndarray]) -> float:
        """
        Evaluate the metric over the entire dataset by iterating over batches.
        """
        explanations = tf.convert_to_tensor(explanations, dtype=tf.float32)
        assert len(explanations) == len(self.inputs), \
            "The number of explanations must match the number of inputs."

        scores = []
        for inp, tgt, phi in batch_tensor((self.inputs, self.targets, explanations),
                                          self.batch_size or len(self.inputs)):
            batch_scores = self.detailed_evaluate(inp, tgt, phi)
            scores.append(batch_scores)

        return float(np.mean(np.concatenate(scores, axis=0)))

    @abstractmethod
    def detailed_evaluate(
            self,
            inputs: tf.Tensor,
            targets: tf.Tensor,
            explanations: tf.Tensor
    ) -> np.ndarray:
        """
        Evaluate the metric for a single batch of inputs, targets, and explanations.

        Parameters
        ----------
        inputs
            Batch of input samples.
        targets
            Batch of target labels.
        explanations
            Batch of explanations.

        Returns
        -------
        scores
            A numpy array of shape (B,) with score per sample.
        """
        raise NotImplementedError()


class AverageDropMetric(BaseAverageXMetric):
    """
    Average Drop (AD) — measures relative decrease in the model score when the
    input is masked by the explanation (lower AD is better).

    For each sample i:
        base_i  = g(f, x_i, y_i)              # scalar via operator/inference fn
        after_i = g(f, x_i ⊙ M_i, y_i)        # M_i from explanation
        AD_i    = ReLU(base_i - after_i) / (base_i + eps)

    This implementation:
    - Uses `self.batch_inference_function` to compute the scalar scores with the
      optional `activation` applied (softmax/sigmoid) if requested at init.
    - Builds M_i by |explanation|, channel-average (if 4D), per-sample min-max to [0,1],
      then broadcast to input shape, and multiplicatively masks inputs.

    References
    ----------
    Chattopadhay, A., Sarkar, A., Howlader, P., & Balasubramanian, V. N. (2018, March).
    Grad-cam++: Generalized gradient-based visual explanations for deep convolutional networks.
    In 2018 IEEE winter conference on applications of computer vision (WACV) (pp. 839-847). IEEE.

    Parameters
    ----------
    model
        Model used for computing metric.
    inputs
        Input samples under study.
    targets
        One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.
    batch_size
        Number of samples to process at once, if None compute all at once.
    operator
        Function g to explain. It should take 3 parameters (f, x, y) and return a scalar,
        with f the model, x the inputs and y the targets. If None, use the standard
        operator g(f, x, y) = f(x)[y].
    activation
        A string that belongs to [None, 'sigmoid', 'softmax']. Specify if we should add
        an activation layer once the model has been called.

    Notes
    -----
    `evaluate(explanations)` returns the mean AD over the dataset.
    """

    def detailed_evaluate(
            self,
            inputs: tf.Tensor,
            targets: tf.Tensor,
            explanations: tf.Tensor
    ) -> np.ndarray:
        """
        Compute Average Drop scores for a batch of samples.

        This method evaluates how much the model's confidence decreases when inputs
        are masked according to their explanation-based importance. For each sample:

        1. Compute base score: f(x_i, y_i)
        2. Create normalized mask M_i from explanation
        3. Compute perturbed score: f(x_i ⊙ M_i, y_i)
        4. Calculate AD_i = max(0, base_i - perturbed_i) / (base_i + ε)

        A lower Average Drop indicates that important features (according to the
        explanation) are correctly identified, as masking them causes significant
        performance degradation.

        Parameters
        ----------
        inputs : tf.Tensor
            Batch of input samples. Shape: (B, H, W, C) for images,
            (B, T, F) for time series, or (B, ...) for other data types.
        targets : tf.Tensor
            Batch of target labels. Shape: (B, num_classes) for one-hot encoded,
            or (B,) for class indices/regression targets.
        explanations : tf.Tensor
            Batch of attribution maps. Shape must be compatible with inputs
            (same spatial/temporal dimensions, optionally without channel dimension).

        Returns
        -------
        scores : np.ndarray
            Per-sample Average Drop scores, shape (B,).
            Values range from 0 (no drop or increase) to ~1 (complete drop).
            Lower values indicate better explanations.

        Notes
        -----
        - Uses ReLU to ignore cases where masking increases confidence
        - Normalization by base score makes the metric scale-invariant
        - The mask is constructed via `_perturb_with_mask`, which applies
          absolute value, channel averaging, and min-max normalization
        """
        base = self.batch_inference_function(self.model, inputs, targets, self.batch_size)
        perturbed = self._perturb_with_mask(inputs, explanations)
        after = self.batch_inference_function(self.model, perturbed, targets, self.batch_size)

        ad = tf.nn.relu(base - after) / (base + _EPS)  # per-sample
        return ad.numpy()


class AverageIncreaseMetric(BaseAverageXMetric):
    """
    Average Increase in Confidence (AIC) — fraction of samples for which the
    masked input yields a *higher* score (higher is better).

    For each sample i:
        base_i  = g(f, x_i, y_i)
        after_i = g(f, x_i ⊙ M_i, y_i)
        AIC_i   = 1[after_i > base_i]

    The dataset-level AIC is the mean of {AIC_i}.

    Notes
    -----
    - Works best with probabilistic outputs; set `activation="softmax"` or `"sigmoid"`
      if your model returns logits.
    - `evaluate(explanations)` returns the mean indicator over the dataset.

    References
    ----------
    Chattopadhay, A., Sarkar, A., Howlader, P., & Balasubramanian, V. N. (2018, March).
    Grad-cam++: Generalized gradient-based visual explanations for deep convolutional networks.
    In 2018 IEEE winter conference on applications of computer vision (WACV) (pp. 839-847). IEEE.

    Parameters
    ----------
    model
        Model used for computing metric.
    inputs
        Input samples under study.
    targets
        One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.
    batch_size
        Number of samples to process at once, if None compute all at once.
    operator
        Function g to explain. It should take 3 parameters (f, x, y) and return a scalar,
        with f the model, x the inputs and y the targets. If None, use the standard
        operator g(f, x, y) = f(x)[y].
    activation
        A string that belongs to [None, 'sigmoid', 'softmax']. Specify if we should add
        an activation layer once the model has been called.

    """

    def detailed_evaluate(
            self,
            inputs: tf.Tensor,
            targets: tf.Tensor,
            explanations: tf.Tensor
    ) -> np.ndarray:
        """
        Compute Average Increase indicators for a batch of samples.

        This method evaluates whether masking inputs according to their explanation-based
        importance *increases* the model's confidence. For each sample:

        1. Compute base score: f(x_i, y_i)
        2. Create normalized mask M_i from explanation
        3. Compute perturbed score: f(x_i ⊙ M_i, y_i)
        4. Calculate AIC_i = 1 if after_i > base_i, else 0

        A higher Average Increase indicates that the explanation highlights features
        which, when isolated, are sufficient or even more predictive than the full input.
        This can reveal whether explanations capture truly discriminative features.

        Parameters
        ----------
        inputs : tf.Tensor
            Batch of input samples. Shape: (B, H, W, C) for images,
            (B, T, F) for time series, or (B, ...) for other data types.
        targets : tf.Tensor
            Batch of target labels. Shape: (B, num_classes) for one-hot encoded,
            or (B,) for class indices/regression targets.
        explanations : tf.Tensor
            Batch of attribution maps. Shape must be compatible with inputs
            (same spatial/temporal dimensions, optionally without channel dimension).

        Returns
        -------
        scores : np.ndarray
            Per-sample binary indicators, shape (B,).
            Values are 0 (no increase) or 1 (confidence increased).
            Higher mean values indicate better explanations.

        Notes
        -----
        - Returns binary indicators; dataset-level metric is the mean (i.e., proportion)
        - Best used with probabilistic outputs; consider `activation="softmax"` or
          `"sigmoid"` if the model returns logits
        - The mask is constructed via `_perturb_with_mask`, which applies
          absolute value, channel averaging, and min-max normalization
        """
        base = self.batch_inference_function(self.model, inputs, targets, self.batch_size)
        perturbed = self._perturb_with_mask(inputs, explanations)
        after = self.batch_inference_function(self.model, perturbed, targets, self.batch_size)

        inc = tf.cast(after > base, tf.float32)  # per-sample 0/1
        return inc.numpy()


class AverageGainMetric(BaseAverageXMetric):
    """
    Average Gain (AG) — normalized relative increase when the input is masked
    by the explanation (higher AG is better, complementary to AD).

    For each sample i:
        base_i  = g(f, x_i, y_i)
        after_i = g(f, x_i ⊙ M_i, y_i)
        AG_i    = ReLU(after_i - base_i) / (1 - base_i + eps)

    Notes
    -----
    - Intended for scores in [0,1]. If your model outputs logits, use
      `activation="softmax"` / `"sigmoid"` at construction to operate on probabilities.
    - `evaluate(explanations)` returns the mean AG over the dataset.

    References
    ----------
    Zhang, H., Torres, F., Sicre, R., Avrithis, Y., & Ayache, S. (2024).
    Opti-CAM: Optimizing saliency maps for interpretability.
    Computer Vision and Image Understanding, 248, 104101.

    Parameters
    ----------
    model
        Model used for computing metric.
    inputs
        Input samples under study.
    targets
        One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.
    batch_size
        Number of samples to process at once, if None compute all at once.
    operator
        Function g to explain. It should take 3 parameters (f, x, y) and return a scalar,
        with f the model, x the inputs and y the targets. If None, use the standard
        operator g(f, x, y) = f(x)[y].
    activation
        A string that belongs to [None, 'sigmoid', 'softmax']. Specify if we should add
        an activation layer once the model has been called.

    """

    def detailed_evaluate(
            self,
            inputs: tf.Tensor,
            targets: tf.Tensor,
            explanations: tf.Tensor
    ) -> np.ndarray:
        """
        Compute Average Gain scores for a batch of samples.

        This method evaluates the relative increase in model confidence when inputs
        are masked according to their explanation-based importance. For each sample:

        1. Compute base score: f(x_i, y_i)
        2. Create normalized mask M_i from explanation
        3. Compute perturbed score: f(x_i ⊙ M_i, y_i)
        4. Calculate AG_i = max(0, after_i - base_i) / (1 - base_i + ε)

        A higher Average Gain indicates that the explanation successfully identifies
        features which, when isolated, are sufficient to maintain or increase the
        model's confidence. This is complementary to Average Drop and measures the
        explanation's ability to capture discriminative features.

        Parameters
        ----------
        inputs : tf.Tensor
            Batch of input samples. Shape: (B, H, W, C) for images,
            (B, T, F) for time series, or (B, ...) for other data types.
        targets : tf.Tensor
            Batch of target labels. Shape: (B, num_classes) for one-hot encoded,
            or (B,) for class indices/regression targets.
        explanations : tf.Tensor
            Batch of attribution maps. Shape must be compatible with inputs
            (same spatial/temporal dimensions, optionally without channel dimension).

        Returns
        -------
        scores : np.ndarray
            Per-sample Average Gain scores, shape (B,).
            Values range from 0 (no gain or decrease) to potentially > 1.
            Higher values indicate better explanations.

        Notes
        -----
        - Uses ReLU to ignore cases where masking decreases confidence
        - Normalization by (1 - base_i) accounts for the remaining headroom to score=1
        - Designed for probabilistic outputs in [0, 1]; use `activation="softmax"` or
          `"sigmoid"` if the model returns logits
        - The mask is constructed via `_perturb_with_mask`, which applies
          absolute value, channel averaging, and min-max normalization
        """
        base = self.batch_inference_function(self.model, inputs, targets, self.batch_size)
        perturbed = self._perturb_with_mask(inputs, explanations)
        after = self.batch_inference_function(self.model, perturbed, targets, self.batch_size)

        ag = tf.nn.relu(after - base) / (1.0 - base + _EPS)  # per-sample
        return ag.numpy()
