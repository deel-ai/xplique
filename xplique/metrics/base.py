"""
Module related to abstract attribution metric
"""

from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np

from ..commons import Tasks, numpy_sanitize, get_inference_function, batch_tensor
from ..attributions.base import WhiteBoxExplainer, BlackBoxExplainer
from ..types import Callable, Optional, Union, OperatorSignature


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
    activation
        A string that belongs to [None, 'sigmoid', 'softmax']. Specify if we should add
        an activation layer once the model has been called. It is useful, for instance
        if you want to measure a 'drop of probability' by adding a sigmoid or softmax
        after getting your logits. If None does not add a layer to your model.
    """

    def __init__(self,
                 model: Callable,
                 inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
                 batch_size: Optional[int] = 64,
                 activation: Optional[str] = None):
        if activation is None:
            self.model = model
        else:
            assert activation in ['sigmoid', 'softmax'], \
            "activation must be in ['sigmoid', 'softmax']"
            if activation == 'sigmoid':
                self.model = lambda x: tf.nn.sigmoid(model(x))
            else:
                self.model = lambda x: tf.nn.softmax(model(x), axis=-1)
        self.inputs, self.targets = numpy_sanitize(inputs, targets)
        self.batch_size = batch_size


class BaseComplexityMetric(ABC):
    """
    Base interface for Complexity Metrics.
    These metrics only depend on the explanations themselves.
    Parameters
    ----------
    batch_size
        Number of samples to evaluate at once.
    """
    def __init__(self, batch_size: Optional[int] = 32):
        self.batch_size = batch_size

    @abstractmethod
    def detailed_evaluate(self, explanations: tf.Tensor) -> np.ndarray:
        """
        Per-batch evaluation of explanations (no reduction).

        Parameters
        ----------
        explanations
            Explanation for the inputs, labels to evaluate.

        Returns
        -------
        scores
            A numpy array of shape (B,) with score per sample.
        """
        raise NotImplementedError()

    def evaluate(self, explanations: Union[tf.Tensor, np.ndarray]) -> float:
        """
        Compute the aggregated score of the given explanations.

        Parameters
        ----------
        explanations
            Explanation for the inputs, labels to evaluate.
        batch_size
            Number of samples to evaluate at once.

        Returns
        -------
        score
            Score of the explanations.
        """
        aggregated_results = []
        for exp_batch in batch_tensor(explanations, self.batch_size):
            batch_results = self.detailed_evaluate(exp_batch)
            aggregated_results.extend(batch_results)
        return float(np.mean(aggregated_results))

    def __call__(self,
                 explanations: Union[tf.Tensor, np.ndarray],
                 batch_size: int = 32) -> float:
        """Evaluate alias"""
        return self.evaluate(explanations)


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
    activation
        A string that belongs to [None, 'sigmoid', 'softmax']. Specify if we should add
        an activation layer once the model has been called. It is useful, for instance
        if you want to measure a 'drop of probability' by adding a sigmoid or softmax
        after getting your logits. If None does not add a layer to your model.
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
                 operator: Optional[Union[Tasks, str, OperatorSignature]] = None,
                 activation: Optional[str] = None,
                 ):
        # pylint: disable=R0913
        super().__init__(model, inputs, targets, batch_size, activation)

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


class BaseRandomizationMetric(ExplainerMetric, ABC):
    """
    Base class for randomization-based sanity check metrics.

    These metrics compare explanations before and after some perturbation
    (to targets, model, or inputs) to verify explainer sensitivity.

    Parameters
    ----------
    model
        Model used for computing explanations.
    inputs
        Input samples to be explained.
    targets
        One-hot encoded labels or regression targets.
    batch_size
        Number of samples to evaluate at once.
    activation
        Optional activation layer to add after model.
    seed
        Random seed for reproducibility.
    """

    def __init__(self,
                 model: Callable,
                 inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 targets: Optional[Union[tf.Tensor, np.ndarray]],
                 batch_size: Optional[int] = 64,
                 activation: Optional[str] = None,
                 seed: int = 42):
        super().__init__(model=model, inputs=inputs, targets=targets,
                         batch_size=batch_size, activation=activation)
        self.seed = seed
        tf.random.set_seed(self.seed)

        if self.targets is None:
            self.targets = self.model.predict(inputs, batch_size=batch_size)
        self.n_classes = int(self.targets.shape[-1])

    @abstractmethod
    def _get_perturbed_context(self,
                               inputs: tf.Tensor,
                               targets: tf.Tensor,
                               explainer: Union[WhiteBoxExplainer, BlackBoxExplainer]) -> tuple:
        """
        Prepare perturbed inputs/targets/explainer for comparison.

        Returns
        -------
        tuple
            (perturbed_inputs, perturbed_targets, perturbed_explainer)
        """
        raise NotImplementedError()

    @abstractmethod
    def _compute_similarity(self,
                            exp_original: tf.Tensor,
                            exp_perturbed: tf.Tensor) -> tf.Tensor:
        """
        Compute similarity metric between original and perturbed explanations.

        Returns
        -------
        tf.Tensor
            Per-sample similarity scores of shape (B,).
        """
        raise NotImplementedError()

    def _preprocess_explanation(self, exp: tf.Tensor) -> tf.Tensor:
        """Ensure consistent explanation shape."""
        exp = tf.convert_to_tensor(exp, dtype=tf.float32)
        if exp.shape.rank == 3:
            exp = exp[..., tf.newaxis]
        return exp

    def _cleanup_perturbed_context(self, explainer, *args):  # pylint: disable=unused-argument
        """Hook for cleanup after perturbed context usage. Override if needed."""

    def _batch_evaluate(self,
                        inputs: tf.Tensor,
                        targets: tf.Tensor,
                        explainer: Union[WhiteBoxExplainer, BlackBoxExplainer]) -> tf.Tensor:
        """Compute per-sample scores for a batch."""
        # Original explanations
        exp_original = explainer.explain(inputs=inputs, targets=targets)
        exp_original = self._preprocess_explanation(exp_original)

        # Get perturbed context and compute explanations
        p_inputs, p_targets, p_explainer = self._get_perturbed_context(
            inputs, targets, explainer)
        exp_perturbed = p_explainer.explain(inputs=p_inputs, targets=p_targets)
        exp_perturbed = self._preprocess_explanation(exp_perturbed)

        # Cleanup if necessary (eg restore model weights in model randomization)
        self._cleanup_perturbed_context(p_explainer, inputs, targets, explainer)

        return self._compute_similarity(exp_original, exp_perturbed)

    def evaluate(self, explainer: Union[WhiteBoxExplainer, BlackBoxExplainer]) -> float:
        """Compute mean similarity score over the dataset."""
        scores = None
        for inp_batch, tgt_batch in batch_tensor(
                (self.inputs, self.targets), self.batch_size or len(self.inputs)):
            batch_scores = self._batch_evaluate(
                tf.convert_to_tensor(inp_batch, dtype=tf.float32),
                tf.convert_to_tensor(tgt_batch, dtype=tf.float32),
                explainer)
            scores = batch_scores if scores is None else tf.concat([scores, batch_scores], axis=0)
        return float(tf.reduce_mean(scores))
