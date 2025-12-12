"""
Randomization / sanity-check metrics for explanation methods.

This module implements:
- RandomLogitMetric: sensitivity of explainers to random target logits.
- ModelRandomizationMetric: sensitivity of explainers to model parameter randomization.

In attribution methods that pass the sanity check, explanations should change
significantly when either the target logit or the model parameters are randomized.
Thus, a low similarity (SSIM or Spearman correlation) between explanations
before and after randomization indicates a faithful explainer.
"""
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import tensorflow as tf

from ..attributions.base import BlackBoxExplainer, WhiteBoxExplainer
from ..commons import batch_tensor
from ..types import Callable, Optional, Union
from .base import ExplainerMetric


_EPS = 1e-8


def ssim(a: tf.Tensor,
         b: tf.Tensor,
         batched: bool = False,
         **kwargs) -> tf.Tensor:
    """
    Compute the Structural Similarity Index Measure (SSIM) between two images
    (or batches of images) using TensorFlow's built-in implementation.

    Parameters
    ----------
    a
        First image tensor. If `batched=False`, shape (H, W, C);
        if `batched=True`, shape (B, H, W, C).
    b
        Second image tensor, same shape as `a`.
    batched
        If True, compute SSIM per-sample (return shape (B,)),
        otherwise return a scalar.

    Other Parameters
    ----------------
    win_size : int, default 11
        Size of the gaussian filter. Will be adjusted to not exceed image dimensions.
    filter_sigma : float, default 1.5
        Standard deviation for Gaussian kernel.
    k1 : float, default 0.01
        Algorithm parameter, K1 (small constant).
    k2 : float, default 0.03
        Algorithm parameter, K2 (small constant).

    Returns
    -------
    tf.Tensor
        SSIM value(s). If `batched=True`, shape (B,); otherwise scalar.
        Returns 1.0 when images have zero dynamic range (constant values).

    Notes
    -----
    - Automatically handles zero-dynamic-range cases by returning 1.0
    - Filter size is adapted to not exceed image dimensions
    - Uses `tf.image.ssim` internally
    - Computes data range from min/max of both images combined
    """

    filter_size = int(kwargs.get("win_size", 11))
    filter_sigma = float(kwargs.get("filter_sigma", 1.5))
    k1 = float(kwargs.get("k1", 0.01))
    k2 = float(kwargs.get("k2", 0.03))

    def _adapt_filter_size(img: tf.Tensor, fs: int) -> int:
        # img is (H, W, C)
        h = img.shape[0]
        w = img.shape[1]
        if h is not None and w is not None:
            fs = min(fs, h, w)
            if fs % 2 == 0:
                fs -= 1
            fs = max(fs, 1)
        return fs

    def _ssim_pair(image_a: tf.Tensor, image_b: tf.Tensor) -> tf.Tensor:
        fs = _adapt_filter_size(image_a, filter_size)

        stacked = tf.stack([image_a, image_b], axis=0)
        max_point = tf.reduce_max(stacked)
        min_point = tf.reduce_min(stacked)
        data_range = tf.cast(tf.abs(max_point - min_point), image_a.dtype)

        # safe max_val for tf.image.ssim
        safe_data_range = tf.where(
            data_range > 0,
            data_range,
            tf.constant(1.0, dtype=image_a.dtype),
        )

        ssim_val = tf.image.ssim(
            image_a, image_b,
            max_val=safe_data_range,
            filter_size=fs,
            filter_sigma=filter_sigma,
            k1=k1, k2=k2,
        )
        # if original dynamic range was zero -> SSIM = 1
        return tf.where(
            data_range > 0,
            ssim_val,
            tf.constant(1.0, dtype=image_a.dtype),
        )

    if batched:
        # a, b: (B, H, W, C)
        ssim_values = tf.map_fn(
            lambda ab: _ssim_pair(ab[0], ab[1]),
            (a, b),
            fn_output_signature=a.dtype,
        )
        return ssim_values
    return _ssim_pair(a, b)


def _rankdata_average_ties(x: tf.Tensor) -> tf.Tensor:
    """
    Compute average ranks for tied values in batched data.

    This function assigns ranks to elements in each row of a batch, where tied
    (equal) values receive the average of the positions they occupy. This is
    commonly used in computing Spearman correlation with tie handling.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor of shape (B, F), where B is the batch size and F is the
        number of features per sample.

    Returns
    -------
    tf.Tensor
        Tensor of shape (B, F) containing the average ranks for each element.
        Ranks are 0-indexed (range [0, F-1]).

    Notes
    -----
    The ranking process handles ties by averaging positions:

    1. Values are sorted within each batch
    2. Tied values (equal elements) are grouped together
    3. Each group receives the average position of all its members
    4. Ranks are mapped back to the original order

    Examples
    --------
    >>> x = tf.constant([[30, 10, 20, 20, 10]], dtype=tf.float32)
    >>> _rankdata_average_ties(x)
    <tf.Tensor: shape=(1, 5), dtype=float32, numpy=
    array([[4.0, 0.5, 2.5, 2.5, 0.5]], dtype=float32)>

    The input [30, 10, 20, 20, 10] has:
    - Two 10s at positions 0-1 (sorted) → average rank 0.5
    - Two 20s at positions 2-3 (sorted) → average rank 2.5
    - One 30 at position 4 (sorted) → rank 4.0
    """
    # x: (B, F)
    x = tf.convert_to_tensor(x)
    B = tf.shape(x)[0]
    F = tf.shape(x)[1]

    idx = tf.argsort(x, axis=1, stable=True)                        # (B, F)
    x_sorted = tf.gather(x, idx, batch_dims=1)                      # (B, F)

    # group boundaries where value changes (ties -> same group)
    change = tf.not_equal(x_sorted[:, 1:], x_sorted[:, :-1])        # (B, F-1)
    boundary = tf.concat([tf.zeros((B, 1), tf.bool), change], axis=1)  # (B, F)
    group_id = tf.cumsum(tf.cast(boundary, tf.int32), axis=1)       # (B, F), starts at 0

    pos = tf.tile(tf.range(F)[None, :], [B, 1])                     # (B, F)
    pos_f = tf.cast(pos, tf.float32)

    # unique segment ids across batch
    seg_id = tf.range(B)[:, None] * F + group_id                    # (B, F)
    seg_id_flat = tf.reshape(seg_id, [-1])
    pos_flat = tf.reshape(pos_f, [-1])

    num_segments = B * F
    mean_per_seg = tf.math.unsorted_segment_mean(pos_flat, seg_id_flat, num_segments)
    ranks_sorted = tf.reshape(tf.gather(mean_per_seg, seg_id_flat), (B, F))

    inv = tf.argsort(idx, axis=1)                                   # inverse perm
    ranks = tf.gather(ranks_sorted, inv, batch_dims=1)              # back to original order
    return ranks


def batched_spearman(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """
    Compute the Spearman rank correlation coefficient in batch.

    Parameters
    ----------
    a
        Tensor of shape (B, F). Each row is a feature vector.
    b
        Tensor of shape (B, F). Same shape as `a`.

    Returns
    -------
    tf.Tensor
        Tensor of shape (B,), Spearman correlation per row.
    """
    # ranks via argsort(argsort(.)) but handling ties with average ranks
    rank_a = _rankdata_average_ties(a)
    rank_b = _rankdata_average_ties(b)

    mean_a = tf.reduce_mean(rank_a, axis=1, keepdims=True)
    mean_b = tf.reduce_mean(rank_b, axis=1, keepdims=True)

    da = rank_a - mean_a
    db = rank_b - mean_b

    cov = tf.reduce_mean(da * db, axis=1)              # (B,)
    std_a = tf.math.reduce_std(rank_a, axis=1)         # (B,)
    std_b = tf.math.reduce_std(rank_b, axis=1)         # (B,)

    return cov / (std_a * std_b + _EPS)


class ModelRandomizationStrategy(ABC):
    """
    Interface for model randomization strategies.
    """

    @abstractmethod
    def randomize(self, model: tf.keras.Model) -> tf.keras.Model:
        """
        Randomize the model parameters.

        Parameters
        ----------
        model
            Model to randomize.

        Returns
        -------
        randomized_model
            The same model instance, with randomized parameters.
        """
        raise NotImplementedError()


class ProgressiveLayerRandomization(ModelRandomizationStrategy):
    def __init__(self,
                 stop_layer: Union[str, int, float, List[Union[str, int]]],
                 reverse: bool = True):
        if isinstance(stop_layer, tuple):
            stop_layer = list(stop_layer)
        if not isinstance(stop_layer, (str, int, float, list)):
            raise TypeError("`stop_layer` must be str, int, float or list of str or int.")
        if isinstance(stop_layer, float):
            if not (0.0 <= stop_layer <= 1.0):
                raise ValueError("Fractional `stop_layer` must be in [0, 1].")
        elif isinstance(stop_layer, list):
            for elem in stop_layer:
                if not isinstance(elem, (str, int)):
                    raise TypeError("List elements for `stop_layer` must be str or int.")

        self.stop_layer = stop_layer
        self.reverse = reverse

    def randomize(self, model: tf.keras.Model) -> tf.keras.Model:
        # Only count layers that actually have weights (matches your unit tests)
        weight_layers = [l for l in model.layers if l.get_weights()]
        layer_list = weight_layers[::-1] if self.reverse else weight_layers
        n_layers = len(layer_list)

        # resolve stop_index in this order
        if isinstance(self.stop_layer, str):
            indices = [i for i, l in enumerate(layer_list) if l.name == self.stop_layer]
            if not indices:
                raise ValueError(f"Layer '{self.stop_layer}' not found in model.")
            stop_index = indices[0]
        elif isinstance(self.stop_layer, int):
            stop_index = self.stop_layer
        elif isinstance(self.stop_layer, float):
            stop_index = int(n_layers * self.stop_layer)
        else:  # list
            resolved: List[int] = []
            for elem in self.stop_layer:
                if isinstance(elem, str):
                    idxs = [i for i, l in enumerate(layer_list) if l.name == elem]
                    if not idxs:
                        raise ValueError(f"Layer '{elem}' not found in model.")
                    resolved.append(idxs[0])
                else:
                    resolved.append(elem)
            stop_index = min(resolved)

        stop_index = max(0, min(stop_index, n_layers))

        # Randomize weights UP TO (but excluding) stop_index in traversal order
        for layer in layer_list[:stop_index]:
            current_weights = layer.get_weights()
            if not current_weights:
                continue
            new_weights = [
                tf.random.uniform(shape=w.shape, dtype=tf.as_dtype(w.dtype)).numpy()
                for w in current_weights
            ]
            layer.set_weights(new_weights)

        return model


class RandomLogitMetric(ExplainerMetric):
    """
    Random Logit Invariance metric.

    Tests whether explanations change when the *target logit* is randomized
    to a different class.

    For each sample x, y:
        - compute explanation for the true class y,
        - randomly draw an off-class y' != y,
        - compute explanation for y',
        - measure SSIM between both explanations.

    A low SSIM indicates that explanations are sensitive to the target label
    (desirable if we expect class-specific explanations).

    Ref.: Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018).
    Sanity checks for saliency maps. Advances in neural information processing systems, 31.
    https://arxiv.org/abs/1810.03292

    Parameters
    ----------
    model
        Model used to compute explanations.
    inputs
        Input samples.
    targets
        One-hot encoded labels, shape (N, C).
    batch_size
        Number of samples to evaluate at once.
    activation
        Optional activation applied in the explainer/model, not used directly here.
    seed
        Random seed used when sampling off-classes.

    Returns (API)
    -------------
    evaluate(explainer) -> float
        Mean SSIM over the dataset.
    """
    def __init__(self,
                 model: Callable,
                 inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 targets: Optional[Union[tf.Tensor, np.ndarray]],
                 batch_size: Optional[int] = 64,
                 activation: Optional[str] = None,
                 seed: int = 42):
        super().__init__(model=model, inputs=inputs, targets=targets, batch_size=batch_size, activation=activation)
        if self.targets is None:
            self.targets = self.model.predict(inputs, batch_size=batch_size)
        self.seed = seed

        # infer number of classes from targets or model output
        self.n_classes = int(self.targets.shape[-1])
        tf.random.set_seed(self.seed)

    def _batch_scores(
            self,
            inputs: tf.Tensor,
            targets: tf.Tensor,
            explainer: Union[WhiteBoxExplainer, BlackBoxExplainer]
    ) -> tf.Tensor:
        """
        Compute per-sample SSIM scores for a batch.
        """
        batch_size = tf.shape(inputs)[0]
        true_class = tf.argmax(targets, axis=-1, output_type=tf.int32)  # (B,)

        # sample off-class uniformly in {0, ..., C-1} \ {true_class}
        k = self.n_classes - 1
        rnd = tf.random.uniform(shape=(batch_size,), minval=0, maxval=k, dtype=tf.int32)
        off_class = tf.where(rnd >= true_class, rnd + 1, rnd)

        true_one_hot = tf.cast(targets, tf.float32)
        off_one_hot = tf.one_hot(off_class, depth=self.n_classes, dtype=tf.float32)

        # explanations for true class and random off-class
        exp_true = explainer.explain(inputs=inputs, targets=true_one_hot)
        exp_off = explainer.explain(inputs=inputs, targets=off_one_hot)

        # ensure 4D shape (B, H, W, C_attr)
        exp_true = tf.convert_to_tensor(exp_true, dtype=tf.float32)
        exp_off = tf.convert_to_tensor(exp_off, dtype=tf.float32)

        if exp_true.shape.rank == 3:
            exp_true = exp_true[..., tf.newaxis]
        if exp_off.shape.rank == 3:
            exp_off = exp_off[..., tf.newaxis]

        # SSIM per-sample
        scores = ssim(exp_true, exp_off, batched=True)
        return scores

    def evaluate(self,
                 explainer: Union[WhiteBoxExplainer, BlackBoxExplainer]) -> float:
        """
        Compute the Random Logit Invariance score over the dataset.

        Parameters
        ----------
        explainer
            Attribution method implementing `explain(inputs, targets)`.

        Returns
        -------
        float
            Mean SSIM over the dataset.
        """
        scores = None
        for inp_batch, tgt_batch in batch_tensor(
                (self.inputs, self.targets),
                self.batch_size or len(self.inputs)):
            batch_scores = self._batch_scores(
                tf.convert_to_tensor(inp_batch, dtype=tf.float32),
                tf.convert_to_tensor(tgt_batch, dtype=tf.float32),
                explainer
            )
            scores = batch_scores if scores is None else tf.concat([scores, batch_scores], axis=0)

        return float(tf.reduce_mean(scores))


class ModelRandomizationMetric(ExplainerMetric):
    """
    Model Randomization metric.

    Tests whether explanations degrade when model parameters are randomized.
    This implements a sanity check inspired by Adebayo et al. (2018).

    For each sample x, y:
        - compute explanation under the original model,
        - randomize model parameters according to a strategy,
        - compute explanation under the randomized model,
        - measure Spearman rank correlation between both explanations.

    A low Spearman correlation indicates that explanations are sensitive to
    the model parameters (desirable for a faithful explainer).

    Ref.: Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018).
    Sanity checks for saliency maps. Advances in neural information processing systems, 31.
    https://arxiv.org/abs/1810.03292

    Parameters
    ----------
    model
        Model to be randomized.
    inputs
        Input samples.
    targets
        One-hot encoded labels, shape (N, C),
        or integer labels which will be one-hot encoded.
    explainer
        Attribution method implementing `explain(inputs, targets)`.
    randomization_strategy
        Strategy to randomize the model parameters.
    batch_size
        Number of samples to evaluate at once.
    activation
        Optional activation, not used directly here.
    seed
        Random seed for reproducibility.

    Returns (API)
    -------------
    evaluate(explainer) -> float
        Mean Spearman correlation over the dataset.
    """
    def __init__(self,
                 model: Callable,
                 inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 targets: Optional[Union[tf.Tensor, np.ndarray]],
                 randomization_strategy: ModelRandomizationStrategy = ProgressiveLayerRandomization(0.25),
                 batch_size: Optional[int] = 64,
                 activation: Optional[str] = None,
                 seed: int = 42):
        super().__init__(model=model, inputs=inputs, targets=targets, batch_size=batch_size, activation=activation)
        if self.targets is None:
            self.targets = self.model.predict(inputs, batch_size=batch_size)

        self.randomization_strategy = randomization_strategy
        self.seed = seed

        # Clone the model for randomization
        self.randomized_model = tf.keras.models.clone_model(self.model)
        self.randomized_model.set_weights(self.model.get_weights())

        # infer number of classes from targets or model output
        if self._is_integer_dtype(self.targets):
            sample_input = tf.convert_to_tensor(self.inputs[:1])
            sample_pred = tf.convert_to_tensor(self.model(sample_input))
            self.n_classes = int(sample_pred.shape[-1])
        else:
            self.n_classes = int(self.targets.shape[-1])

        tf.random.set_seed(self.seed)

    def _to_one_hot(self, targets: tf.Tensor) -> tf.Tensor:
        """Convert targets to one-hot if needed."""
        if targets.dtype.is_integer:
            return tf.one_hot(targets, depth=self.n_classes, dtype=tf.float32)
        return tf.cast(targets, tf.float32)

    @staticmethod
    def _is_integer_dtype(x) -> bool:
        """Check if x has integer dtype (Tensor or numpy array)."""
        if isinstance(x, tf.Tensor):
            return x.dtype.is_integer
        if isinstance(x, np.ndarray):
            return np.issubdtype(x.dtype, np.integer)
        return False

    def _batch_scores(
            self,
            inputs: tf.Tensor,
            targets: tf.Tensor,
            explainer: Union[WhiteBoxExplainer, BlackBoxExplainer]
    ) -> tf.Tensor:
        """
        Compute per-sample Spearman scores for a batch.
        """
        one_hot = self._to_one_hot(targets)

        # explanations under original model
        exp_original = explainer.explain(inputs=inputs, targets=one_hot)
        exp_original = tf.convert_to_tensor(exp_original, dtype=tf.float32)
        if exp_original.shape.rank == 3:
            exp_original = exp_original[..., tf.newaxis]

        # randomize the cloned model
        self.randomization_strategy.randomize(self.randomized_model)

        # temporarily swap models in explainer
        original_model = explainer.model
        explainer.model = self.randomized_model

        exp_rand = explainer.explain(inputs=inputs, targets=one_hot)
        exp_rand = tf.convert_to_tensor(exp_rand, dtype=tf.float32)
        if exp_rand.shape.rank == 3:
            exp_rand = exp_rand[..., tf.newaxis]

        # restore original model in explainer
        explainer.model = original_model

        # channel-average if needed, flatten to (B, F)
        if exp_original.shape.rank == 4:
            exp_original = tf.reduce_mean(exp_original, axis=-1)
            exp_rand = tf.reduce_mean(exp_rand, axis=-1)

        exp_original = tf.reshape(exp_original, (tf.shape(inputs)[0], -1))
        exp_rand = tf.reshape(exp_rand, (tf.shape(inputs)[0], -1))

        scores = batched_spearman(exp_original, exp_rand)
        return scores

    def evaluate(self,
                 explainer: Union[WhiteBoxExplainer, BlackBoxExplainer]) -> float:
        """
        Compute the Model Randomization score over the dataset.
        """
        scores = None
        for inp_batch, tgt_batch in batch_tensor(
                (self.inputs, self.targets),
                self.batch_size or len(self.inputs)):
            batch_scores = self._batch_scores(
                tf.convert_to_tensor(inp_batch, dtype=tf.float32),
                tf.convert_to_tensor(tgt_batch),
                explainer
            )
            scores = batch_scores if scores is None else tf.concat([scores, batch_scores], axis=0)

        return float(tf.reduce_mean(scores))
