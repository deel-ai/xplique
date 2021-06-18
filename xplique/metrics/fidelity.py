"""
Fidelity (or Faithfullness) metrics
"""

from inspect import isfunction

import numpy as np
import tensorflow as tf

from ..types import Union, Callable, Optional, Tuple

def mu_fidelity(model: tf.keras.Model,
                inputs: tf.Tensor,
                labels: tf.Tensor,
                explanations: tf.Tensor,
                grid_size: Optional[int] = None,
                subset_percent: float = 0.1,
                baseline_mode: Union[float, Callable] = 0.0,
                nb_samples: int = 200,
                batch_size: int = 64) -> Tuple[float, np.ndarray]:
                # pylint: disable=R0913
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
        Input samples, with N number of samples, W & H the sample dimensions, and C the
        number of channels.
    labels
        One hot encoded labels for each sample, with N the number of samples, and L
        the number of classes.
    explanations
        Feature attributions for each samples, with N number of samples, W & H the sample
        dimensions.
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
    batch_size
        Number of samples to explain at once, if None compute all at once.

    Returns
    -------
    fidelity_score
        Metric score.
    predictions_attributions
        Values of each predictions (index 0) and his according attribution sum (index 1) for each
        inputs passed (axis 1).
    """
    # by default use the original equation (pixel-wise modification)
    if grid_size is None:
        grid_size = inputs.shape[1]
    subset_size = int(grid_size ** 2 * subset_percent)  # cardinal of subset

    # prepare the random masks that will designate the modified subset (S in original equation)
    # we ensure the masks have exactly `subset_size` pixels set to baseline
    subset_masks = np.random.rand(nb_samples, grid_size ** 2).argsort(axis=-1) > subset_size
    # and interpolate them if needed
    subset_masks = subset_masks.astype(np.float32).reshape((nb_samples, grid_size, grid_size, 1))
    subset_masks = tf.image.resize(subset_masks, inputs.shape[1:-1], method="nearest")

    base_predictions = np.sum(model.predict(inputs, batch_size=batch_size) * labels, -1)

    predictions, sum_of_attributions = [], []
    correlations = []

    for inp, label, phi, base in zip(inputs, labels, explanations, base_predictions):
        baseline = baseline_mode(inp) if isfunction(baseline_mode) else baseline_mode
        # use the masks to set the selected subsets to baseline state
        degraded_inputs = inp * subset_masks + (1.0 - subset_masks) * baseline
        # measure the two terms that should be correlated
        preds = base - np.sum(model.predict(degraded_inputs, batch_size=batch_size) * label, -1)
        attrs = np.sum(phi * (1.0 - subset_masks), (1, 2, 3))
        corr_score = np.corrcoef(preds, attrs)[0, 1]

        # sanity check: if the model predictions are the same, no variation
        if np.isnan(corr_score):
            corr_score = 0.0

        predictions.append(preds)
        sum_of_attributions.append(attrs)
        correlations.append(corr_score)

    fidelity_score = np.mean(correlations)
    predictions_attributions = np.stack([predictions, sum_of_attributions])

    return fidelity_score, predictions_attributions


def deletion(model: tf.keras.Model,
             inputs: tf.Tensor,
             labels: tf.Tensor,
             explanations: tf.Tensor,
             baseline_mode: Union[float, Callable] = 0.0,
             steps: int = 10,
             batch_size: int = 64) -> Tuple[float, np.ndarray]:
    """
    Used to calculate the deletion score. This score measures the decrease of the prediction
    score when removing progressively the most important pixels. Lower is better.

    Ref. Petsiuk & al., RISE: Randomized Input Sampling for Explanation of Black-box Models (2018).
    https://arxiv.org/abs/1806.07421

    Parameters
    ----------
    model
        Model used for computing metric.
    inputs
        Input samples, with N number of samples, W & H the sample dimensions, and C the
        number of channels.
    labels
        One hot encoded labels for each sample, with N the number of samples, and L
        the number of classes.
    explanations
        Feature attributions for each samples, with N number of samples, W & H the sample
        dimensions.
    baseline_mode
        Value of the baseline state, will be called with the inputs if it is a function.
    steps
        Number of steps between the start and the end state.
    batch_size
        Number of samples to explain at once, if None compute all at once.

    Returns
    -------
    auc
        Metric score, area over the deletion curve, lower is better.
    curve
        Mean curve of deletion. The first index (0) stores the percentages of
        deletion while the second index (1) gives the average results of the predictions.
    """
    return _causal_metric(model, inputs, labels, explanations, "deletion", baseline_mode, steps,
                          batch_size)


def insertion(model: tf.keras.Model,
              inputs: tf.Tensor,
              labels: tf.Tensor,
              explanations: tf.Tensor,
              baseline_mode: Union[float, Callable] = 0.0,
              steps: int = 10,
              batch_size: int = 64) -> Tuple[float, np.ndarray]:
    """
    Used to calculate the insertion score. This score measures the increase of the prediction
    score when adding progressively the most important pixels. Higher is better.

    Ref. Petsiuk & al., RISE: Randomized Input Sampling for Explanation of Black-box Models (2018).
    https://arxiv.org/abs/1806.07421

    Parameters
    ----------
    model
        Model used for computing metric.
    inputs
        Input samples, with N number of samples, W & H the sample dimensions, and C the
        number of channels.
    labels
        One hot encoded labels for each sample, with N the number of samples, and L
        the number of classes.
    explanations
        Feature attributions for each samples, with N number of samples, W & H the sample
        dimensions.
    baseline_mode
        Value of the baseline state, will be called with the inputs if it is a function.
    steps
        Number of steps between the start and the end state.
    batch_size
        Number of samples to explain at once, if None compute all at once.

    Returns
    -------
    auc
        Metric score, area over the insertion curve, higher is better.
    curve
        Mean curve of insertion. The first index (0) stores the percentages of
        insertion while the second index (1) gives the average results of the predictions.
    """
    return _causal_metric(model, inputs, labels, explanations, "insertion", baseline_mode, steps,
                          batch_size)


def _causal_metric(model: tf.keras.Model,
                   inputs: tf.Tensor,
                   labels: tf.Tensor,
                   explanations: tf.Tensor,
                   causal_mode: str = "deletion",
                   baseline_mode: Union[float, Callable] = 0.0,
                   steps: int = 10,
                   batch_size: int = 64):
    """
    Used to compute the insertion and deletion metrics.

    Parameters
    ----------
    model
        Model used for computing metric.
    inputs
        Input samples, with N number of samples, W & H the sample dimensions, and C the
        number of channels.
    labels
        One hot encoded labels for each sample, with N the number of samples, and L
        the number of classes.
    explanations
        Feature attributions for each samples, with N number of samples, W & H the sample
        dimensions.
    causal_mode
        If insertion, the path is baseline to original image, for deletion the path is original
        image to baseline.
    baseline_mode
        Value of the baseline state, will be called with the inputs if it is a function.
    steps
        Number of steps between the start and the end state.
    batch_size
        Number of samples to explain at once, if None compute all at once.

    Returns
    -------
    auc
        Metric score, area over the curve.
    curve
        Mean curve of insertion/deletion. The first index (0) stores the percentages of
        deletion/insertion while the second index (1) gives the average results of the predictions.
    """
    nb_features = np.prod(inputs.shape[1:-1])
    inputs_flatten = inputs.reshape((len(inputs), nb_features, inputs.shape[-1]))
    explanations_flatten = explanations.reshape((len(explanations), -1))

    # for each sample, sort by most important features according to the explanation
    most_important_features = np.argsort(explanations_flatten, axis=-1)[:, ::-1]

    baselines = baseline_mode(inputs) if isfunction(baseline_mode) else np.ones_like(
        inputs) * baseline_mode
    baselines_flatten = baselines.reshape(inputs_flatten.shape)

    steps = np.linspace(0, nb_features, steps, dtype=np.int32)

    scores = []
    if causal_mode == "deletion":
        start = inputs_flatten
        end = baselines_flatten
    elif causal_mode == "insertion":
        start = baselines_flatten
        end = inputs_flatten
    else:
        raise NotImplementedError(f'Unknown causal mode `{causal_mode}`.')

    for step in steps:
        ids_to_flip = most_important_features[:, :step]
        batch_inputs = start.copy()

        for i, ids in enumerate(ids_to_flip):
            batch_inputs[i, ids] = end[i, ids]

        batch_inputs = batch_inputs.reshape((-1, *inputs.shape[1:]))

        predictions = np.sum(model.predict(batch_inputs, batch_size=batch_size) * labels, -1)
        scores.append(predictions)

    curve = (steps / nb_features, np.mean(scores, -1))
    auc = np.trapz(curve[1], curve[0])

    return auc, np.array(curve)
