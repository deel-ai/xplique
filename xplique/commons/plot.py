"""
Pretty plots option for explanations
"""
from math import ceil

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from ..types import Optional, Union


def _standardize_image(image: Union[tf.Tensor, np.ndarray],
                       clip_percentile: Optional[float] = None) -> np.ndarray:
    """
    Prepares an image for matplotlib. Applies a normalization and, if specified, a clipping
    operation to remove outliers.

    Parameters
    ----------
    image
        Image to prepare.
    clip_percentile
        Percentile value to use if clipping is needed, e.g a value of 1 will perform a clipping
        between percentile 1 and 99. This parameter allows to avoid outliers  in case of too
        extreme values.

    Returns
    -------
    image
        Image ready to be used with matplotlib.
    """
    image = np.array(image, np.float32)

    # if needed, apply clip_percentile
    if clip_percentile is not None:
        clip_min = np.percentile(image, clip_percentile)
        clip_max = np.percentile(image, 100 - clip_percentile)
        image = np.clip(image, clip_min, clip_max)

    # normalize
    image -= image.min()
    image /= image.max()

    return image


def plot_attributions(
        explanations: Union[tf.Tensor, np.ndarray],
        images: Optional[Union[tf.Tensor, np.ndarray]] = None,
        cmap: str = "viridis",
        alpha: float = 0.5,
        clip_percentile: Optional[float] = 0.1,
        absolute_value: bool = False,
        cols: int = 10,
        **plot_kwargs
):
    """
    Displays a series of explanations and their associated images if these are provided.
    Applies pre-processing to facilitate the interpretation of heatmaps.

    Parameters
    ----------
    explanations
        Attributions values to plot.
    images
        Images associated to explanations. If provided, there must be one explanation for each
        image.
    cmap
        Matplotlib color map to apply.
    alpha
        Opacity value for the explanation.
    clip_percentile
        Percentile value to use if clipping is needed, e.g a value of 1 will perform a clipping
        between percentile 1 and 99. This parameter allows to avoid outliers  in case of too
        extreme values.
    absolute_value
        Whether an absolute value is applied to the explanations.
    cols
        Number of columns.
    plot_kwargs
        Additional parameters passed to `plt.imshow()`.
    """
    if images is not None:
        assert len(images) == len(explanations), "If you provide images, there must be as many" \
                                                 "as explanations."
    rows = ceil(len(explanations) / cols)

    # to plot heatmap we need to reduce the channel informations
    if len(explanations.shape) > 3:
        explanations = np.mean(explanations, -1)
    if absolute_value:
        explanations = np.abs(explanations)

    for i, explanation in enumerate(explanations):
        plt.subplot(rows, cols, i+1)

        if images is not None:
            plt.imshow(_standardize_image(images[i]))

        plt.imshow(_standardize_image(explanation, clip_percentile), cmap=cmap, alpha=alpha,
                   **plot_kwargs)
        plt.axis('off')
