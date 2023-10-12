"""
Pretty plots option for explanations
"""

from math import ceil

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from ..types import Union, List, Optional

from .image import _clip_normalize, _adjust_figure


def _show_heatmap(
        explanation: Union[np.array, tf.Tensor],
        features: List[str],
        title: str = "",
        cmap: str = "coolwarm",
        colorbar: bool = False,
        clip_percentile: Optional[float] = 0.1,
        absolute_value: bool = False,
        img_size: float = 2.,
        **plot_kwargs
):
    """
    Display a heatmap representing the attributions for timeseries.

    Parameters
    ----------
    explanations
        Attributions for one prediction, they are numpy arrays or tensorflow tensors.
        (With features * timesteps format).
    features
        List of features names, should match the first dimension of explanations.
    title
        Title of the plot.
    cmap
        Matplotlib color map to apply.
    colorbar
        If the color bar should be shown.
    clip_percentile
        Percentile value to use if clipping is needed, e.g a value of 1 will perform a clipping
        between percentile 1 and 99. This parameter allows to avoid outliers  in case of too
        extreme values.
    absolute_value
        Whether an absolute value is applied to the explanations.
    img_size
        Size of each subplots (in inch), considering we keep aspect ratio.
    plot_kwargs
        Additional parameters passed to `plt.imshow()`.

    Returns
    -------
    image
        output of ax.imshow().
    """

    explanation = _clip_normalize(explanation, clip_percentile, absolute_value)

    image = plt.imshow(np.array(explanation, np.float32).transpose(), cmap=cmap,
                       vmax=1, vmin=-1, **plot_kwargs)

    plt.title(title, fontsize=3 * img_size)

    plt.xlabel("time-steps", fontsize=2 * img_size)
    time_steps = list(range(-explanation.shape[0], 0))
    plt.xticks(np.arange(len(time_steps)), time_steps, fontsize=0.8 * img_size)

    plt.ylabel("features", fontsize=2 * img_size)
    plt.yticks(np.arange(len(features)), features, fontsize=1 * img_size)

    if colorbar:
        plt.colorbar(image)


def plot_timeseries_attributions(
        explanations: Union[tf.Tensor, np.ndarray],
        features: List[str],
        title: Optional[str] = None,
        subtitles: Optional[List[str]] = None,
        filepath: Optional[str] = None,
        cmap: str = "coolwarm",
        colorbar: bool = False,
        clip_percentile: Optional[float] = 0.1,
        absolute_value: bool = False,
        cols: int = 5,
        img_size: float = 3.,
        **plot_kwargs
):
    """
    Displays a series of explanations and their associated images if these are provided.
    Applies pre-processing to facilitate the interpretation of heatmaps.

    Parameters
    ----------
    explanations
        Attributions values to plot.
    features
        List of features names, should match the first dimension of explanations.
    title
        Title of the plot.
    subtitles
        List of titles for the different samples i.e. subplots.
    filepath
        Path the file will be saved at.
        If None, the function will call plt.show()
    cmap
        Matplotlib color map to apply.
    colorbar
        If the color bar should be shown.
    clip_percentile
        Percentile value to use if clipping is needed, e.g a value of 1 will perform a clipping
        between percentile 1 and 99. This parameter allows to avoid outliers  in case of too
        extreme values.
    absolute_value
        Whether an absolute value is applied to the explanations.
    cols
        Number of columns.
    img_size
        Size of each subplots (in inch), considering we keep aspect ratio.
    plot_kwargs
        Additional parameters passed to `plt.imshow()`.
    """
    if subtitles is not None:
        assert len(subtitles) == len(explanations), "If you provide subtitles, " +\
                                                    "there must be as many as explanations."
    else:
        subtitles = [None] * len(explanations)

    rows = ceil(len(explanations) / cols)
    # get width and height of our images
    l_width, l_height = explanations.shape[1:]

    # define the figure margin, width, height in inch
    margin = 0.3 + 0.7 * int(colorbar) + int(title is not None)
    spacing = 1.5
    _adjust_figure(cols, rows, img_size, spacing, margin, l_width, l_height)

    if title is not None:
        plt.suptitle(title, fontsize=4 * img_size)

    for i, explanation in enumerate(explanations):
        plt.subplot(rows, cols, i+1)

        _show_heatmap(explanation, features=features, title=subtitles[i],
                      cmap=cmap, colorbar=colorbar, img_size=img_size,
                      clip_percentile=clip_percentile, absolute_value=absolute_value, **plot_kwargs)
    if filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()
