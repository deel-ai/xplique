"""
Pretty plots option for explanations
"""

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from ..types import Union, Dict, List, Any


def _arrange_subplots(
        explanations: Dict[str, Union[np.array, tf.Tensor]],
) -> Dict[str, Any]:
    """
    Determine how to arrange subplots so that the final output have coherent proportions.

    Parameters
    ----------
    explanations
        dictionary of explanations

    Returns
    -------
    subplot_kwargs
        Dictionary of arguments for the plt.subplots() method.

    """
    # initialize sizes and shapes
    nb_subplots = len(explanations)
    expl_shape = list(explanations.values())[0].shape
    # matplotlib dimension are flipped compare to usual numpy of pandas
    get_size = lambda x, y: (int(y * (1 + expl_shape[0] / 5)), int(x * (1 + expl_shape[1] / 5)))

    found_arrange = False
    # iterate to found arrangement
    while not found_arrange:
        for i in range(1, nb_subplots + 1):
            if nb_subplots % i == 0:
                plot_size = get_size(i, nb_subplots / i)
                if plot_size[0] <= plot_size[1] <= 2 * plot_size[0]:
                    found_arrange = True
                    nrows = i
                    ncols = int(nb_subplots / i)
                    break
        nb_subplots += 1
    return {"nrows": nrows, "ncols": ncols, "figsize": plot_size}


def _show_heatmap(
        axe: matplotlib.axes.Axes,
        explanations: Union[np.array, tf.Tensor],
        features: List[str],
        title: str = "",
        cmap: str = "coolwarm",
        **plot_kwargs
) -> matplotlib.image.AxesImage:
    """
    Display a heatmap representing the attributions for timeseries.

    Parameters
    ----------
    explanations
        Attributions, they are numpy arrays or tensorflow tensors.
        (With features * timesteps format).
    features
        List of features names, should match the first dimension of explanations
    title
        Title of the plot.
    cmap
        Matplotlib color map to apply.
    plot_kwargs
        Additional parameters passed to `plt.imshow()`.

    Returns
    -------
    image
        output of ax.imshow().
    """
    image = axe.imshow(np.array(explanations).transpose(), cmap=cmap, **plot_kwargs)

    axe .set_title(title, fontsize=12)

    axe .set_xlabel("time-steps", fontsize=10)
    time_steps = list(range(-explanations.shape[0], 0))
    axe .set_xticks(np.arange(len(time_steps)))
    axe .set_xticklabels(time_steps, fontsize=5)

    axe .set_ylabel("features", fontsize=10)
    axe .set_yticks(np.arange(len(features)))
    axe .set_yticklabels(features, fontsize=8)

    return image


def plot_attributions(
        explanations: Union[Dict[str, Union[np.array, tf.Tensor]], np.array, tf.Tensor],
        features: List[str],
        title: str = "",
        filepath: str = None,
        cmap: str = "coolwarm",
        colorbar: bool = False,
        **plot_kwargs,
):
    """
    Display a heatmap representing the attributions for timeseries,
    if it is called with a dictionary of explanations,
    it will make subplots and show an heatmap for all elements of the dictionary.

    Parameters
    ----------
    explanations
        Can either be a dictionary of explanations or one explanation.
        The explanations are numpy arrays or tensorflow tensors. (With features * timesteps format).
    features
        List of features names, should match the first dimension of explanations
    title
        Title of the plot.
    filepath
        Path the file will be saved at.
        If None, the function will call plt.show()
    cmap
        Matplotlib color map to apply.
    colorbar
        If the color bar should be shown.
    plot_kwargs
        Additional parameters passed to `plt.imshow()`.
    """
    if isinstance(explanations, dict):
        # find the right arrangement
        subplot_kwargs = _arrange_subplots(explanations)
        nrows, ncols = subplot_kwargs["nrows"], subplot_kwargs["ncols"]
        fig, axes = plt.subplots(**subplot_kwargs)

        # plot multiple heatmaps
        for i, (method, explanation) in enumerate(explanations.items()):
            if nrows == 1 or ncols == 1:
                axe = axes[i]
            else:
                axe = axes[i % nrows, i // nrows]
            image = _show_heatmap(
                axe,
                explanation,
                features,
                title=method,
                cmap=cmap,
                **plot_kwargs
            )
        fig.suptitle(title, fontsize=16)
        plt.subplots_adjust(wspace=0.4)
        if colorbar:
            fig.colorbar(image, ax=axes.ravel().tolist())
    else:
        # plot a single heatmap
        fig, axe = plt.subplots(
            figsize=(1 + explanations.shape[1] / 5, 1 + explanations.shape[0] / 5)
        )
        image = _show_heatmap(axe, explanations, features, title=title, cmap=cmap, **plot_kwargs)
        if colorbar:
            fig.colorbar(image, ax=axe)
        fig.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()
