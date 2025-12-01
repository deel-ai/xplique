"""
Pretty plots option for explanations
"""
from math import ceil

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from ..types import Optional, Union


def _normalize(image: Union[tf.Tensor, np.ndarray]) -> np.ndarray:
    """
    Normalize an image in [0, 1].

    Parameters
    ----------
    image
        Image to prepare.

    Returns
    -------
    image
        Image ready to be used with matplotlib (in range[0, 1]).
    """
    image = np.array(image, np.float32)

    image -= image.min()
    image /= image.max()

    return image

def _clip_percentile(tensor: Union[tf.Tensor, np.ndarray],
                     percentile: float) -> np.ndarray:
    """
    Apply clip according to percentile value (percentile, 100-percentile) of a tensor
    only if percentile is not None.

    Parameters
    ----------
    tensor
        tensor to clip.

    Returns
    -------
    tensor_clipped
        Tensor clipped accordingly to the percentile value.
    """

    assert 0. <= percentile <= 100., "Percentile value should be in [0, 100]"

    if percentile is not None:
        clip_min = np.percentile(tensor, percentile)
        clip_max = np.percentile(tensor, 100. - percentile)
        tensor = np.clip(tensor, clip_min, clip_max)

    return tensor


def _clip_normalize(explanation: Union[tf.Tensor, np.ndarray],
                       clip_percentile: Optional[float] = 0.1,
                       absolute_value: bool = False,
                       normalize: bool = True) -> Union[tf.Tensor, np.ndarray]:
    if absolute_value:
        explanation = np.abs(explanation)

    if clip_percentile:
        explanation = _clip_percentile(explanation, clip_percentile)

    if normalize:
        explanation = _normalize(explanation)

    return explanation


def generate_heatmap(explanation,
                     size: tuple,
                     clip_percentile: Optional[float] = 0.1,
                     absolute_value: bool = False,
                     normalize_value: bool = True) -> np.ndarray:
    """
    Generate a heatmap from the explanation to a specified 2d size.

    Parameters
    ----------
    explanation
        Attribution / heatmap to plot.
    size
        Target size of the heatmap (height, width).
    clip_percentile
        Percentile value to use if clipping is needed, e.g a value of 1 will perform a clipping
        between percentile 1 and 99. This parameter allows to avoid outliers in case of too
        extreme values.
    absolute_value
        Whether an absolute value is applied to the explanations.
    normalize_value
        Whether an normalization is applied to the explanations.

    Returns
    -------
    heatmap
        The generated heatmap as a numpy array.
    """
    if len(explanation.shape) == 4:
        raise ValueError("Explanation should be 2D or 3D (with channels reduced), "
                         f"got shape {explanation.shape}.")

    heatmap = _clip_normalize(explanation, clip_percentile, absolute_value, normalize_value)

    # resize the explanation to match the image size
    if size is not None and size != heatmap.shape[:2]:
        heatmap = tf.image.resize(heatmap, size,
                                  method=tf.image.ResizeMethod.BILINEAR).numpy()
    return heatmap


def plot_attribution(explanation,
                      image: Optional[np.ndarray] = None,
                      cmap: str = "jet",
                      alpha: float = 0.5,
                      clip_percentile: Optional[float] = 0.1,
                      absolute_value: bool = False,
                      normalize_value: bool = True,
                      **plot_kwargs):
    """
    Displays a single explanation and the associated image (if provided).
    Applies a series of pre-processing to facilitate the interpretation of heatmaps.

    Parameters
    ----------
    explanation
        Attribution / heatmap to plot.
    image
        Image associated to the explanations.
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
    normalize_value
        Whether an normalization is applied to the explanations.
    plot_kwargs
        Additional parameters passed to `plt.imshow()`.
    """
    if image is not None:
        image = _normalize(image)
        if image.shape[-1] == 1:
            plt.imshow(image[:, :, 0], cmap="Greys")
        else:
            plt.imshow(image)

    if len(explanation.shape) == 4: # images channel are reduced
        explanation = np.mean(explanation, -1)

    explanation = generate_heatmap(explanation,
                                   size=image.shape[:2] if image is not None else None,
                                   clip_percentile=clip_percentile,
                                   absolute_value=absolute_value,
                                   normalize_value=normalize_value)

    plt.imshow(explanation, cmap=cmap, alpha=alpha, **plot_kwargs)
    plt.axis('off')


def _adjust_figure(cols, rows, img_size, spacing, margin, l_width, l_height):
    figwidth = cols * img_size + (cols-1) * spacing + 2 * margin
    figheight = rows * img_size * l_height/l_width + (rows-1) * spacing + 2 * margin

    left = margin/figwidth
    bottom = margin/figheight

    fig = plt.figure()
    fig.set_size_inches(figwidth, figheight)

    fig.subplots_adjust(
        left = left,
        bottom = bottom,
        right = 1.-left,
        top = 1.-bottom,
        wspace = spacing/img_size,
        hspace= spacing/img_size * l_width/l_height
    )
    return fig


def plot_attributions(
        explanations: Union[tf.Tensor, np.ndarray],
        images: Optional[Union[tf.Tensor, np.ndarray]] = None,
        cmap: str = "viridis",
        alpha: float = 0.5,
        clip_percentile: Optional[float] = 0.1,
        absolute_value: bool = False,
        cols: int = 5,
        img_size: float = 2.,
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
    img_size
        Size of each subplots (in inch), considering we keep aspect ratio.
    plot_kwargs
        Additional parameters passed to `plt.imshow()`.
    """
    if images is not None:
        assert len(images) == len(explanations), "If you provide images, there must be as many " \
                                                 "as explanations."

    rows = ceil(len(explanations) / cols)
    # get width and height of our images
    l_width, l_height = explanations.shape[1:3]

    # define the figure margin, width, height in inch
    margin = 0.3
    spacing = 0.3
    _adjust_figure(cols, rows, img_size, spacing, margin, l_width, l_height)

    for i, explanation in enumerate(explanations):
        plt.subplot(rows, cols, i+1)
        plot_attribution(
            explanation,
            image=images[i] if images is not None else None,
            cmap=cmap,
            alpha=alpha,
            clip_percentile=clip_percentile,
            absolute_value=absolute_value,
            **plot_kwargs)

def plot_maco(image, alpha, percentile_image=1.0, percentile_alpha=80):
    """
    Plot maco feature visualization image (take care of merging the alpha).

    Parameters
    ----------
    image
        Image to plot.
    alpha
        Alpha channel to plot.
    percentile_image
        Percentile value to use to ceil the image and avoid extreme values.
    percentile_alpha
        Percentile value to use to ceil the alpha channel. A higher value will result in a more
        transparent image with only the most important features.
    """

    image = np.array(image).copy()
    image = _clip_percentile(image, percentile_image)

    alpha = np.mean(np.array(alpha).copy(), -1, keepdims=True)
    alpha = np.clip(alpha, 0, np.percentile(alpha, percentile_alpha))
    alpha = alpha / alpha.max()

    image = image * alpha
    image = _normalize(image)

    plt.imshow(np.concatenate([image, alpha], -1))
    plt.axis('off')


def plot_examples(
        examples: np.ndarray,
        distances: float = None,
        labels: np.ndarray = None,
        test_labels: np.ndarray = None,
        predicted_labels: np.ndarray = None,
        img_size: float = 2.,
):
    """
    This function is for image data, it show the returns of the explain function.

    Parameters
    ---------
    examples
        Represente the k nearest neighbours of the input. (n, k+1, h, w, c)
    distances
        Distance between input data and examples.
    labels
        Labels of the examples.
    labels_test
        Corresponding to labels of the dataset test.
    predicted_labels
        Predicted labels of the examples.
    img_size:
        Size of each subplots (in inch), considering we keep aspect ratio
    """
    # pylint: disable=too-many-arguments
    if distances is not None:
        assert examples.shape[0] == distances.shape[0],\
            "Number of samples treated should match between examples and distances."
        assert examples.shape[1] == distances.shape[1] + 1,\
            "Number of distances for each input must correspond to the number of examples -1."
    if labels is not None:
        assert examples.shape[0] == labels.shape[0],\
            "Number of samples treated should match between examples and labels."
        assert examples.shape[1] == labels.shape[1] + 1,\
            "Number of labels for each input must correspond to the number of examples -1."

    # number of rows depends if weights are provided
    rows_by_input = 1
    rows = rows_by_input * examples.shape[0]
    cols = examples.shape[1]
    # get width and height of our images
    l_width, l_height = examples.shape[2:4]

    # define the figure margin, width, height in inch
    margin = 0.3
    spacing = 0.3
    figwidth = cols * img_size + (cols-1) * spacing + 2 * margin
    figheight = rows * img_size * l_height/l_width + (rows-1) * spacing + 2 * margin

    left = margin/figwidth
    bottom = margin/figheight

    fig = plt.figure()
    fig.set_size_inches(figwidth, figheight)

    fig.subplots_adjust(
        left = left,
        bottom = bottom,
        right = 1.-left,
        top = 1.-bottom,
        wspace = spacing/img_size,
        hspace= spacing/img_size * l_width/l_height
    )

    # configure the grid to show all results
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.figsize"] = [3 * examples.shape[1], 4]

    # loop to organize and show all results
    for i in range(examples.shape[0]):
        for k in range(examples.shape[1]):
            plt.subplot(rows, cols, rows_by_input * i * cols + k + 1)

            # set title
            if k == 0:
                title = "Original image"
                title += f"\nGround Truth: {test_labels[i]}" if test_labels is not None else ""
                title += f"\nPrediction: {predicted_labels[i, k]}"\
                    if predicted_labels is not None else ""
            else:
                title = f"Example {k}"
                title += f"\nGround Truth: {labels[i, k-1]}" if labels is not None else ""
                title += f"\nPrediction: {predicted_labels[i, k]}"\
                    if predicted_labels is not None else ""
                title += f"\nDistance: {distances[i, k-1]:.4f}" if distances is not None else ""
            plt.title(title)

            # plot image
            img = _normalize(examples[i, k])
            if img.shape[-1] == 1:
                plt.imshow(img[:,:,0], cmap="gray")
            else:
                plt.imshow(img)
            plt.axis("off")
    fig.tight_layout()
