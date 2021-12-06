# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Pretty plots option for explanations
"""
from math import ceil

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from ..types import Optional
from ..types import Union


def _standardize_image(
    image: Union[tf.Tensor, np.ndarray], clip_percentile: Optional[float] = None
) -> np.ndarray:
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
    cols: int = 5,
    img_size: float = 2.0,
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
    img_size:
        Size of each subplots (in inch), considering we keep aspect ratio
    plot_kwargs
        Additional parameters passed to `plt.imshow()`.
    """
    if images is not None:
        assert len(images) == len(explanations), (
            "If you provide images, there must be as many" "as explanations."
        )
    rows = ceil(len(explanations) / cols)

    # to plot heatmap we need to reduce the channel informations
    if len(explanations.shape) > 3:
        explanations = np.mean(explanations, -1)
    if absolute_value:
        explanations = np.abs(explanations)

    # prepare nice display

    # get width and height of our images
    l_width, l_height = explanations.shape[1:]

    # define the figure margin, width, height in inch
    margin = 0.3
    spacing = 0.3
    figwidth = cols * img_size + (cols - 1) * spacing + 2 * margin
    figheight = rows * img_size * l_height / l_width + (rows - 1) * spacing + 2 * margin

    left = margin / figwidth
    bottom = margin / figheight

    fig = plt.figure()
    fig.set_size_inches(figwidth, figheight)

    fig.subplots_adjust(
        left=left,
        bottom=bottom,
        right=1.0 - left,
        top=1.0 - bottom,
        wspace=spacing / img_size,
        hspace=spacing / img_size * l_width / l_height,
    )

    for i, explanation in enumerate(explanations):
        plt.subplot(rows, cols, i + 1)

        if images is not None:
            img = _standardize_image(images[i])
            if img.shape[-1] == 1:
                plt.imshow(img[:, :, 0], cmap="Greys")
            else:
                plt.imshow(img)

        plt.imshow(
            _standardize_image(explanation, clip_percentile),
            cmap=cmap,
            alpha=alpha,
            **plot_kwargs
        )
        plt.axis("off")
