"""
Pretty plots option for explanations
"""
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from ..types import Optional, Union

def process_image(
    img,
    percentile = False,
    percentile_val = None
):
    """
    A function that will normalize images and that can clip its values if
    asked.

    Parameters
    ----------
    img:
        The image object to process. It can be a tf.Tensor or a np.ndarray of
        shapes (W, H) or (W, H, C)
    percentile:
        A boolean, that says if we should clipped the normalized values of
        img
    percentile_val:
        The percentile value we want to clip the values of img in.
    """
    try:
        img = img.cpu().numpy()
    except AttributeError:
        try:
            img = img.numpy()
        except AttributeError:
            pass

    img = np.array(img, dtype=np.float32)

    # check if channel first
    if img.shape[0] == 1:
        img = img[0]
    elif img.shape[0] == 3:
        img = np.moveaxis(img, 0, 2)
    # check if cmap
    if img.shape[-1] == 1:
        img = img[:,:,0]
    # normalize
    if img.max() > 1 or img.min() < 0:
        img -= img.min()
        img /= img.max()
    # check if clip percentile
    if percentile:
        img = np.clip(
            img,
            np.percentile(img,percentile_val),
            np.percentile(img, 100-percentile_val)
        )

    return img

def plot_img_explanation(
    explanation: np.ndarray,
    img: Optional[Union[np.ndarray, tf.Tensor]] = None,
    mode: Optional[str] = "abs",
    percentile: bool = False,
    percentile_value: float = 0.05,
    cmap: Optional[str] = "viridis",
    alpha: Optional[float] = 0.6,
    display_original: bool = False,
    **kwargs):
    """
    A plot function which given an explanation highlights the more valuable features.
    If given, the original image can be used as background. In addition, you can
    display this original image too.

    Parameters
    ----------
    explanation:
        The explanation returned by an attribution method

    img:
        The array from which we ask our model an explanation

    mode:
        How you decide to interpret your explanation
        - 'abs': consider the sensitivity, i.e the magnitude of features contributions
        to the decision
        - 'pos': consider only the features that positively contributes to the decision
        - 'neg': consider only the features that negatively contributes to the decision
        - 'none': consider an heatmap of contribution

    percentile:
        A boolean, that says if we should clipped the normalized values of
        img

    percentile_val:
        The percentile value we want to clip the values of img in.

    cmap:
        How to highlight the explanation. For example, you can choose from:
        ['jet', 'viridis', 'plasma', 'inferno', 'magma', 'cividis']

    alpha:
        The alpha parameter (mainly) allows to distinguish more or less the original image
        in the background

    display_original:
        A boolean, which if set to true will plot the plain original image on the left of
        the explanation (therefore img must be provided)

    kwargs:
        Any additional arguments you want to use when calling imshow for the explanation
    """

    # consider mode for explanation
    if mode=="abs":
        exp = np.abs(explanation)
    elif mode=="pos":
        exp = np.where(explanation>0, explanation, 0)
    elif mode=="neg":
        exp = -np.where(explanation<0, explanation, 0)
    else:
        exp = explanation

    exp = process_image(exp, percentile, percentile_value)

    if display_original:
        assert (
            img is not None
        ), "To use display option you need to provide the original image"

        img = process_image(img, percentile=False)

        axarr = plt.subplots(1,2)[1]

        axarr[0].imshow(img)
        axarr[0].axis('off')
        axarr[0].grid(None)

        axarr[1].imshow(img)
        axarr[1].imshow(exp, cmap=cmap, alpha=alpha, **kwargs)
        axarr[1].axis('off')
        axarr[1].grid(None)

    else:

        if img is not None:
            img = process_image(img, percentile=False)
            plt.imshow(img)

        plt.imshow(exp, cmap=cmap, alpha=alpha, **kwargs)
        plt.axis('off')
        plt.grid(None)

def plot_several_images_explanations(
    explanations: np.ndarray,
    imgs: Optional[Union[np.ndarray, tf.Tensor]] = None,
    mode: Optional[str] = "abs",
    percentile: bool = False,
    percentile_value: float = 0.05,
    cmap: Optional[str] = "viridis",
    alpha: Optional[float] = 0.6,
    display_original: bool = False,
    nb_cols: int = 4,
    imsize: float = 2.,
    **kwargs
): # pylint: disable=R0913
    """
    A plot function which given explanations coming from different images highlights
    the more valuable features.
    If given, the original images can be used as background. In addition, you can
    display those original images next to their explanations.

    Parameters
    ----------
    explanations:
        The explanations returned by an attribution method

    img:
        The arrays from which we ask our model an explanation

    mode:
        How you decide to interpret your explanation
        - 'abs': consider the sensitivity, i.e the magnitude of features contributions
        to the decision
        - 'pos': consider only the features that positively contributes to the decision
        - 'neg': consider only the features that negatively contributes to the decision
        - 'none': consider an heatmap of contribution

    percentile:
        A boolean, that says if we should clipped the normalized values of
        img

    percentile_val:
        The percentile value we want to clip the values of img in.

    cmap:
        How to highlight the explanation. For example, you can choose from:
        ['jet', 'viridis', 'plasma', 'inferno', 'magma', 'cividis']

    alpha:
        The alpha parameter (mainly) allows to distinguish more or less the original image
        in the background

    display_original:
        A boolean, which if set to true will plot the original image on the left of
        the corresponding explanations (therefore img must be provided)

    nb_cols:
        Number of columns you want in your plot. If you use display_original, it is better
        to choose an even number. The number of rows is then automatically computed

    imsize:
        Size of each subplots (in inch), considering we keep aspect ratio

    kwargs:
        Any additional arguments you want to use when calling imshow for the explanation
    """
    assert (
        len(explanations.shape)==3
    ), "Shape of explanations must match (N, W, H)"

    nb_explanations = len(explanations)

    # get width and height of our image
    l_width,l_height = explanations.shape[1:]

    # parameter for a nice display
    margin = 0.3 #inch
    spacing = 0.3 #inch

    # define the figure width
    figwidth=nb_cols*imsize+(nb_cols-1)*spacing+2*margin

    nb_explanations = len(explanations)

    if display_original:
        assert(
            len(imgs)==nb_explanations
        ), "You did not provide as many original images as explanations"
        nb_rows = 2*(((nb_explanations-1)//nb_cols) + 1)

        # define others figures parmeters
        figheight=nb_rows*imsize*l_height/l_width+(nb_rows-1)*spacing+2*margin

        left=margin/figwidth
        bottom = margin/figheight

        fig = plt.figure()
        fig.set_size_inches(figwidth,figheight)

        fig.subplots_adjust(
            left=left,
            bottom=bottom,
            right=1.-left,
            top=1.-bottom,
            wspace=spacing/imsize,
            hspace=spacing/imsize*l_width/l_height)

        for exp_index, explanation in enumerate(explanations):
            # add the explanation
            plt.subplot(nb_rows, nb_cols, exp_index*2+1)
            plot_img_explanation(
                explanation,
                img = imgs[exp_index],
                mode = mode,
                percentile = percentile,
                percentile_value = percentile_value,
                cmap = cmap,
                alpha = alpha,
                **kwargs
            )
            plt.title("index: %s"%(exp_index))

            # add the original image
            plt.subplot(nb_rows, nb_cols, exp_index*2+2)

            img = process_image(imgs[exp_index], percentile=False)
            plt.imshow(img)
            plt.axis('off')
            plt.grid(None)
            plt.title("index: %s"%(exp_index))

    else:

        nb_rows = ((nb_explanations-1)//nb_cols) + 1

        # define others figures parmeters
        figheight=nb_rows*imsize*l_height/l_width+(nb_rows-1)*spacing+2*margin

        left=margin/figwidth
        bottom = margin/figheight

        fig = plt.figure()
        fig.set_size_inches(figwidth,figheight)

        fig.subplots_adjust(left=left, bottom=bottom, right=1.-left, top=1.-bottom,
            wspace=spacing/imsize, hspace=spacing/imsize*l_width/l_height)

        if imgs is not None:
            for exp_index, explanation in enumerate(explanations):

                plt.subplot(nb_rows, nb_cols, exp_index+1)
                plot_img_explanation(
                    explanation,
                    img = imgs[exp_index],
                    mode = mode,
                    percentile = percentile,
                    percentile_value = percentile_value,
                    cmap = cmap,
                    alpha = alpha,
                    **kwargs
                )
                plt.title("index: %s"%(exp_index))

        else:

            for exp_index, explanation in enumerate(explanations):
                plt.subplot(nb_rows, nb_cols, exp_index+1)
                plot_img_explanation(
                    explanation,
                    mode = mode,
                    percentile = percentile,
                    percentile_value = percentile_value,
                    cmap = cmap,
                    alpha = alpha,
                    **kwargs
                )
                plt.title("index: %s"%(exp_index))

    plt.tight_layout()

def plot_image_several_explanations(
    explanations: np.ndarray,
    explanations_name: Optional[str] = None,
    img: Optional[Union[np.ndarray, tf.Tensor]] = None,
    mode: Optional[str] = "abs",
    percentile: bool = False,
    percentile_value: float = 0.05,
    cmap: Optional[str] = "viridis",
    alpha: Optional[float] = 0.6,
    display_original: bool = False,
    nb_cols: int = 3,
    imsize: float = 2.,
    **kwargs
): # pylint: disable=R0913
    """
    A plot function which given several explanations coming from different methods for
    one original image highlights the more valuable features (according to the method).
    If given, the original image can be used as background. In addition, you can
    display the original image first as a reference.

    Parameters
    ----------
    explanations:
        The explanations returned by attribution methods for an original image

    explanations_name:
        List of method's name used to obtain the different explanations. If None,
        then it will only be indexed

    img:
        The array from which we ask our models an explanation

    mode:
        How you decide to interpret your explanation
        - 'abs': consider the sensitivity, i.e the magnitude of features contributions
        to the decision
        - 'pos': consider only the features that positively contributes to the decision
        - 'neg': consider only the features that negatively contributes to the decision
        - 'none': consider an heatmap of contribution

    percentile:
        A boolean, that says if we should clipped the normalized values of
        img

    percentile_val:
        The percentile value we want to clip the values of img in.

    cmap:
        How to highlight the explanation. For example, you can choose from:
        ['jet', 'viridis', 'plasma', 'inferno', 'magma', 'cividis']

    alpha:
        The alpha parameter (mainly) allows to distinguish more or less the original image
        in the background

    display_original:
        A boolean, which if set to true will plot the original image first on the grid

    nb_cols:
        Number of columns you want in your plot.

    imsize:
        Size of each subplots (in inch), considering we keep aspect ratio

    kwargs:
        Any additional arguments you want to use when calling imshow for the explanation
    """

    # get width and height of our image
    l_width,l_height = explanations.shape[1:]

    # parameter for a nice display
    margin = 0.3 #inch
    spacing = 0.3 #inch

    # define the figure width
    figwidth=nb_cols*imsize+(nb_cols-1)*spacing+2*margin

    nb_methods = len(explanations)

    if display_original:

        nb_rows = (nb_methods//nb_cols) + 1

        # define others figures parmeters
        figheight=nb_rows*imsize*l_height/l_width+(nb_rows-1)*spacing+2*margin

        left=margin/figwidth
        bottom = margin/figheight

        fig = plt.figure()
        fig.set_size_inches(figwidth,figheight)

        fig.subplots_adjust(
            left=left,
            bottom=bottom,
            right=1.-left,
            top=1.-bottom,
            wspace=spacing/imsize,
            hspace=spacing/imsize*l_width/l_height
        )

        # add the original image
        plt.subplot(nb_rows, nb_cols, 1)

        img = process_image(img, percentile=False)
        plt.imshow(img)
        plt.axis('off')
        plt.grid(None)
        plt.title("Original Image")

        # add all the explanations methods
        for exp_index, explanation in enumerate(explanations):

            plt.subplot(nb_rows, nb_cols, exp_index+2)
            plot_img_explanation(
                explanation,
                img = img,
                mode = mode,
                percentile = percentile,
                percentile_value = percentile_value,
                cmap = cmap,
                alpha = alpha,
                **kwargs
            )

            if explanations_name is not None:
                plt.title(explanations_name[exp_index])
            else:
                plt.title("Explanation: %s"%(exp_index))
        plt.tight_layout()

    else:

        nb_rows = ((nb_methods-1)//nb_cols) + 1

        # define others figures parmeters
        figheight=nb_rows*imsize*l_height/l_width+(nb_rows-1)*spacing+2*margin

        left=margin/figwidth
        bottom = margin/figheight

        fig = plt.figure()
        fig.set_size_inches(figwidth,figheight)

        fig.subplots_adjust(
            left=left,
            bottom=bottom,
            right=1.-left,
            top=1.-bottom,
            wspace=spacing/imsize,
            hspace=spacing/imsize*l_width/l_height
        )

        # add all the explanations methods
        for exp_index, explanation in enumerate(explanations):

            plt.subplot(nb_rows, nb_cols, exp_index+1)
            plot_img_explanation(
                explanation,
                img = img,
                mode = mode,
                percentile = percentile,
                percentile_value = percentile_value,
                cmap = cmap,
                alpha = alpha,
                **kwargs
            )

            if explanations_name is not None:
                plt.title(explanations_name[exp_index])
            else:
                plt.title("Explanation: %s"%(exp_index))

        plt.tight_layout()
