"""
Utilities for displaying images with bounding boxes and optional heatmap overlays.
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL.Image import Image as PILImage

from xplique.utils_functions.object_detection.base.box_manager import (
    BaseBoxCoordinatesTranslator,
    BoxFormat,
    BoxType,
    NumpyBoxCoordinatesTranslator,
)
from xplique.utils_functions.object_detection.base.multi_box_tensor import BaseMultiBoxTensor
from xplique.utils_functions.object_detection.tf.box_manager import TfBoxCoordinatesTranslator
from xplique.utils_functions.object_detection.tf.multi_box_tensor import TfMultiBoxTensor

from ..types import Dict, List, Optional, Tuple, Union

try:
    from xplique.utils_functions.object_detection.torch.box_manager import (
        TorchBoxCoordinatesTranslator,
    )
    from xplique.utils_functions.object_detection.torch.multi_box_tensor import TorchMultiBoxTensor
except ImportError:
    # If PyTorch is not installed, we can still use the plotting utilities without the
    # torch-specific translator and tensor.
    TorchMultiBoxTensor = None
    TorchBoxCoordinatesTranslator = None


_TARGET_BOX_TYPE = BoxType(BoxFormat.XYXY, is_normalized=False)


def _get_image_size(image) -> Tuple[int, int]:
    """Return (width, height) for both PIL images and NumPy arrays."""
    if isinstance(image, np.ndarray):
        return (image.shape[1], image.shape[0])  # (W, H) from (H, W, C)
    return image.size  # PIL Image already returns (W, H)


def _make_translator(multibox_results, box_type) -> BaseBoxCoordinatesTranslator:
    """Instantiate the right BoxCoordinatesTranslator based on tensor framework."""
    if TorchMultiBoxTensor is not None and isinstance(multibox_results, TorchMultiBoxTensor):
        return TorchBoxCoordinatesTranslator(box_type, _TARGET_BOX_TYPE)
    if isinstance(multibox_results, TfMultiBoxTensor):
        return TfBoxCoordinatesTranslator(box_type, _TARGET_BOX_TYPE)
    return NumpyBoxCoordinatesTranslator(box_type, _TARGET_BOX_TYPE)


def _draw_boxes_on_ax(
    ax: plt.Axes,
    image: Union[np.ndarray, PILImage],
    multibox_results: BaseMultiBoxTensor,
    classes_labels: List[str],
    label_to_color: Dict[str, str],
    box_translator: BaseBoxCoordinatesTranslator,
    heatmap: Optional[np.ndarray] = None,
    cmap: Optional[str] = "viridis",
    alpha: Optional[float] = 0.5,
    title: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Draw a single image with boxes onto an existing Axes using a pre-built translator."""
    class_id_to_label = {i: classes_labels[i] for i in range(len(classes_labels))}

    ax.imshow(image)

    if heatmap is not None:
        ax.imshow(heatmap, cmap=cmap, alpha=alpha)

    boxes = multibox_results.boxes()
    scores = multibox_results.scores()
    probas = multibox_results.probas()

    found_labels = set()
    for box_coords, score, proba in zip(boxes, scores, probas):
        translated = box_translator.translate(
            box_coords[np.newaxis], image_size=_get_image_size(image)
        )
        xmin, ymin, xmax, ymax = box_translator.box_manager.to_numpy_tuple(*translated[0])
        cl = box_translator.box_manager.probas_argmax(proba)
        color = label_to_color.get(classes_labels[cl])
        if color is None and verbose:
            print(
                f"Warning: No color defined for class '{classes_labels[cl]}'. "
                f"Using default color 'black'."
            )
        name = class_id_to_label.get(cl, "unknown")
        found_labels.add(name)
        if verbose:
            print(
                f"cl:{cl}, Drawing box for {name} with color {color} at coords "
                f"({xmin}, {ymin}, {xmax}, {ymax}) with score {score:.2f}"
            )
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=color, linewidth=2
            )
        )
        ax.text(
            xmin,
            ymin - 15,
            f"{score:.2f}",
            color=color,
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.0},
        )

    handles = [
        mpatches.Patch(color=color, label=label)
        for label, color in label_to_color.items()
        if label in found_labels
    ]
    ax.legend(handles=handles)
    if title is not None:
        ax.set_title(title)


def plot_image_detections(
    image: Union[np.ndarray, PILImage],
    multibox_results: BaseMultiBoxTensor,
    classes_labels: List[str],
    label_to_color: Dict[str, str],
    box_type: BoxType = BoxType(BoxFormat.XYXY, is_normalized=False),
    heatmap: Optional[np.ndarray] = None,
    cmap: Optional[str] = "viridis",
    alpha: Optional[float] = 0.5,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    verbose: bool = False,
):
    """
    Display an image with bounding boxes, optionally with a heatmap overlay.

    Parameters
    ----------
    image
        Input image to display. Accepts either a PIL ``Image`` or a NumPy array
        of shape (H, W, C).
    multibox_results
        Bounding box annotations.
    classes_labels
        List of class labels.
    label_to_color
        Dictionary mapping labels to colors.
    box_type
        Box type describing the coordinate format and normalization of the boxes
        stored in multibox_results. Defaults to XYXY non-normalized (pixel coordinates).
    heatmap
        Optional explanation heatmap (2D array) to overlay on the image.
    cmap
        Optional Matplotlib colormap for the explanation heatmap.
    alpha
        Optional Alpha transparency for the explanation heatmap overlay.
    title
        Optional title for the plot.
    ax
        Optional Matplotlib Axes object to plot on. If None, a new figure and axes will be created.
    verbose
        Whether to print debug information.

    Returns
    -------
    fig
        Matplotlib figure object.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    box_translator = _make_translator(multibox_results, box_type)
    _draw_boxes_on_ax(
        ax,
        image,
        multibox_results,
        classes_labels,
        label_to_color,
        box_translator,
        heatmap,
        cmap,
        alpha,
        title,
        verbose,
    )
    return fig


def plot_images_detections(
    images: List[Union[np.ndarray, PILImage]],
    multibox_results_list: List[BaseMultiBoxTensor],
    classes_labels: List[str],
    label_to_color: Dict[str, str],
    box_type: BoxType = BoxType(BoxFormat.XYXY, is_normalized=False),
    heatmaps: Optional[List[np.ndarray]] = None,
    cmap: Optional[str] = "viridis",
    alpha: Optional[float] = 0.5,
    titles: Optional[List[str]] = None,
    num_cols: int = 5,
    verbose: bool = False,
):
    """
    Display multiple images with bounding boxes, optionally with heatmap overlays.

    Parameters
    ----------
    images
        List of images to display.
        Each image can be a PIL ``Image`` or a NumPy array of shape (H, W, C).
    multibox_results_list
        List of bounding box annotations for each image.
    classes_labels
        List of class labels.
    label_to_color
        Dictionary mapping labels to colors.
    box_type
        Box type describing the coordinate format and normalization of the boxes
        stored in multibox_results. Defaults to XYXY non-normalized (pixel coordinates).
    heatmaps
        Optional list of explanation heatmaps (2D arrays) to overlay on the images.
    cmap
        Optional Matplotlib colormap for the explanation heatmaps.
    alpha
        Optional Alpha transparency for the explanation heatmap overlays.
    titles
        Optional list of titles for each subplot.
    num_cols
        Number of columns in the subplot grid. Default is 5.
    verbose
        Whether to print debug information.

    Returns
    -------
    fig
        Matplotlib figure object.

    Raises
    ------
    ValueError
        If the number of images and multibox_results are not equal.
    """
    # Validate inputs
    if len(images) != len(multibox_results_list):
        raise ValueError(
            f"Number of images ({len(images)}) must match number of multibox_results "
            f"({len(multibox_results_list)})"
        )

    if heatmaps is not None and len(heatmaps) != len(images):
        raise ValueError(
            f"Number of heatmaps ({len(heatmaps)}) must match number of images ({len(images)})"
        )

    if titles is not None and len(titles) != len(images):
        raise ValueError(
            f"Number of titles ({len(titles)}) must match number of images ({len(images)})"
        )

    num_images = len(images)

    # Build translator once for the first element (framework is the same for all)
    box_translator = _make_translator(multibox_results_list[0], box_type)

    num_cols = min(num_cols, num_images)
    num_rows = int(np.ceil(num_images / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

    # Handle case where we have only one subplot
    if num_images == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx in range(num_images):
        _draw_boxes_on_ax(
            axes[idx],
            images[idx],
            multibox_results_list[idx],
            classes_labels,
            label_to_color,
            box_translator,
            heatmaps[idx] if heatmaps is not None else None,
            cmap,
            alpha,
            titles[idx] if titles is not None else None,
            verbose,
        )

    # Hide unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    return fig
