from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from xplique.utils_functions.object_detection.base.box_formatter import MultiBoxTensor
from xplique.utils_functions.object_detection.base.box_manager import BoxManager


def display_image_with_boxes(image,
                             multibox_results: MultiBoxTensor,
                             box_manager: BoxManager,
                             classes_labels: list[str],
                             label_to_color: dict,
                             heatmap: Optional[np.ndarray] = None,
                             cmap: Optional[str] = "viridis",
                             alpha: Optional[float] = 0.5,
                             verbose: bool = False):
    """
    Display an image with bounding boxes, optionally with a heatmap overlay.

    Parameters
    ----------
    image
        Base image to display.
    multibox_results
        Bounding box annotations.
    box_manager
        Box manager for coordinate operations.
    classes_labels
        List of class labels.
    label_to_color
        Dictionary mapping labels to colors.
    heatmap
        Optional explanation heatmap (2D array) to overlay on the image.
    cmap
        Optional Matplotlib colormap for the explanation heatmap.
    alpha
        Optional Alpha transparency for the explanation heatmap overlay.
    verbose
        Whether to print debug information.

    Returns
    -------
    fig
        Matplotlib figure object.
    """
    class_id_to_label = {i: classes_labels[i] for i in range(len(classes_labels))}

    fig, ax = plt.subplots()
    ax.imshow(image)

    # Overlay heatmap if provided
    if heatmap is not None:
        ax.imshow(heatmap, cmap=cmap, alpha=alpha)

    # Filter results
    boxes = multibox_results.boxes()
    scores = multibox_results.scores()
    probas = multibox_results.probas()

    found_labels = set()
    for box_coords, score, proba in zip(boxes, scores, probas):
        if box_manager.normalized:
            # denormalize the boxes to pixel coordinates
            xmin, ymin, xmax, ymax = box_manager.denormalize_boxes(box_coords, image.size)
        else:
            # box not normalized, assume they are already in the correct image dimensions
            xmin, ymin, xmax, ymax = box_coords
        xmin, ymin, xmax, ymax = box_manager.to_numpy_tuple(xmin, ymin, xmax, ymax)
        cl = box_manager.probas_argmax(proba)
        color = label_to_color.get(classes_labels[cl])
        if color is None:
            print(
                f"Warning: No color defined for class '{classes_labels[cl]}'. Using default color 'black'.")
        name = class_id_to_label.get(cl, 'unknown')
        found_labels.add(name)
        if verbose:
            print(
                f"cl:{cl}, Drawing box for {name} with color {color} at coords ({xmin}, {ymin}, {xmax}, {ymax}) with score {score:.2f}")
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=color, linewidth=2))
        ax.text(
            xmin,
            ymin - 15,
            f"{score:.2f}",
            color=color,
            fontsize=10,
            bbox=dict(
                facecolor='white',
                alpha=0.0))

    handles = [mpatches.Patch(color=color, label=label)
               for label, color in label_to_color.items() if label in found_labels]
    plt.legend(handles=handles)
    return fig
