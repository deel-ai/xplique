"""
Utilities for displaying images with bounding boxes and optional heatmap overlays.
"""
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
                             title: Optional[str] = None,
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
    title
        Optional title for the plot.
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
        if color is None and verbose:
            print(
                f"Warning: No color defined for class '{classes_labels[cl]}'. "
                f"Using default color 'black'.")
        name = class_id_to_label.get(cl, 'unknown')
        found_labels.add(name)
        if verbose:
            print(
                f"cl:{cl}, Drawing box for {name} with color {color} at coords "
                f"({xmin}, {ymin}, {xmax}, {ymax}) with score {score:.2f}")
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=color, linewidth=2))
        ax.text(
            xmin,
            ymin - 15,
            f"{score:.2f}",
            color=color,
            fontsize=10,
            bbox={'facecolor': 'white', 'alpha': 0.0})

    handles = [mpatches.Patch(color=color, label=label)
               for label, color in label_to_color.items() if label in found_labels]
    plt.legend(handles=handles)
    if title is not None:
        ax.set_title(title)
    return fig


# pylint: disable=too-many-branches
def display_images_with_boxes(images: list,
                               multibox_results_list: list[MultiBoxTensor],
                               box_manager: BoxManager,
                               classes_labels: list[str],
                               label_to_color: dict,
                               heatmaps: Optional[list[np.ndarray]] = None,
                               cmap: Optional[str] = "viridis",
                               alpha: Optional[float] = 0.5,
                               titles: Optional[list[str]] = None,
                               verbose: bool = False):
    """
    Display multiple images with bounding boxes, optionally with heatmap overlays.

    Parameters
    ----------
    images
        List of images to display.
    multibox_results_list
        List of bounding box annotations for each image.
    box_manager
        Box manager for coordinate operations.
    classes_labels
        List of class labels.
    label_to_color
        Dictionary mapping labels to colors.
    heatmaps
        Optional list of explanation heatmaps (2D arrays) to overlay on the images.
    cmap
        Optional Matplotlib colormap for the explanation heatmaps.
    alpha
        Optional Alpha transparency for the explanation heatmap overlays.
    titles
        Optional list of titles for each subplot.
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
            f"({len(multibox_results_list)})")

    if heatmaps is not None and len(heatmaps) != len(images):
        raise ValueError(
            f"Number of heatmaps ({len(heatmaps)}) must match number of images ({len(images)})")

    if titles is not None and len(titles) != len(images):
        raise ValueError(
            f"Number of titles ({len(titles)}) must match number of images ({len(images)})")

    num_images = len(images)
    class_id_to_label = {i: classes_labels[i] for i in range(len(classes_labels))}

    # Calculate grid dimensions (try to make it as square as possible)
    num_cols = int(np.ceil(np.sqrt(num_images)))
    num_rows = int(np.ceil(num_images / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

    # Handle case where we have only one subplot
    if num_images == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx in range(num_images):
        ax = axes[idx]
        image = images[idx]
        multibox_results = multibox_results_list[idx]

        ax.imshow(image)

        # Overlay heatmap if provided
        if heatmaps is not None:
            ax.imshow(heatmaps[idx], cmap=cmap, alpha=alpha)

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
            if color is None and verbose:
                print(
                    f"Warning: No color defined for class '{classes_labels[cl]}'. "
                    f"Using default color 'black'.")
            name = class_id_to_label.get(cl, 'unknown')
            found_labels.add(name)
            if verbose:
                print(
                    f"cl:{cl}, Drawing box for {name} with color {color} at coords "
                    f"({xmin}, {ymin}, {xmax}, {ymax}) with score {score:.2f}")
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color=color, linewidth=2))
            ax.text(
                xmin,
                ymin - 15,
                f"{score:.2f}",
                color=color,
                fontsize=10,
                bbox={'facecolor': 'white', 'alpha': 0.0})

        handles = [mpatches.Patch(color=color, label=label)
                   for label, color in label_to_color.items() if label in found_labels]
        ax.legend(handles=handles)

        # Set title if provided
        if titles is not None:
            ax.set_title(titles[idx])

    # Hide unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    return fig
