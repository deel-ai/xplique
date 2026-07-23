"""Tests for object-detection plotting utilities."""

import warnings

import matplotlib
import numpy as np
import pytest
import tensorflow as tf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xplique.plots.image import generate_heatmap
from xplique.plots.object_detection import plot_image_detections, plot_images_detections
from xplique.utils_functions.object_detection.base.box_manager import BoxFormat, BoxType
from xplique.utils_functions.object_detection.tf.multi_box_tensor import TfMultiBoxTensor


class _NumpyMultiBoxTensor:
    def __init__(self, predictions):
        self.predictions = predictions

    def boxes(self):
        return self.predictions[:, :4]

    def scores(self):
        return self.predictions[:, 4]

    def probas(self):
        return self.predictions[:, 5:]


def _make_multibox(boxes: np.ndarray, num_classes: int = 2) -> TfMultiBoxTensor:
    """Build a minimal TfMultiBoxTensor: (N, 4 + 1 + num_classes)."""
    n = len(boxes)
    scores = np.ones((n, 1), dtype=np.float32)
    probas = np.zeros((n, num_classes), dtype=np.float32)
    probas[:, 0] = 1.0
    tensor = np.concatenate([boxes, scores, probas], axis=1)
    return TfMultiBoxTensor(tf.constant(tensor))


CLASSES = ["cat", "dog"]
COLORS = {"cat": "red", "dog": "blue"}


def test_pixel_boxes_correct_flag():
    """Pixel-coord boxes with is_normalized=False (default) must plot without error."""
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    boxes = np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32)

    plot_image_detections(image, _make_multibox(boxes), CLASSES, COLORS)
    matplotlib.pyplot.close("all")


def test_normalized_boxes_correct_flag():
    """Normalized boxes [0, 1] with is_normalized=True must plot without error."""
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    boxes = np.array([[0.1, 0.1, 0.9, 0.9]], dtype=np.float32)

    box_type = BoxType(BoxFormat.XYXY, is_normalized=True)
    plot_image_detections(image, _make_multibox(boxes), CLASSES, COLORS, box_type=box_type)
    matplotlib.pyplot.close("all")


def test_pixel_boxes_declared_as_normalized_raises():
    """
    Pixel-coord boxes declared as is_normalized=True must raise ValueError.

    The translator skips normalization (assumes [0,1]) then denormalizes by
    multiplying by image size, producing values far beyond the image bounds.
    E.g. xmax=60 on a 64px image -> 60 * 64 = 3840, which exceeds 64 * 3 = 192.
    """
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    boxes = np.array([[10.0, 10.0, 60.0, 60.0]], dtype=np.float32)

    box_type = BoxType(BoxFormat.XYXY, is_normalized=True)
    # wrong: pixel coords passed as normalized

    with pytest.raises(ValueError, match="far exceed image dimensions"):
        plot_image_detections(image, _make_multibox(boxes), CLASSES, COLORS, box_type=box_type)


def test_normalized_boxes_declared_as_pixel_warns():
    """
    Normalized boxes declared as is_normalized=False must issue UserWarning.

    The check fires on the raw input coords before translation: all values are
    below 1.0, which suggests is_normalized=False may be incorrect. A warning
    is used because small pixel-coordinate boxes can also be valid.
    """
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    boxes = np.array([[0.1, 0.1, 0.9, 0.9]], dtype=np.float32)

    box_type = BoxType(BoxFormat.XYXY, is_normalized=False)
    # wrong: normalized coords passed as pixel

    with pytest.warns(UserWarning, match="are all below 1.0"):
        plot_image_detections(image, _make_multibox(boxes), CLASSES, COLORS, box_type=box_type)
    matplotlib.pyplot.close("all")


def test_normalized_boxes_at_warning_limit_does_not_warn():
    """max coord = 1.0 is not below the warning threshold -> no warning."""
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    boxes = np.array([[0.1, 0.1, 1.0, 1.0]], dtype=np.float32)

    box_type = BoxType(BoxFormat.XYXY, is_normalized=False)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        plot_image_detections(image, _make_multibox(boxes), CLASSES, COLORS, box_type=box_type)
    assert not recorded
    matplotlib.pyplot.close("all")


def test_normalized_boxes_above_warning_limit_does_not_warn():
    """max coord = 1.1 is above the warning threshold -> no warning."""
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    boxes = np.array([[0.1, 0.1, 1.1, 1.1]], dtype=np.float32)

    box_type = BoxType(BoxFormat.XYXY, is_normalized=False)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        plot_image_detections(image, _make_multibox(boxes), CLASSES, COLORS, box_type=box_type)
    assert not recorded
    matplotlib.pyplot.close("all")


def test_generate_heatmap_resizes_rank_two_attribution():
    heatmap = generate_heatmap(
        np.arange(6, dtype=np.float32).reshape(2, 3),
        size=(8, 12),
        clip_percentile=None,
        normalize_value=False,
    )

    assert heatmap.shape == (8, 12)


def test_detection_heatmap_uses_the_image_coordinate_extent():
    image = np.zeros((20, 40, 3), dtype=np.float32)
    detections = _NumpyMultiBoxTensor(np.array([[5, 5, 15, 10, 0.9, 1.0, 0.0]]))

    fig = plot_image_detections(
        image,
        detections,
        classes_labels=["object", "other"],
        label_to_color={"object": "red"},
        heatmap=np.zeros((2, 2), dtype=np.float32),
    )
    try:
        image_artist, heatmap_artist = fig.axes[0].images
        assert heatmap_artist.get_extent() == image_artist.get_extent()
    finally:
        plt.close(fig)


def test_plot_images_detections_rejects_empty_lists():
    with pytest.raises(ValueError, match="at least one element"):
        plot_images_detections(
            [],
            [],
            classes_labels=["object", "other"],
            label_to_color={"object": "red"},
        )


def test_plot_images_detections_accepts_numpy_image_batch():
    images = np.zeros((1, 20, 40, 3), dtype=np.float32)
    detections = _NumpyMultiBoxTensor(np.array([[5, 5, 15, 10, 0.9, 1.0, 0.0]]))

    fig = plot_images_detections(
        images,
        [detections],
        classes_labels=["object", "other"],
        label_to_color={"object": "red"},
    )
    plt.close(fig)
