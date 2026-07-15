"""Regression tests for object-detection heatmap overlays."""

import matplotlib.pyplot as plt
import numpy as np

from xplique.plots.image import generate_heatmap
from xplique.plots.object_detection import plot_image_detections


class _NumpyMultiBoxTensor:
    def __init__(self, predictions):
        self.predictions = predictions

    def boxes(self):
        return self.predictions[:, :4]

    def scores(self):
        return self.predictions[:, 4]

    def probas(self):
        return self.predictions[:, 5:]


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
