"""
Test utils functions for object detection
"""

import numpy as np

from xplique.utils_functions.object_detection import _box_iou

from ..utils import almost_equal


def test_iou_mask():
    """Assert the Mask IoU calculation is ok"""
    dtype = np.float32

    m1 = np.array([
        [10, 20, 30, 40]
    ], dtype=dtype)
    m2 = np.array([
        [15, 20, 30, 40]
    ], dtype=dtype)
    m3 = np.array([
        [0, 20, 10, 40]
    ], dtype=dtype)
    m4 = np.array([
        [0, 0, 100, 100]
    ], dtype=dtype)

    assert almost_equal(_box_iou(m1, m2), 300.0 / 400.0)
    assert almost_equal(_box_iou(m1, m3), 0.0)
    assert almost_equal(_box_iou(m3, m2), 0.0)
    assert almost_equal(_box_iou(m1, m4), 400.0 / 10_000)
    assert almost_equal(_box_iou(m2, m4), 300.0 / 10_000)
    assert almost_equal(_box_iou(m3, m4), 200.0 / 10_000)
