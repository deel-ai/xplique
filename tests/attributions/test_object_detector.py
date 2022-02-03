import numpy as np
import tensorflow as tf

from xplique.attributions import BoundingBoxesExplainer, Rise
from xplique.attributions.object_detector import SegmentationIouCalculator, BoxIouCalculator, \
    ImageObjectDetectorExplainer, ImageObjectDetectorScoreCalculator, IObjectFormater

from ..utils import almost_equal, generate_model, generate_data


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

    iou_calculator = BoxIouCalculator()

    assert almost_equal(iou_calculator.intersect(m1, m2), 300.0 / 400.0)
    assert almost_equal(iou_calculator.intersect(m1, m3), 0.0)
    assert almost_equal(iou_calculator.intersect(m3, m2), 0.0)
    assert almost_equal(iou_calculator.intersect(m1, m4), 400.0 / 10_000)
    assert almost_equal(iou_calculator.intersect(m2, m4), 300.0 / 10_000)
    assert almost_equal(iou_calculator.intersect(m3, m4), 200.0 / 10_000)


def test_iou_segmentation():
    """Assert the segmentation IoU computation is ok"""

    m1 = np.array([
        [0, 0],
        [1, 1],
    ])[None, :, :]
    m2 = np.array([
        [1, 1],
        [0, 0],
    ])[None, :, :]
    m3 = np.array([
        [1, 0],
        [1, 0],
    ])[None, :, :]

    iou_calculator = SegmentationIouCalculator()

    assert almost_equal(iou_calculator.intersect(m1, m2), 0.0)
    assert almost_equal(iou_calculator.intersect(m1, m3), 1.0/3.0)
    assert almost_equal(iou_calculator.intersect(m3, m2), 1.0/3.0)
    assert almost_equal(iou_calculator.intersect(m1, m1), 1.0)
    assert almost_equal(iou_calculator.intersect(m2, m2), 1.0)


def test_image_object_detector():
    """Assert input shape returned is correct"""
    input_shape = (8, 8, 1)
    nb_labels = 2
    x, y = generate_data(input_shape, nb_labels, nb_labels)
    model = generate_model(input_shape, nb_labels)

    method = Rise(model, nb_samples=10)

    obj_ref = tf.cast([
                [0, 0, 100, 100, 0.9, 1.0, 0.0],
            ], tf.float32)

    class BBoxFormater(IObjectFormater):

        def format_objects(self, predictions):
            if np.all(predictions.numpy() == obj_ref.numpy()):
                return obj_ref[:4], obj_ref[4:5], obj_ref[5:]


            bboxes = tf.cast([
                [0, 10, 20, 30],
                [0, 0, 100, 100],
            ], tf.float32)

            proba = tf.cast([0.9, 0.1], tf.float32)

            classif = tf.cast([[1.0, 0.0], [0.0, 1.0]], tf.float32)

            return bboxes, proba, classif

    formater = BBoxFormater()
    explainer = ImageObjectDetectorExplainer(method, formater, BoxIouCalculator())

    phis = explainer(x, obj_ref)

    assert phis.shape == (1, input_shape[0], input_shape[1])

