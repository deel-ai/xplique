"""
Test object detection BoundingBoxesExplainer
"""
import os
import sys
sys.path.append(os.getcwd())

import unittest

import tensorflow as tf

from tests.utils import generate_data, generate_object_detection_model

from xplique.commons.exceptions import InvalidModelException
from xplique.attributions import BoundingBoxesExplainer, Rise, Saliency


def test_object_detector():
    """Assert input shape returned is correct"""
    input_shape = (8, 8, 1)
    nb_samples = 3
    nb_labels = 2
    max_nb_boxes = 4
    x, _ = generate_data(input_shape, nb_labels, nb_samples)
    model = generate_object_detection_model(input_shape, max_nb_boxes=max_nb_boxes, nb_labels=nb_labels)

    method = Rise(model, nb_samples=10)

    obj_ref = tf.cast([
                [0, 0, 100, 100, 0.9, 1.0, 0.0],
                [50, 50, 150, 150, 0.5, 1.0, 0.0],
                [0, 10, 20, 30, 0.7, 0.0, 1.0],
            ], tf.float32)

    explainer = BoundingBoxesExplainer(method)

    test_raise_assertion_error = unittest.TestCase().assertRaises
    test_raise_assertion_error(InvalidModelException, explainer.gradient)
    test_raise_assertion_error(InvalidModelException, explainer.batch_gradient)

    phis = explainer(x, obj_ref)

    assert phis.shape == (obj_ref.shape[0], input_shape[0], input_shape[1], 1)


def test_gradient_object_detector():
    """Assert input shape returned is correct"""
    input_shape = (8, 8, 1)
    nb_samples = 3
    nb_labels = 2
    max_nb_boxes = 4
    x, _ = generate_data(input_shape, nb_labels, nb_samples)
    model = generate_object_detection_model(input_shape, max_nb_boxes=max_nb_boxes, nb_labels=nb_labels)

    method = Saliency(model)

    obj_ref = tf.cast([
                [0, 0, 100, 100, 0.9, 1.0, 0.0],
                [50, 50, 150, 150, 0.5, 1.0, 0.0],
                [0, 10, 20, 30, 0.7, 0.0, 1.0],
            ], tf.float32)

    explainer = BoundingBoxesExplainer(method)

    phis = explainer(x, obj_ref)

    assert phis.shape == (obj_ref.shape[0], input_shape[0], input_shape[1], 1)