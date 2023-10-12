"""
Test object detection operator
"""
import os
import sys
sys.path.append(os.getcwd())

from itertools import combinations

import numpy as np
import tensorflow as tf

from tests.utils import almost_equal, generate_object_detection_model, generate_data

from xplique.attributions import Occlusion, SmoothGrad
from xplique.commons.operators import object_detection_operator


def test_object_detector():
    """Assert input shape returned is correct"""
    input_shape = (8, 8, 1)
    nb_samples = 3
    nb_labels = 2
    max_nb_boxes = 4
    x, _ = generate_data(input_shape, nb_labels, nb_samples)
    model = generate_object_detection_model(input_shape, max_nb_boxes=max_nb_boxes, nb_labels=nb_labels)

    # 3 bounding boxes, one for each input sample
    obj_ref = tf.cast([
                [0, 0, 100, 100, 0.9, 1.0, 0.0],
                [50, 50, 150, 150, 0.5, 1.0, 0.0],
                [0, 10, 20, 30, 0.7, 0.0, 1.0],
            ], tf.float32)

    explainer = Occlusion(model, operator=object_detection_operator, patch_size=4, patch_stride=2)

    # test with only one box to explain by image (3, 7)
    phis = explainer(x, obj_ref)
    assert phis.shape == (obj_ref.shape[0], input_shape[0], input_shape[1], 1)
    assert phis[0, 0, 0] != np.nan

    phis2 = explainer(x, tf.expand_dims(obj_ref, axis=1))
    assert phis.shape[:-1] == phis2.shape[:-1]
    assert almost_equal(phis, phis2)


def test_gradient_object_detector():
    """Assert input shape returned is correct"""
    input_shape = (8, 8, 1)
    nb_samples = 3
    nb_labels = 2
    max_nb_boxes = 4
    x, _ = generate_data(input_shape, nb_labels, nb_samples)
    model = generate_object_detection_model(input_shape, max_nb_boxes=max_nb_boxes, nb_labels=nb_labels)

    explainer = SmoothGrad(model, nb_samples=10, operator=object_detection_operator)

    obj_ref = tf.cast([
                [0, 0, 100, 100, 0.9, 1.0, 0.0],
                [50, 50, 150, 150, 0.5, 1.0, 0.0],
                [0, 10, 20, 30, 0.7, 0.0, 1.0],
            ], tf.float32)

    phis = explainer(x, obj_ref)

    assert phis.shape == (obj_ref.shape[0], input_shape[0], input_shape[1], 1)


def test_several_boxes_object_detector():
    """Assert input shape returned is correct"""
    input_shape = (8, 8, 1)
    nb_samples = 3
    nb_labels = 2
    max_nb_boxes = 4
    x, _ = generate_data(input_shape, nb_labels, nb_samples)
    model = generate_object_detection_model(input_shape, max_nb_boxes=max_nb_boxes, nb_labels=nb_labels)

    explainer = SmoothGrad(model, nb_samples=10, operator=object_detection_operator)

    obj_ref = tf.cast([
                [0, 0, 100, 100, 0.9, 1.0, 0.0],
                [50, 50, 150, 150, 0.5, 1.0, 0.0],
                [0, 10, 20, 30, 0.7, 0.0, 1.0],
            ], tf.float32)

    phis = explainer(x, obj_ref)
    assert phis.shape[:3] == (obj_ref.shape[0], input_shape[0], input_shape[1])

    several_object_refs = tf.tile(tf.expand_dims(obj_ref, axis=1), [1, 5, 1])
    
    phis = explainer(x, several_object_refs)

    assert phis.shape == (several_object_refs.shape[0], input_shape[0], input_shape[1], 1)


def test_all_object_detector_operators():
    """Assert input shape returned is correct"""
    input_shape = (8, 8, 1)
    nb_samples = 3
    nb_labels = 2
    max_nb_boxes = 4
    x, _ = generate_data(input_shape, nb_labels, nb_samples)
    model = generate_object_detection_model(input_shape, max_nb_boxes=max_nb_boxes, nb_labels=nb_labels)

    obj_ref = tf.cast([
                [0, 0, 100, 100, 0.9, 1.0, 0.0],
                [50, 50, 150, 150, 0.5, 1.0, 0.0],
                [0, 10, 20, 30, 0.7, 0.0, 1.0],
            ], tf.float32)

    # set params
    parameters_normal = {
        "include_detection_probability": True,
        "include_classification_score": True,
    }

    parameters_intersection = {
        "include_detection_probability": False,
        "include_classification_score": False,
    }

    parameters_probability = {
        "include_detection_probability": True,
        "include_classification_score": False,
    }

    parameters_classification = {
        "include_detection_probability": False,
        "include_classification_score": True,
    }

    # create operators
    normal_op = lambda model, inputs, targets: \
            object_detection_operator(model, inputs, targets, **parameters_normal)

    intersection_op = lambda model, inputs, targets: \
            object_detection_operator(model, inputs, targets, **parameters_intersection)

    probability_op = lambda model, inputs, targets: \
            object_detection_operator(model, inputs, targets, **parameters_probability)

    classification_op = lambda model, inputs, targets: \
            object_detection_operator(model, inputs, targets, **parameters_classification)

    # compute explanations
    phis = Occlusion(model, operator=object_detection_operator, patch_size=4, patch_stride=2)(x, obj_ref)

    normal_phis = Occlusion(model, operator=normal_op, patch_size=4, patch_stride=2)(x, obj_ref)

    intersection_phis = Occlusion(model, operator=intersection_op, patch_size=4, patch_stride=2)(x, obj_ref)

    probability_phis = Occlusion(model, operator=probability_op, patch_size=4, patch_stride=2)(x, obj_ref)

    classification_phis = Occlusion(model, operator=classification_op, patch_size=4, patch_stride=2)(x, obj_ref)

    for phi in [phis, normal_phis, intersection_phis, probability_phis, classification_phis]:
        assert phi.shape == (obj_ref.shape[0], input_shape[0], input_shape[1], 1)
    
    assert almost_equal(phis, normal_phis)

    for phi1, phi2 in combinations([normal_phis, intersection_phis, probability_phis, classification_phis], 2):
        assert not almost_equal(phi1, phi2)