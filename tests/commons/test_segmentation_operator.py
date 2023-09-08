"""
Ensure we can use the operator functionality on various models
"""

import numpy as np
import tensorflow as tf

from xplique.attributions import (Saliency, GradientInput, IntegratedGradients, SmoothGrad, VarGrad,
                                  SquareGrad, GradCAM, Occlusion, Rise, GuidedBackprop, DeconvNet,
                                  GradCAMPP, Lime, KernelShap, SobolAttributionMethod,
                                  HsicAttributionMethod)
from xplique.commons.operators_operations import semantic_segmentation_operator
from ..utils import generate_data, almost_equal


def default_methods(model, operator):
    return [
        Saliency(model, operator=operator),
        GradientInput(model, operator=operator),
        SmoothGrad(model, operator=operator),
        VarGrad(model, operator=operator),
        SquareGrad(model, operator=operator),
        IntegratedGradients(model, operator=operator),
        Occlusion(model, operator=operator),
        Rise(model, operator=operator, nb_samples=2),
        GuidedBackprop(model, operator=operator),
        DeconvNet(model, operator=operator),
        SobolAttributionMethod(model, operator=operator, grid_size=2, nb_design=2),
        HsicAttributionMethod(model, operator=operator, grid_size=2, nb_design=2),
    ]


def get_segmentation_model(input_shape=(20, 20, 1)):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(input_shape),
    ])
    model.compile()
    return model


def test_segmentation_operator():
    input_shape = (20, 20, 3)
    segmentation_model = get_segmentation_model(input_shape)

    x, _ = generate_data(input_shape, 10, 10)
    y, _ = generate_data(input_shape, 10, 10)

    methods = default_methods(segmentation_model, semantic_segmentation_operator)
    for method in methods:

        assert hasattr(method, 'inference_function')
        assert hasattr(method, 'batch_inference_function')
        assert hasattr(method, 'gradient')
        assert hasattr(method, 'batch_gradient')

        phis = method(x, y)

        assert x.shape[:-1] == phis.shape[:3]


def test_segmentation_operator_computation():
    image = [[[[0, 0, 1.0],
               [0, 0, 1.0],
               [1.0, 1.0, 1.0],],
              [[0, 0, 0],
               [0, 1.0, 0],
               [1.0, 1.0, 1.0],],
              [[0, 0, 0],
               [0, 0, 0],
               [1.0, 0, 0],],]]
    image = tf.transpose(tf.convert_to_tensor(image, tf.float32), perm=[0, 2, 3, 1])
    
    model = lambda x: tf.concat([x, tf.expand_dims(tf.reduce_mean(x, axis=-1), axis=-1)], axis=-1)

    target_1 = [[[[0, 0, 1.0],
                  [0, 0, 1.0],
                  [1.0, 1.0, 1.0],],
                 [[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],],
                 [[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],],
                 [[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],],]]
    target_1 = tf.transpose(tf.convert_to_tensor(target_1, tf.float32), perm=[0, 2, 3, 1])

    target_2 = [[[[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],],
                 [[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],],
                 [[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],],
                 [[0, 0, 0],
                  [0, 0, 0],
                  [1.0, 1.0, 1.0],],]]
    target_2 = tf.transpose(tf.convert_to_tensor(target_2, tf.float32), perm=[0, 2, 3, 1])

    scores = model(np.array(image)) * target_2

    score_1 = semantic_segmentation_operator(model, image, np.array(target_1))
    score_2 = semantic_segmentation_operator(model, np.array(image), target_2)
    
    assert almost_equal(score_1, 1.0)
    assert almost_equal(score_2, (7.0 / 3) / 3)
