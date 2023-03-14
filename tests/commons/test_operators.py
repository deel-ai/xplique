"""
Ensure we can use the operator functionnality on various models
"""

import numpy as np
import tensorflow as tf

from xplique.attributions import (Saliency, GradientInput, IntegratedGradients, SmoothGrad, VarGrad,
                                  SquareGrad, GradCAM, Occlusion, Rise, GuidedBackprop, DeconvNet,
                                  GradCAMPP, Lime, KernelShap, SobolAttributionMethod,
                                  HsicAttributionMethod)
from xplique.commons.operators import (check_operator, predictions_operator, regression_operator,
                                       binary_segmentation_operator, segmentation_operator)
from xplique.commons.exceptions import InvalidOperatorException
from ..utils import generate_data, generate_regression_model


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


def get_segmentation_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input((20, 20, 1)),
    ])
    model.compile()
    return model


def get_concept_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input((6)),
        tf.keras.layers.Dense((10))
    ])
    model.compile()
    return model


def test_check_operator():
    # ensure that the check operator detects non-operator

    # operator must have at least 3 arguments
    function_with_2_arguments = lambda x,y: 0

    # operator must be Callable
    not_a_function = [1, 2, 3]

    for operator in [function_with_2_arguments, not_a_function]:
        try:
            check_operator(operator)
            assert False
        except InvalidOperatorException:
            pass


def test_proposed_operators():
    # ensure all proposed operators are operators
    for operator in [predictions_operator, regression_operator,
                     binary_segmentation_operator, segmentation_operator]:
        check_operator(operator)


def test_regression_operator():
    input_shape, nb_labels, samples = ((10, 10, 1), 10, 20)
    x, y = generate_data(input_shape, nb_labels, samples)
    regression_model = generate_regression_model(input_shape, nb_labels)

    methods = default_methods(regression_model, regression_operator)
    for method in methods:

        assert hasattr(method, 'inference_function')
        assert hasattr(method, 'batch_inference_function')
        assert hasattr(method, 'gradient')
        assert hasattr(method, 'batch_gradient')

        phis = method(x, y)

        assert x.shape[:-1] == phis.shape[:3]
    

def test_segmentation_operator():
    segmentation_model = get_segmentation_model()

    x, y = generate_data((20, 20, 3), 10, 10)

    def segmentation_operator(model, x, y):
        # explaining channel 0
        return tf.reduce_sum(model(x)[:,:,0], (1, 2))

    methods = default_methods(segmentation_model, segmentation_operator)
    for method in methods:

        assert hasattr(method, 'inference_function')
        assert hasattr(method, 'batch_inference_function')
        assert hasattr(method, 'gradient')
        assert hasattr(method, 'batch_gradient')

        phis = method(x, y)

        assert x.shape[:-1] == phis.shape[:3]


def test_concept_operator():
    concept_model = get_concept_model()

    x, y = generate_data((20, 20, 1), 10, 10)

    random_projection = tf.cast(np.random.uniform(size=(20*20, 6)), tf.float32)

    def concept_operator(model, x, y):
        x = tf.reshape(x, (-1, 20*20))
        print(x.shape, random_projection.shape)
        ui = x @ random_projection
        return tf.reduce_sum(model(ui) * y, axis=-1)

    methods = default_methods(concept_model, concept_operator)

    for method in methods:

        assert hasattr(method, 'inference_function')
        assert hasattr(method, 'batch_inference_function')
        assert hasattr(method, 'gradient')
        assert hasattr(method, 'batch_gradient')

        phis = method(x, y)

        assert x.shape[:-1] == phis.shape[:3]
