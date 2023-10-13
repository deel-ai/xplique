import numpy as np

from xplique.attributions import HsicAttributionMethod
from xplique.attributions.global_sensitivity_analysis import (
    SobolevEstimator,
    BinaryEstimator,
    RbfEstimator,
)
from ..utils import generate_data, generate_model, almost_equal


def test_hsic_kernels_shape():
    """
    Test if the kernels are correctly computed.
    """
    input_shapes = [(20, 20, 1), (20, 20, 3)]
    nb_labels = 10

    for input_shape in input_shapes:
        x, y = generate_data(input_shape, nb_labels, 10)
        model = generate_model(input_shape, nb_labels)

        method = HsicAttributionMethod(model, grid_size=2, nb_design=8)
        stis_maps = method.explain(x, y)

        assert x.shape[:-1] == stis_maps.shape[:-1]


def test_estimators():
    """Ensure every proposed estimator is working"""

    estimators = [SobolevEstimator(), BinaryEstimator(), RbfEstimator()]
    input_shapes = [(20, 20, 1), (20, 20, 3)]
    nb_labels = 10

    for input_shape in input_shapes:
        x, y = generate_data(input_shape, nb_labels, 10)
        model = generate_model(input_shape, nb_labels)

        for estimator in estimators:
            method = HsicAttributionMethod(
                model, grid_size=2, nb_design=8, estimator=estimator
            )
            stis_maps = method.explain(x, y)

            assert x.shape[:-1] == stis_maps.shape[:-1]

def test_estimator_batch_size():
    """Ensure estimator batch size is correctly used"""
    input_shape = (20, 20, 1)
    nb_labels = 10
    x, y = generate_data(input_shape, nb_labels, 10)
    model = generate_model(input_shape, nb_labels)

    for estimator_batch_size in [None, 1, 4, 7, 23, 37, 103]:
        method = HsicAttributionMethod(
            model, grid_size=5, nb_design=10, estimator_batch_size=estimator_batch_size
        )
        if estimator_batch_size is None:
            assert method.estimator.batch_size == 100000
        else:
            assert method.estimator.batch_size == estimator_batch_size
        
        stis_maps = method.explain(x, y)
        assert x.shape[:-1] == stis_maps.shape[:-1]
