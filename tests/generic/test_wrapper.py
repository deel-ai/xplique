"""
Test for the wrapper
Also serve as a generic test
"""

import numpy as np
import tensorflow as tf

from xplique.attributions import KernelShap, Lime, Occlusion
from ..utils import generate_dataset, generate_linear_model, almost_equal


def test_wrapper():
    """
    Test the wrapping of a sklearn model, by comparing it to a keras model.
    Both are linear regression model with really close weights.
    We expect the explanations on those models to be close as well.
    """
    nb_samples = 100
    epsilon = 1.e-4 * nb_samples
    dataset, _, features_coef = generate_dataset(nb_samples)

    libraries = [
        "sklearn",
        "keras",
    ]

    models = {
        lib: generate_linear_model(features_coef, lib)
        for lib in libraries
    }

    # we want both models to be close enough
    y_preds = [np.array(model(dataset)).reshape((nb_samples,)) for model in models.values()]
    assert almost_equal(y_preds[0], y_preds[1], epsilon)

    # Define explainers
    model_explainers = {}
    for library, model in models.items():
        model_explainers[library] = {
            "KernelShap": KernelShap(
                model,
                nb_samples=200,  # 2000
                ref_value=0.0,
                batch_size=nb_samples
            ),
            "Lime": Lime(
                model,
                nb_samples=2000,  # 2000
                ref_value=0.0,
                distance_mode="cosine",
                kernel_width=1.0,
                batch_size=nb_samples
            ),
            "Occlusion": Occlusion(
                model,
                patch_size=1,
                patch_stride=1,
                occlusion_value=0.0,
                batch_size=nb_samples
            ),
        }

    # cast a subset to tf format
    inputs_tf = tf.cast(dataset, tf.float32)
    targets_tf = tf.ones((len(dataset), 1))

    explanations = {}

    # compute explanations for all methods
    for library, explainers in model_explainers.items():
        explanations[library] = {}
        for method, explainer in explainers.items():
            # Lime is not stable compared, a seed is needed
            tf.random.set_seed(0)
            # compute explanation for a given method
            explanations[library][method] = explainer(inputs_tf, targets_tf)

    for method in explanations[libraries[0]].keys():
        expl0 = explanations[libraries[0]][method]
        expl1 = explanations[libraries[1]][method]
        assert almost_equal(expl0, expl1, epsilon)
