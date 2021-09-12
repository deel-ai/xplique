import numpy as np

from xplique.attributions import (Saliency, GradientInput, IntegratedGradients, SmoothGrad, VarGrad,
                                  SquareGrad, GradCAM, Occlusion, Rise, GuidedBackprop, DeconvNet,
                                  GradCAMPP)
from xplique.metrics import MuFidelity, Deletion, Insertion, AverageStability
from xplique.metrics.base import ExplanationMetric, ExplainerMetric
from ..utils import generate_data, generate_model


def _default_methods(model, output_layer_index=-2):
    return [
        Saliency(model, output_layer_index),
        GradientInput(model, output_layer_index),
        SmoothGrad(model, output_layer_index, nb_samples=5),
        VarGrad(model, output_layer_index),
        SquareGrad(model, output_layer_index),
        IntegratedGradients(model, output_layer_index, steps=5),
        GradCAM(model, output_layer_index),
        Occlusion(model, patch_size=4, patch_stride=4),
        Rise(model, nb_samples=10),
        GuidedBackprop(model, output_layer_index),
        DeconvNet(model, output_layer_index),
        GradCAMPP(model, output_layer_index),
    ]


def test_common():
    """Test that all the attributions method works as explainer"""

    input_shape, nb_labels, samples = ((16, 16, 3), 10, 20)
    x, y = generate_data(input_shape, nb_labels, samples)
    model = generate_model(input_shape, nb_labels)

    explainers = _default_methods(model)

    metrics = [
        Deletion(model, x, y, steps=3),
        Insertion(model, x, y, steps=3),
        MuFidelity(model, x, y, nb_samples=3),
        AverageStability(model, x, y, nb_samples=3)
    ]

    for explainer in explainers:
        explanations = explainer(x, y)
        for metric in metrics:
            assert hasattr(metric, 'evaluate')
            if isinstance(metric, ExplainerMetric):
                score = metric(explainer)
            else:
                score = metric(explanations)
            print(f"\n\n\n {type(score)} \n\n\n")
            assert type(score) in [np.float32, np.float64, float]
