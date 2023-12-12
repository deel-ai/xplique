import numpy as np
import tensorflow as tf

from xplique.attributions import (Saliency, GradientInput, IntegratedGradients, SmoothGrad, VarGrad,
                                  SquareGrad, GradCAM, Occlusion, Rise, GuidedBackprop, DeconvNet,
                                  GradCAMPP)
from xplique.metrics import MuFidelity, Deletion, Insertion, AverageStability
from xplique.metrics.base import ExplanationMetric, ExplainerMetric
from ..utils import generate_data, generate_model, generate_regression_model


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
    model = generate_model(input_shape, nb_labels)
    output_layer_index = -2

    inputs_np, targets_np = generate_data(input_shape, nb_labels, samples)
    inputs_tf, targets_tf = tf.cast(inputs_np, tf.float32), tf.cast(targets_np, tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((inputs_np, targets_np))
    batched_dataset = tf.data.Dataset.from_tensor_slices((inputs_np, targets_np)).batch(3)

    explainers = _default_methods(model, output_layer_index)

    for inputs, targets in [(inputs_np, targets_np),
                            (inputs_tf, targets_tf),
                            (dataset, None),
                            (batched_dataset, None)]:

        metrics = [
            Deletion(model, inputs, targets, steps=3),
            Insertion(model, inputs, targets, steps=3),
            MuFidelity(model, inputs, targets, nb_samples=3),
            AverageStability(model, inputs, targets, nb_samples=3)
        ]

        for explainer in explainers:
            explanations = explainer(inputs, targets)
            for metric in metrics:
                assert hasattr(metric, 'evaluate')
                if isinstance(metric, ExplainerMetric):
                    score = metric(explainer)
                else:
                    assert hasattr(metric, 'inference_function')
                    assert hasattr(metric, 'batch_inference_function')
                    score = metric(explanations)
                assert type(score) in [np.float32, np.float64, float]


def test_add_activation():
    """Test that adding a softmax or sigmoid layer still works"""
    input_shape, nb_labels, samples = ((16, 16, 3), 5, 8)
    x, y = generate_data(input_shape, nb_labels, samples)
    model = generate_model(input_shape, nb_labels)

    explainers = _default_methods(model)

    activations = ["sigmoid", "softmax"]

    for explainer in explainers:
        explanations = explainer(x, y)
        for activation in activations:
            metrics = [
                Deletion(model, x, y, steps=3, activation=activation),
                Insertion(model, x, y, steps=3, activation=activation),
                MuFidelity(model, x, y, nb_samples=3, activation=activation),
            ]
            for metric in metrics:
                assert hasattr(metric, 'evaluate')
                if isinstance(metric, ExplainerMetric):
                    score = metric(explainer)
                else:
                    assert hasattr(metric, 'inference_function')
                    assert hasattr(metric, 'batch_inference_function')
                    score = metric(explanations)
                    if not isinstance(metric, MuFidelity):
                        assert np.all(score <= 1)
                        assert np.all(score >= 0)
                assert type(score) in [np.float32, np.float64, float]

def test_data_types_shapes():
    """Test that all the attributions method works as explainer"""

    data_types_input_shapes = {
        "tabular": (20,),
        "time-series": (20, 10),
        "images rgb": (20, 16, 3),
        "images black and white": (28, 28, 1),
    }

    explainer_metrics = {
        AverageStability: {"nb_samples": 3}
    }
    explanation_metrics = {
        Deletion: {"steps": 3},
        Insertion: {"steps": 3},
        MuFidelity: {"nb_samples": 3, "grid_size": None, "subset_percent": 0.9},
    }

    for data_type, input_shape in data_types_input_shapes.items():
        input_shape, nb_labels, samples = (input_shape, 5, 15)
        inputs, targets = generate_data(input_shape, nb_labels, samples)

        if len(input_shape) == 3:  # image => conv2D
            model = generate_model(input_shape, nb_labels)
            model.layers[-1].activation = tf.keras.activations.linear
        else:  # others => dense
            model = generate_regression_model(input_shape, nb_labels)

        explainer = Saliency(model)
        explanation = explainer(inputs, targets)

        for metric_class, params in explanation_metrics.items():
            metric = metric_class(model, inputs, targets, **params)
            score = metric(explanation)
            assert type(score) in [np.float32, np.float64, float]

        for metric_class, params in explainer_metrics.items():
            metric = metric_class(model, inputs, targets, **params)
            score = metric(explainer)
            assert type(score) in [np.float32, np.float64, float]
