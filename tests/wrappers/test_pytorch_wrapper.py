import numpy as np

import tensorflow as tf
import torch.nn as nn

import pytest

from xplique.wrappers import TorchWrapper
from xplique.attributions import (Saliency, GradientInput, IntegratedGradients, SmoothGrad, VarGrad,
                                  SquareGrad, Occlusion, Rise, SobolAttributionMethod, Lime, KernelShap,
                                  HsicAttributionMethod)
from xplique.metrics import MuFidelity, Deletion, Insertion, AverageStability
from xplique.metrics.base import ExplainerMetric

from ..utils import generate_data

def map_four_by_four(inp):

    width = inp.shape[0]
    height = inp.shape[1]

    mapping = np.zeros((width,height))
    for i in range(width):
        if i%2 != 0:
            mapping[i] = mapping[i-1]
        else:
            for j in range(height):
                mapping[i][j] = (width/2) * (i//2) + (j//2)

    mapping = tf.cast(mapping, dtype=tf.int32)
    return mapping

def _default_methods_cnn(model):
    return [
        Saliency(model),
        GradientInput(model),
        IntegratedGradients(model),
        SmoothGrad(model),
        SquareGrad(model),
        VarGrad(model),
        Occlusion(model),
        Rise(model),
        SobolAttributionMethod(model),
        HsicAttributionMethod(model),
        Lime(model, nb_samples = 20, map_to_interpret_space=map_four_by_four),
        KernelShap(model, nb_samples = 20, map_to_interpret_space=map_four_by_four),
    ]

def _default_methods_regression(model):
    return [
        Saliency(model, operator="regression"),
        GradientInput(model, operator="regression"),
        IntegratedGradients(model, operator="regression"),
        SmoothGrad(model, operator="regression"),
        SquareGrad(model, operator="regression"),
        VarGrad(model, operator="regression"),
        Occlusion(model, operator="regression", patch_size=1, patch_stride=1),
        Lime(model, operator="regression", nb_samples = 20),
        KernelShap(model, operator="regression", nb_samples = 20),
    ]

def generate_torch_model(input_shape=(32, 32, 3), output_shape=10):
    c_in = input_shape[-1]
    h_in = input_shape[0]
    w_in = input_shape[1]

    model = nn.Sequential()

    model.append(nn.Conv2d(c_in, 4, (2, 2)))
    h_out = h_in - 1
    w_out = w_in -1
    c_out = 4

    model.append(nn.ReLU())
    model.append(nn.MaxPool2d((2, 2)))
    h_out = int((h_out - 2)/2 + 1)
    w_out = int((w_out - 2)/2 + 1)

    model.append(nn.Dropout(0.25))
    model.append(nn.Flatten())
    flatten_size = c_out * h_out * w_out

    model.append(nn.Linear(int(flatten_size) ,output_shape))

    return model

def generate_regression_torch(features_shape, output_shape=1):
    in_size = np.prod(features_shape)
    model = nn.Sequential()
    model.append(nn.Flatten())
    model.append(nn.Linear(in_size, 4))
    model.append(nn.ReLU())
    model.append(nn.Linear(4, 4))
    model.append(nn.ReLU())
    model.append(nn.Linear(4, output_shape))

    return model

def test_assert_eval_mode():
    input_shapes = [(28, 28, 1), (32, 32, 3)]
    nb_labels = 10

    for input_shape in input_shapes:
        model = generate_torch_model(input_shape, nb_labels)
        with pytest.raises(AssertionError):
            wrapped_model = TorchWrapper(model, device='cpu')
        model.eval()
        wrapped_model = TorchWrapper(model, device='cpu')

def test_cnn_wrapper():
    input_shapes = [(28, 28, 1), (32, 32, 3)]
    nb_labels = 10

    for input_shape in input_shapes:
        x, y = generate_data(input_shape, nb_labels, 16)
        model = generate_torch_model(input_shape, nb_labels)
        model.eval()
        wrapped_model = TorchWrapper(model, device='cpu')

        explainers = _default_methods_cnn(wrapped_model)

        for explainer in explainers:
            saliency_maps = explainer.explain(x, y)
            assert x.shape[:3] == saliency_maps.shape[:3]

def test_dense_wrapper():

    features_shape, output_shape, samples = ((10,), 1, 16)
    model = generate_regression_torch(features_shape, output_shape)
    model.eval()
    x, y = generate_data(features_shape, output_shape, samples)
    wrapped_model = TorchWrapper(model, device='cpu', is_channel_first=False)

    explainers = _default_methods_regression(wrapped_model)

    for explainer in explainers:
        try:
            explanations = explainer.explain(x, y)
        except:
            raise AssertionError(
                "Explanation failed for method ", explainer.__class__.__name__)
        assert explanations.shape == [samples, *features_shape]

def test_metric_cnn():
    """Test that wrapped torch model can also leverage the metrics module"""

    input_shape, nb_labels, samples = ((16, 16, 3), 5, 8)
    x, y = generate_data(input_shape, nb_labels, samples)
    model = generate_torch_model(input_shape, nb_labels)
    model.eval()
    wrapped_model = TorchWrapper(model, device='cpu')

    explainers = _default_methods_cnn(wrapped_model)

    metrics = [
        Deletion(wrapped_model, x, y, steps=3),
        Insertion(wrapped_model, x, y, steps=3),
        MuFidelity(wrapped_model, x, y, nb_samples=3),
        AverageStability(wrapped_model, x, y, nb_samples=3)
    ]

    for explainer in explainers:
        explanations = explainer(x, y)
        for metric in metrics:
            assert hasattr(metric, 'evaluate')
            if isinstance(metric, ExplainerMetric):
                score = metric(explainer)
            else:
                assert hasattr(metric, 'inference_function')
                assert hasattr(metric, 'batch_inference_function')
                score = metric(explanations)
            assert type(score) in [np.float32, np.float64, float]

def test_metric_dense():
    """Test that wrapped torch model can also leverage the metrics module"""

    input_shape, nb_labels, samples = ((16, 16, 3), 5, 8)
    x, y = generate_data(input_shape, nb_labels, samples)
    model = generate_regression_torch(input_shape, nb_labels)
    model.eval()
    wrapped_model = TorchWrapper(model, device='cpu')

    explainers = _default_methods_regression(wrapped_model)

    metrics = [
        Deletion(wrapped_model, x, y, steps=3, activation="sigmoid"),
        Insertion(wrapped_model, x, y, steps=3, activation="softmax"),
        MuFidelity(wrapped_model, x, y, nb_samples=3),
        AverageStability(wrapped_model, x, y, nb_samples=3)
    ]

    for explainer in explainers:
        explanations = explainer(x, y)
        for metric in metrics:
            assert hasattr(metric, 'evaluate')
            if isinstance(metric, ExplainerMetric):
                score = metric(explainer)
            else:
                assert hasattr(metric, 'inference_function')
                assert hasattr(metric, 'batch_inference_function')
                score = metric(explanations)
            assert type(score) in [np.float32, np.float64, float]
