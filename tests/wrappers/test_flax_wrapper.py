from functools import partial

import numpy as np

import flax.linen as nn
import jax
import jax.numpy as jnp
import tensorflow as tf

import pytest

from xplique.wrappers import FlaxWrapper
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

def generate_flax_model(input_shape=(32, 32, 3), output_shape=10):

    class SimpleCNN(nn.Module):
        
        @nn.compact
        def __call__(self, x):
            x = nn.Conv(features=4, kernel_size=(2, 2))(x)
            x = nn.relu(x)
            x = nn.max_pool(x, window_shape=(2, 2))
            # deterministic dropout to avoid passing an RNG during forward pass.
            x = nn.Dropout(0.25, deterministic=True)(x)
            x = jax.vmap(jnp.ravel)(x)  # flatten in parallel
            x = nn.Dense(output_shape)(x)
            return x
        
    params_key = jax.random.PRNGKey(seed=0)

    x = jnp.ones((1, *input_shape))
    params = SimpleCNN().init(params_key, x)

    return SimpleCNN(), params

def generate_regression_flax(features_shape, output_shape=1):

    class SimpleMLP(nn.Module):
        
        @nn.compact
        def __call__(self, x):
            x = jax.vmap(jnp.ravel)(x)
            x = nn.Dense(4)(x)
            x = nn.relu(x)
            x = nn.Dense(4)(x)
            x = nn.relu(x)
            x = nn.Dense(output_shape)(x)
            return x

    params_key = jax.random.PRNGKey(seed=0)

    x = jnp.ones((1, *features_shape))
    params = SimpleMLP().init(params_key, x)

    return SimpleMLP(), params

def test_cnn_wrapper():
    input_shapes = [(28, 28, 1), (32, 32, 3)]
    nb_labels = 10

    for input_shape in input_shapes:
        x, y = generate_data(input_shape, nb_labels, 16)
        model, params = generate_flax_model(input_shape, nb_labels)
        wrapped_model = FlaxWrapper(model, params)

        explainers = _default_methods_cnn(wrapped_model)

        for explainer in explainers:
            saliency_maps = explainer.explain(x, y)
            assert x.shape[:3] == saliency_maps.shape[:3]

def test_dense_wrapper():

    features_shape, output_shape, samples = ((10,), 1, 16)
    model, params = generate_regression_flax(features_shape, output_shape)
    x, y = generate_data(features_shape, output_shape, samples)
    wrapped_model = FlaxWrapper(model, params)

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
    model, params = generate_flax_model(input_shape, nb_labels)
    wrapped_model = FlaxWrapper(model, params)

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
    model, params = generate_regression_flax(input_shape, nb_labels)
    wrapped_model = FlaxWrapper(model, params)

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
