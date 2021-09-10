import numpy as np
import tensorflow as tf

from xplique.attributions import (Saliency, GradientInput, IntegratedGradients, SmoothGrad, VarGrad,
                                  SquareGrad, Occlusion, Rise, GuidedBackprop, DeconvNet, Lime,
                                  KernelShap)
from ..utils import generate_regression_model, generate_data

def _default_methods(model, output_layer_index):
    return [
        Saliency(model, output_layer_index),
        GradientInput(model, output_layer_index),
        SmoothGrad(model, output_layer_index),
        VarGrad(model, output_layer_index),
        SquareGrad(model, output_layer_index),
        IntegratedGradients(model, output_layer_index),
        GuidedBackprop(model, output_layer_index),
        DeconvNet(model, output_layer_index),
        Lime(model),
        KernelShap(model),
        Occlusion(model, patch_size=1, patch_stride=1),
    ]

def test_tabular_data():
    """Test applied to most attributions method"""

    features_shape, output_shape, samples = ((10,), 1, 20)
    model = generate_regression_model(features_shape, output_shape)
    output_layer_index = -1

    inputs_np, targets_np = generate_data(features_shape, output_shape, samples)
    inputs_tf, targets_tf = tf.cast(inputs_np, tf.float32), tf.cast(targets_np, tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((inputs_np, targets_np))
    # batched_dataset = tf.data.Dataset.from_tensor_slices((inputs_np, targets_np)).batch(4)

    methods = _default_methods(model, output_layer_index)

    for inputs, targets in [(inputs_np, targets_np),
                            (inputs_tf, targets_tf),
                            (dataset, None),
                            # (batched_dataset, None)
                            ]:
        for method in methods:
            try:
                explanations = method.explain(inputs, targets)
            except:
                raise AssertionError(
                    "Explanation failed for method ", method.__class__.__name__)

            # all explanation must have an explain method
            assert hasattr(method, 'explain')

            # all explanations returned must be numpy array
            assert isinstance(explanations, tf.Tensor)

            # all explanations shape should match features shape
            assert explanations.shape == [samples, *features_shape]

def test_multioutput_regression():
    """
    Test if in a multioutput regression settings the methods have the expected behavior
    """

    features_shape, output_shape, samples = ((10,), 4, 20)
    model = generate_regression_model(features_shape, output_shape=output_shape)
    output_layer_index = -1

    inputs_np, targets_np = generate_data(features_shape, output_shape, samples)
    inputs_tf, targets_tf = tf.cast(inputs_np, tf.float32), tf.cast(targets_np, tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((inputs_np, targets_np))
    # batched_dataset = tf.data.Dataset.from_tensor_slices((inputs_np, targets_np)).batch(4)

    methods = _default_methods(model, output_layer_index)

    for inputs, targets in [(inputs_np, targets_np),
                            (inputs_tf, targets_tf),
                            (dataset, None),
                            # (batched_dataset, None)
                            ]:
        for method in methods:
            try:
                explanations = method.explain(inputs, targets)
            except:
                raise AssertionError(
                    "Explanation failed for method ", method.__class__.__name__)

            # all explanation must have an explain method
            assert hasattr(method, 'explain')

            # all explanations returned must be numpy array
            assert isinstance(explanations, tf.Tensor)

            # all explanations shape should match features shape
            assert explanations.shape == [samples, *features_shape]

def test_batch_size():
    """
    Ensure the functioning of attributions for special batch size cases with tabular data
    """

    input_shape, nb_targets, samples = ((10,), 5, 20)
    inputs, targets = generate_data(input_shape, nb_targets, samples)
    model = generate_regression_model(input_shape, nb_targets)
    output_layer_index = -1

    batch_sizes = [None, 1, 32]

    for bs in batch_sizes:

        methods = [
            Saliency(model, output_layer_index, bs),
            GradientInput(model, output_layer_index, bs),
            SmoothGrad(model, output_layer_index, bs),
            VarGrad(model, output_layer_index, bs),
            SquareGrad(model, output_layer_index, bs),
            IntegratedGradients(model, output_layer_index, bs),
            GuidedBackprop(model, output_layer_index, bs),
            DeconvNet(model, output_layer_index, bs),
            Lime(model, bs),
            KernelShap(model, bs),
            Occlusion(model, bs, patch_size=1, patch_stride=1),
        ]

        for method in methods:
            try:
                explanations = method.explain(inputs, targets)
            except:
                raise AssertionError(
                    "Explanation failed for method ", method.__class__.__name__,
                    " batch size ", bs)
