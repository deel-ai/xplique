import numpy as np
import tensorflow as tf

from xplique.attributions import (Saliency, GradientInput, IntegratedGradients, SmoothGrad, VarGrad,
                                  SquareGrad, GradCAM, Occlusion, Rise, GuidedBackprop, DeconvNet,
                                  GradCAMPP, Lime, KernelShap, SobolAttributionMethod,
                                  HsicAttributionMethod)
from xplique.attributions.base import BlackBoxExplainer
from ..utils import generate_data, generate_model, generate_regression_model, almost_equal


def _default_methods(model, output_layer_index=None, bs=32):
    return [
        Saliency(model, output_layer_index, bs),
        GradientInput(model, output_layer_index, bs),
        SmoothGrad(model, output_layer_index, bs, nb_samples=2),
        VarGrad(model, output_layer_index, bs, nb_samples=2),
        SquareGrad(model, output_layer_index, bs, nb_samples=2),
        IntegratedGradients(model, output_layer_index, bs, steps=2),
        GradCAM(model, output_layer_index, bs),
        Occlusion(model, bs, patch_size=10, patch_stride=10),
        Rise(model, bs, nb_samples=2),
        GuidedBackprop(model, output_layer_index, bs),
        DeconvNet(model, output_layer_index, bs),
        GradCAMPP(model, output_layer_index, bs),
        Lime(model, bs, nb_samples=2),
        KernelShap(model, bs, nb_samples=2),
        SobolAttributionMethod(model, grid_size=2, nb_design=2),
        HsicAttributionMethod(model, grid_size=2, nb_design=2),
    ]


def test_common():
    """Test applied to all the attributions"""

    input_shape, nb_labels, samples = ((32, 32, 3), 10, 20)
    model = generate_model(input_shape, nb_labels)
    output_layer_index = -2

    inputs_np, targets_np = generate_data(input_shape, nb_labels, samples)
    inputs_tf, targets_tf = tf.cast(inputs_np, tf.float32), tf.cast(targets_np, tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((inputs_np, targets_np))
    batched_dataset = tf.data.Dataset.from_tensor_slices((inputs_np, targets_np)).batch(3)

    methods = _default_methods(model, output_layer_index)

    for inputs, targets in [(inputs_np, targets_np),
                            (inputs_tf, targets_tf),
                            (dataset, None),
                            (batched_dataset, None)]:
        for method in methods:
            explanations = method.explain(inputs, targets)

            # all explanation must have an explain method
            assert hasattr(method, 'explain')

            # all explanations returned must be either a tf.Tensor or ndarray
            assert isinstance(explanations, (tf.Tensor, np.ndarray))

            # we should have one explanation for each inputs
            assert len(explanations) == len(inputs_np)


def test_batch_size():
    """Ensure the functioning of attributions for special batch size cases"""

    input_shape, nb_labels, samples = ((10, 10, 3), 5, 20)
    inputs, targets = generate_data(input_shape, nb_labels, samples)
    model = generate_model(input_shape, nb_labels)
    output_layer_index = -1

    batch_sizes = [None, 1, 32]

    for bs in batch_sizes:

        methods = _default_methods(model, output_layer_index, bs)

        for method in methods:
            explanations = method.explain(inputs, targets)

            # all explanations returned must be either a tf.Tensor or ndarray
            assert isinstance(explanations, (tf.Tensor, np.ndarray))

            # we should have one explanation for each inputs
            assert len(explanations) == len(inputs)


def test_model_caching():
    """Test the caching engine, used to avoid re-tracing"""

    model = generate_model()
    output_layer_index = -1

    # the key used for caching is the following tuple
    cache_key = (id(model.input), id(model.output))

    cache_len_before = len(BlackBoxExplainer._cache_models.keys())  # pylint:
    # disable=protected-access

    assert (cache_key not in BlackBoxExplainer._cache_models)  # pylint: disable=protected-access

    _ = _default_methods(model, output_layer_index)

    # check that the key is now in the cache
    assert (cache_key in BlackBoxExplainer._cache_models)  # pylint: disable=protected-access

    # ensure that there no more than one key has been added
    assert (len(
        BlackBoxExplainer._cache_models) == cache_len_before + 1)  # pylint: disable=protected-access


def test_data_types_shapes():
    """Test that methods support different inputs shapes"""

    data_types_input_shapes = {
        "tabular": (20,),
        "time-series": (20, 10),
        "images rgb": (20, 16, 3),
        "images black and white": (28, 28, 1),
    }

    not_compatible_methods = {
        "tabular": ["GradCAM", "GradCAMPP", "SobolAttributionMethod", "HsicAttributionMethod"],
        "time-series": ["GradCAM", "GradCAMPP", "SobolAttributionMethod", "HsicAttributionMethod"],
        "images rgb": [],
        "images black and white": [],
    }

    methods = {
        Saliency: {},
        GradientInput: {},
        SmoothGrad: {"nb_samples": 2},
        VarGrad: {"nb_samples": 2},
        SquareGrad: {"nb_samples": 2},
        IntegratedGradients: {"steps": 2},
        GuidedBackprop: {},
        DeconvNet: {},
        GradCAM: {},
        GradCAMPP: {},
        Occlusion: {},
        Rise: {"nb_samples": 2},
        Lime: {"nb_samples": 2},
        KernelShap: {"nb_samples": 2},
        SobolAttributionMethod: {"grid_size": 2, "nb_design": 2},
        HsicAttributionMethod: {"grid_size": 2, "nb_design": 2},
    }

    for data_type, input_shape in data_types_input_shapes.items():
        input_shape, nb_labels, samples = (input_shape, 5, 15)
        inputs, targets = generate_data(input_shape, nb_labels, samples)

        if len(input_shape) == 3:  # image => conv2D
            model = generate_model(input_shape, nb_labels)
            model.layers[-1].activation = tf.keras.activations.linear
        else:  # others => dense
            model = generate_regression_model(input_shape, nb_labels)

        for method, params in methods.items():
            if method.__name__ in not_compatible_methods[data_type]:
                continue

            explainer = method(model, **params)

            explanation = explainer(inputs, targets)

            if len(input_shape) == 3:  # image => explanation (n, h, w, 1)
                assert almost_equal(
                    np.array(explanation.shape),
                    np.array(inputs.shape[:-1] + (1,)))
            else:
                assert almost_equal(np.array(explanation.shape), np.array(inputs.shape))
