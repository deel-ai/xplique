import numpy as np

from xplique.attributions import (Saliency, GradientInput, IntegratedGradients, SmoothGrad, VarGrad,
                                  SquareGrad, GradCAM, Occlusion, Rise, GuidedBackprop, DeconvNet,
                                  GradCAMPP)
from xplique.attributions.base import BlackBoxExplainer
from ..utils import generate_data, generate_model

def _default_methods(model, output_layer_index):
    return [
        Saliency(model, output_layer_index),
        GradientInput(model, output_layer_index),
        SmoothGrad(model, output_layer_index),
        VarGrad(model, output_layer_index),
        SquareGrad(model, output_layer_index),
        IntegratedGradients(model, output_layer_index),
        GradCAM(model, output_layer_index),
        Occlusion(model),
        Rise(model),
        GuidedBackprop(model, output_layer_index),
        DeconvNet(model, output_layer_index),
        GradCAMPP(model, output_layer_index),
    ]


def test_common():
    """Test applied to all the attributions"""

    input_shape, nb_labels, samples = ((32, 32, 3), 10, 20)
    x, y = generate_data(input_shape, nb_labels, samples)
    model = generate_model(input_shape, nb_labels)
    output_layer_index = -2

    methods = _default_methods(model, output_layer_index)

    for method in methods:
        explanations = method.explain(x, y)

        # all explanation must have an explain method
        assert hasattr(method, 'explain')

        # all explanations returned must be numpy array
        assert isinstance(explanations, np.ndarray)


def test_batch_size():
    """Ensure the functioning of attributions for special batch size cases"""

    input_shape, nb_labels, samples = ((10, 10, 3), 5, 20)
    x, y = generate_data(input_shape, nb_labels, samples)
    model = generate_model(input_shape, nb_labels)
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
            GradCAM(model, output_layer_index, bs),
            Occlusion(model, bs),
            Rise(model, bs),
            GuidedBackprop(model, output_layer_index, bs),
            DeconvNet(model, output_layer_index, bs),
            GradCAMPP(model, output_layer_index, bs),
        ]

        for method in methods:
            try:
                explanations = method.explain(x, y)
            except:
                raise AssertionError(
                    "Explanation failed for method ", method.__class__.__name__,
                    " batch size ", bs)


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
