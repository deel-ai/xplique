import numpy as np

from xplique.methods import (Saliency, GradientInput, IntegratedGradients, SmoothGrad, GradCAM,
                             Occlusion, GuidedBackprop)
from xplique.methods.base import BaseExplanation
from ..utils import generate_data, generate_model


def test_common():
    """Test applied to all the methods"""

    input_shape, nb_labels, samples = ((32, 32, 3), 10, 100)
    x, y = generate_data(input_shape, nb_labels, samples)
    model = generate_model(input_shape, nb_labels)
    output_layer_index = -2

    methods = [
        Saliency(model, output_layer_index),
        GradientInput(model, output_layer_index),
        SmoothGrad(model, output_layer_index),
        IntegratedGradients(model, output_layer_index),
        GradCAM(model, output_layer_index),
        Occlusion(model, output_layer_index),
        GuidedBackprop(model, output_layer_index)
    ]

    for method in methods:
        explanations = method.explain(x, y)

        # all explanation must have an explain and single_batch method
        assert hasattr(method, 'explain')
        assert hasattr(method, 'compute')

        # all explanations returned must be numpy array
        assert isinstance(explanations, np.ndarray)


def test_batch_size():
    """Ensure the functioning of methods for special batch size cases"""

    input_shape, nb_labels, samples = ((10, 10, 3), 5, 50)
    x, y = generate_data(input_shape, nb_labels, samples)
    model = generate_model(input_shape, nb_labels)
    output_layer_index = -2

    batch_sizes = [None, 1, 32]

    for bs in batch_sizes:

        methods = [
            Saliency(model, output_layer_index, bs),
            GradientInput(model, output_layer_index, bs),
            SmoothGrad(model, output_layer_index, bs),
            IntegratedGradients(model, output_layer_index, bs),
            GradCAM(model, output_layer_index, bs),
            Occlusion(model, output_layer_index, bs),
            GuidedBackprop(model, output_layer_index, bs)
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
    output_layer_index = -2

    # the key used for caching is the following tuple
    cache_key = (id(model), output_layer_index)

    cache_len_before = len(BaseExplanation._cache_models.keys())  # pylint: disable=protected-access

    assert (cache_key not in BaseExplanation._cache_models)  # pylint: disable=protected-access

    _ = [
        Saliency(model, output_layer_index),
        GradientInput(model, output_layer_index),
        SmoothGrad(model, output_layer_index),
        IntegratedGradients(model, output_layer_index),
        GradCAM(model, output_layer_index),
        Occlusion(model, output_layer_index),
        GuidedBackprop(model, output_layer_index)
    ]

    # check that the key is now in the cache
    assert (cache_key in BaseExplanation._cache_models)  # pylint: disable=protected-access

    # ensure that there no more than one key has been added
    assert (len(
        BaseExplanation._cache_models) == cache_len_before + 1)  # pylint: disable=protected-access
