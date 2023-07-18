import numpy as np

from xplique.commons import forgrad

from ..utils import generate_data, generate_model
from .test_common import _default_methods


def test_base_forgrad():
    """ Ensure forgrad is working with the expected shape """
    shapes = [ (5, 32, 32, 3), (5, 32, 32, 1), (5, 32, 32), (5, 31, 31), (5, 60, 60) ]
    sigmas = [5, 10, 30]

    for shape in shapes:
        heatmaps = np.random.rand(*shape).astype(np.float32)
        for sigma in sigmas:
            filtered_heatmaps = forgrad(heatmaps, sigma)
            assert filtered_heatmaps.shape[:3] == heatmaps.shape[:3]


def test_integration_forgrad():
    """ Test that forgrad integrate with all attributions """
    input_shape, nb_labels, samples = ((32, 32, 3), 10, 20)
    model = generate_model(input_shape, nb_labels)
    output_layer_index = -2

    inputs_np, targets_np = generate_data(input_shape, nb_labels, samples)
    methods = _default_methods(model, output_layer_index)

    for method in methods:
        explanations = method.explain(inputs_np, targets_np)
        explanations_filtered = forgrad(explanations)

        assert explanations_filtered.shape[:3] == explanations.shape[:3]


