from xplique.attributions import Saliency
from ..utils import generate_data, generate_model


def test_output_shape():
    """The output shape must be the same as the input shape, except for the channels"""

    input_shapes = [(28, 28, 1), (32, 32, 3)]
    nb_labels = 10

    for input_shape in input_shapes:
        x, y = generate_data(input_shape, nb_labels, 100)
        model = generate_model(input_shape, nb_labels)

        method = Saliency(model, -2)
        saliency_maps = method.explain(x, y)

        assert x.shape[:3] == saliency_maps.shape[:3]
