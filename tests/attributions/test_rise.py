from xplique.attributions import Rise
from ..utils import generate_data, generate_model


def test_output_shape():
    """The output size (h, w) must be the same as the input"""

    input_shapes = [(28, 28, 1), (32, 32, 3)]
    nb_labels = 10

    for input_shape in input_shapes:
        x, y = generate_data(input_shape, nb_labels, 10)
        model = generate_model(input_shape, nb_labels)

        method = Rise(model, nb_samples=100)
        rise_maps = method.explain(x, y)

        assert x.shape[:-1] == rise_maps.shape
