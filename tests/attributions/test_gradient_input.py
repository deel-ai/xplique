from xplique.attributions import GradientInput
from ..utils import generate_data, generate_model


def test_output_shape():
    """The output shape must be the same as the input shape"""

    input_shapes = [(28, 28, 1), (32, 32, 3)]
    nb_labels = 10

    for input_shape in input_shapes:
        x, y = generate_data(input_shape, nb_labels, 100)
        model = generate_model(input_shape, nb_labels)

        method = GradientInput(model, -2)
        outputs = method.explain(x, y)

        assert x.shape == outputs.shape
