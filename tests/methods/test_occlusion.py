import numpy as np

from xplique.methods import Occlusion
from ..utils import generate_data, generate_model


def test_output_shape():
    """The output shape must be the same as the input shape, except for the channels"""

    input_shapes = [(28, 28, 1), (32, 32, 3)]
    nb_labels = 10

    for input_shape in input_shapes:
        x, y = generate_data(input_shape, nb_labels, 100)
        model = generate_model(input_shape, nb_labels)

        method = Occlusion(model, -2)
        sensitivity = method.explain(x, y)

        assert x.shape[:3] == sensitivity.shape[:3]


def test_polymorphic_parameters():
    """Ensure we could pass tuple or int to define patch parameters"""
    s = 3
    model = generate_model()

    occlusion_int = Occlusion(model, -2, patch_size=s, patch_stride=s)
    occlusion_tuple = Occlusion(model, -2, patch_size=(s, s), patch_stride=(s, s))

    assert occlusion_int.patch_size == occlusion_tuple.patch_size
    assert occlusion_int.patch_stride == occlusion_tuple.patch_stride


def test_mask_generator():
    """Ensure we generate all the possible masks"""

    assert Occlusion.get_masks((10, 10), (1, 1), (1, 1)).shape == (100, 10, 10)
    assert Occlusion.get_masks((10, 10), (2, 2), (2, 2)).shape == (25, 10, 10)
    assert Occlusion.get_masks((10, 10), (2, 2), (3, 3)).shape == (9, 10, 10)
    assert np.array_equal(Occlusion.get_masks((2, 2), (1, 1), (1, 1)), np.array([
        [
            [1, 0],
            [0, 0]
        ],
        [
            [0, 1],
            [0, 0]
        ],
        [
            [0, 0],
            [1, 0]
        ],
        [
            [0, 0],
            [0, 1]
        ],
    ], dtype=np.bool))
