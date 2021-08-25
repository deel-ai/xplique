import numpy as np

from xplique.attributions import Occlusion
from ..utils import generate_data, generate_model, almost_equal


def test_output_shape():
    """The output shape must be the same as the input shape, except for the channels"""

    input_shapes = [(28, 28, 1), (32, 32, 3)]
    nb_labels = 10

    for input_shape in input_shapes:
        x, y = generate_data(input_shape, nb_labels, 20)
        model = generate_model(input_shape, nb_labels)

        method = Occlusion(model)
        sensitivity = method.explain(x, y)

        assert x.shape[:3] == sensitivity.shape


def test_polymorphic_parameters():
    """Ensure we could pass tuple or int to define patch parameters when inputs are images"""
    s = 3

    input_shapes = [(28, 28, 1), (32, 32, 3)]
    nb_labels = 10

    for input_shape in input_shapes:
        features, targets = generate_data(input_shape, nb_labels, 20)
        model = generate_model(input_shape, nb_labels)

        occlusion_int = Occlusion(model, patch_size=s, patch_stride=s)
        occlusion_tuple = Occlusion(model, patch_size=(s, s), patch_stride=(s, s))

        occlusion_int(features, targets)
        occlusion_tuple(features, targets)

        assert occlusion_int.patch_size == occlusion_tuple.patch_size
        assert occlusion_int.patch_stride == occlusion_tuple.patch_stride


def test_mask_generator():
    """Ensure we generate all the possible masks"""

    assert Occlusion._get_masks((10, 10), (1, 1), (1, 1)).shape == (100, 10, 10)
    assert Occlusion._get_masks((10, 10), (2, 2), (2, 2)).shape == (25, 10, 10)
    assert Occlusion._get_masks((10, 10), (2, 2), (3, 3)).shape == (9, 10, 10)
    assert np.array_equal(Occlusion._get_masks((2, 2), (1, 1), (1, 1)), np.array([
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


def test_apply():
    """Ensure we apply correctly the masks"""
    x = np.ones((2, 2, 1), dtype=np.float32)
    applied_value = 49.0

    masks = Occlusion._get_masks((2, 2, 1), (1, 1), (1, 1))
    occluded_x = Occlusion._apply_masks(x, masks, applied_value)

    assert almost_equal(occluded_x[:,:,:,0], np.array([
        [[applied_value, 1.0],
         [1.0, 1.0]],
        [[1.0, applied_value],
         [1.0, 1.0]],
        [[1.0, 1.0],
         [applied_value, 1.0]],
        [[1.0, 1.0],
         [1.0, applied_value]],
    ]))

    masks = Occlusion._get_masks((2, 2, 1), (2, 2), (2, 2))
    occluded_x = Occlusion._apply_masks(x, masks, applied_value)

    assert almost_equal(occluded_x[:, :, :, 0], np.array([
        [[applied_value, applied_value],
         [applied_value, applied_value]]
    ]))


def test_delta_computation():
    """Ensure get correct delta score"""
    baseline_score = np.array([1.0], dtype=np.float32)
    masks = Occlusion._get_masks((2, 2, 1), (1, 1), (1, 1))
    scores = np.array([1.0, 0.0, 10.0, -1.0], dtype=np.float32)

    deltas = Occlusion._compute_sensitivity(baseline_score, scores, masks)
    assert almost_equal(deltas.numpy().flatten(), [0.0, 1.0, -9.0, 2.0])

    baseline_score = np.array([10.0], dtype=np.float32)
    masks = Occlusion._get_masks((2, 2, 1), (2, 2), (2, 2))
    scores = np.array([5.0], dtype=np.float32)

    deltas = Occlusion._compute_sensitivity(baseline_score, scores, masks)
    assert almost_equal(deltas, [5.0])
