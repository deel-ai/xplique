import numpy as np
import tensorflow.keras.backend as K

from xplique.attributions import GradCAM
from ..utils import generate_data, generate_model, almost_equal


def test_output_shape():
    """The output shape must be the same as the input shape"""

    input_shapes = [(28, 28, 1), (32, 32, 3)]
    nb_labels = 10

    for input_shape in input_shapes:
        x, y = generate_data(input_shape, nb_labels, 100)
        model = generate_model(input_shape, nb_labels)

        method = GradCAM(model, -2)
        outputs = method.explain(x, y)

        assert x.shape[:3] == outputs.shape[:3]


def test_conv_layer():
    """We should target the right layer using either int, string or default procedure"""
    K.clear_session()

    model = generate_model()

    last_conv_layer = model.get_layer('conv2d_1')
    first_conv_layer = model.get_layer('conv2d')
    flatten_layer = model.get_layer('flatten')

    # default should target the last conv layer
    gc_default = GradCAM(model)
    assert gc_default.conv_layer == last_conv_layer

    # target the first conv layer
    gc_input_conv = GradCAM(model, conv_layer=0)
    assert gc_input_conv.conv_layer == first_conv_layer

    # target a random flatten layer
    gc_flatten = GradCAM(model, conv_layer='flatten')
    assert gc_flatten.conv_layer == flatten_layer


def test_weights_computation():
    """Ensure the grad-cam weights are correct"""
    activations = np.array([
        [[1.0, 1.0],
         [1.0, 1.0]],

        [[1.0, 1.0],
         [0.0, 0.0]],

        [[1.0, 0.5],
         [0.0, 0.0]],

        [[0.5, 0.0],
         [0.0, 0.0]],
    ])[None, :, :, :]
    grads = np.array([
        [[1.0, 1.0],
         [1.0, 1.0]],

        [[1.0, 1.0],
         [0.0, 0.0]],

        [[1.0, 0.5],
         [0.0, 0.0]],

        [[0.5, 0.0],
         [0.0, 0.0]],
    ])[None, :, :, :]

    # move so that the filters F are at the end [F, W, H] -> [W, H, F]
    activations = np.moveaxis(activations, 1, 3)
    grads = np.moveaxis(grads, 1, 3)

    weights = GradCAM.compute_weights(grads, activations)
    assert almost_equal(weights[0], [4.0/4.0, 2.0/4.0, 1.5/4.0, 0.5/4.0])

    grad_cam = GradCAM.apply_weights(weights, activations)
    assert almost_equal(grad_cam, np.sum(activations * weights, -1)) # as we have no negative value
