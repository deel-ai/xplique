import numpy as np

from xplique.attributions import IntegratedGradients
from ..utils import generate_data, generate_model, almost_equal


def test_output_shape():
    """The output shape must be the same as the input shape"""

    input_shapes = [(28, 28, 1), (32, 32, 3)]
    nb_labels = 10

    for input_shape in input_shapes:
        x, y = generate_data(input_shape, nb_labels, 10)
        model = generate_model(input_shape, nb_labels)

        method = IntegratedGradients(model, -2, steps=100)
        outputs = method.explain(x, y)

        assert x.shape == outputs.shape


def test_straighline_path():
    """The path generated should be the straighline from the baseline to the input"""

    input_shapes = [(28, 28, 1), (32, 32, 3)]
    steps = 10
    x_value = 0.9

    for baseline_value in [0.0, 0.5, 1.0]:
        for input_shape in input_shapes:
            x = np.ones(input_shape, dtype=np.float32) * x_value
            baseline = IntegratedGradients._get_baseline(x.shape, baseline_value)
            path = IntegratedGradients._get_interpolated_points(x[None, :], steps, baseline[None, :])

            true_points = np.linspace(baseline_value, x_value, steps, dtype=np.float32)
            true_points = true_points[:, None, None, None] * np.ones(input_shape, dtype=np.float32)

            for point, true_point in zip(np.array(path), true_points):
                assert point.shape == x.shape
                assert point.min() == point.max()
                assert almost_equal(point, true_point)


def test_trapezoidal_rule():
    """The integral approximation should use trapezoidal rule"""

    # int( f(x) ) ~ sum( (f(a) + f(b)) / 2 ) for all points b > a
    points = np.array([0.1, 0.4, 0.6, 0.8, 0.9, 1.0, 0.8])

    true_trapezoidal = np.trapz(points, dx=1.0 / (len(points)-1))
    ig_trapezoidal = IntegratedGradients._average_gradients(points[None, :, None])

    assert almost_equal(ig_trapezoidal, true_trapezoidal)
