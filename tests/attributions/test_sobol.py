import numpy as np

from xplique.attributions import SobolAttributionMethod
from xplique.attributions.global_sensitivity_analysis import (LatinHypercubeRS,
    ScipySobolSequenceRS, TFSobolSequenceRS, HaltonSequenceRS, JansenEstimator,
    JanonEstimator, GlenEstimator, SaltelliEstimator,HommaEstimator, inpainting,
    blurring, amplitude)
from ..utils import generate_data, generate_model, almost_equal


def test_output_shape():
    """The output size (h, w) must be the same as the input"""

    input_shapes = [(20, 20, 1), (20, 20, 3)]
    nb_labels = 10

    for input_shape in input_shapes:
        x, y = generate_data(input_shape, nb_labels, 10)
        model = generate_model(input_shape, nb_labels)

        method = SobolAttributionMethod(model,grid_size = 2, nb_design = 8)
        stis_maps = method.explain(x, y)

        assert x.shape[:-1] == stis_maps.shape[:-1]


def test_samplers():
    """Ensure every proposed sampler is working"""

    samplers_class = [LatinHypercubeRS, ScipySobolSequenceRS, TFSobolSequenceRS, HaltonSequenceRS]
    input_shapes = [(20, 20, 1), (20, 20, 3)]
    nb_labels = 10


    for input_shape in input_shapes:

        x, y = generate_data(input_shape, nb_labels, 10)
        model = generate_model(input_shape, nb_labels)

        for sampler_c in samplers_class:
            # (fel) some sampler(s) need scipy>=1.7, we should enforce it at some point (py>=3.6)
            try:
                sampler = sampler_c()
                method = SobolAttributionMethod(model, grid_size = 2, nb_design = 8, sampler = sampler)
                stis_maps = method.explain(x, y)

                assert x.shape[:-1] == stis_maps.shape[:-1]
            except ModuleNotFoundError:
                pass


def test_perturbations():
    """Ensure every proposed perturbations is working"""

    perturbations = [inpainting, blurring, amplitude]
    input_shapes = [(20, 20, 1), (20, 20, 3)]
    nb_labels = 10

    for input_shape in input_shapes:
        x, y = generate_data(input_shape, nb_labels, 10)
        model = generate_model(input_shape, nb_labels)

        for perturb in perturbations:
            method = SobolAttributionMethod(model, grid_size = 2, nb_design = 8, perturbation_function = perturb)
            stis_maps = method.explain(x, y)

            assert x.shape[:-1] == stis_maps.shape[:-1]


def test_estimators():
    """Ensure every proposed estimator is working"""

    estimators = [JanonEstimator(), JansenEstimator(), GlenEstimator(), SaltelliEstimator(), HommaEstimator()]
    input_shapes = [(20, 20, 1), (20, 20, 3)]
    nb_labels = 10

    for input_shape in input_shapes:
        x, y = generate_data(input_shape, nb_labels, 10)
        model = generate_model(input_shape, nb_labels)

        for estimator in estimators:
            method = SobolAttributionMethod(model, grid_size = 2, nb_design = 8, estimator = estimator)
            stis_maps = method.explain(x, y)

            assert x.shape[:-1] == stis_maps.shape[:-1]


def test_ishigami():
    """Ensure every proposed estimator compute the correct Sobol total indice estimator on the ishigami function"""

    # Ishigami function
    # Y = sin(X1) + a sin(X2)^2 + b * X3^4 * sin(X1)
    # X in U([-pi, pi])^3

    # analytically we find:
    # V(Y) = a^2 / 8 + b*pi^4/5 + b^2pi^8/18 + 1/2

    # V1 = b * pi^4 / 5 + b^2 * pi^8 / 50 + 1/2
    # V2 = a^2/8
    # V3 = 0

    # V12 = 0, V23 = 0, V13 = 8b^2pi^8/225
    # V13 = 8b^2 pi^8 / 255
    # V23 = 0
    # V123 = 0

    # S1 = V1 / V(Y)
    # S2 = V2 / V(Y)
    # S3 = V3 / V(Y)
    # S13 = V13 / V(Y)

    # ST1 = S1 + S12 + S13 + S123 = S1 + S13
    # ST2 = S2 + S12 + S23 + S123 = S2
    # ST3 = S3 + S13 + S23 + S123 = S13

    def ishigami_function(X, a, b):
        return np.sin(X[:, 0]) + a * np.sin(X[:, 1])**2.0 + b * np.sin(X[:, 0]) * X[:, 2]**4.0

    n = 2**15
    a = 7
    b = 0.1

    # order 1
    Vy = a**2.0 / 8.0 + b * np.pi**4.0 / 5.0 + b**2.0 * np.pi**8.0 / 18.0 + 0.5
    V1 = b * np.pi**4.0 / 5 + b**2.0 * np.pi**8.0 / 50.0 + 0.5
    V2 = a**2.0 / 8.0
    V3 = 0.0

    # order 2
    V12 = 0.0
    V13 = 8.0 * b**2.0 * np.pi**8.0 / 225.0
    V23 = 0.0

    # order 3
    V123 = 0.0

    S1 = V1 / Vy
    S2 = V2 / Vy
    S3 = V3 / Vy

    S12 = V12 / Vy
    S13 = V13 / Vy
    S23 = V23 / Vy

    S123 = V123 / Vy

    ST1 = S1 + S12 + S13 + S123
    ST2 = S2 + S12 + S23 + S123
    ST3 = S3 + S13 + S23 + S123

    STis = np.array([ST1, ST2, ST3])

    X = TFSobolSequenceRS()(3, n)
    X *= 2.0 * np.pi
    X -= np.pi

    Y = ishigami_function(X, a, b)

    estimators = [JanonEstimator(), JansenEstimator(), GlenEstimator(), SaltelliEstimator(), HommaEstimator()]

    for est in estimators:
        predicted_STis = est(X, Y, n)
        assert almost_equal(STis, predicted_STis, 1e-3)



