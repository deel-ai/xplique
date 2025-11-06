import pytest
import numpy as np
import tensorflow as tf

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


@pytest.mark.parametrize("Estimator", [JansenEstimator, HommaEstimator, JanonEstimator, SaltelliEstimator])
def test_ishigami(Estimator):
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

    estimator = Estimator()
    predicted_STis = estimator(X, Y, n)
    assert almost_equal(STis, predicted_STis, 1e-3)


def make_abc_outputs_linear(nb_design, nb_dim, w, seed=123):
    """Return concatenated outputs [A(N), B(N), C_1(N),...,C_D(N)]
       for f(x)=sum_i w_i x_i with Xi~N(0,1). Pure TF, no item assignment."""
    g = tf.random.Generator.from_seed(seed)
    A = g.normal([nb_design, nb_dim], dtype=tf.float32)  # (N, D)
    B = g.normal([nb_design, nb_dim], dtype=tf.float32)  # (N, D)
    w = tf.convert_to_tensor(w, tf.float32)              # (D,)

    # f(X) = X @ w
    yA = tf.tensordot(A, w, axes=[[1], [0]])            # (N,)
    yB = tf.tensordot(B, w, axes=[[1], [0]])            # (N,)

    # Build all C_i in one shot: for each i, replace column i of A with column i of B
    # Shapes: A_b, B_b -> (1, N, D); mask -> (D, 1, D); result -> (D, N, D)
    A_b = A[None, :, :]
    B_b = B[None, :, :]
    mask = tf.one_hot(tf.range(nb_dim), depth=nb_dim, dtype=tf.float32)[:, None, :]  # (D,1,D)

    C = A_b * (1.0 - mask) + B_b * mask                  # (D, N, D)
    yC = tf.tensordot(C, w, axes=[[2], [0]])             # (D, N)

    outputs = tf.concat([yA, yB, tf.reshape(yC, [-1])], axis=0)  # (N + N + D*N,)
    return outputs


def outputs_from_AB(A: tf.Tensor, B: tf.Tensor, w: np.ndarray) -> tf.Tensor:
    """Pack [A, B, C_1..C_D] for f(x)=sum_i w_i x_i, given A,B (N,D)."""
    A = tf.convert_to_tensor(A, tf.float32)
    B = tf.convert_to_tensor(B, tf.float32)
    w = tf.convert_to_tensor(w, tf.float32)                   # (D,)

    yA = tf.tensordot(A, w, axes=[[1], [0]])                  # (N,)
    yB = tf.tensordot(B, w, axes=[[1], [0]])                  # (N,)

    # Build all C_i without item assignment: (D, N, D)
    D = tf.shape(A)[1]
    A_b = A[None, :, :]
    B_b = B[None, :, :]
    mask = tf.one_hot(tf.range(D), depth=D, dtype=tf.float32)[:, None, :]  # (D,1,D)
    C = A_b * (1.0 - mask) + B_b * mask

    yC = tf.tensordot(C, w, axes=[[2], [0]])                  # (D, N)
    return tf.concat([yA, yB, tf.reshape(yC, [-1])], axis=0)  # (N + N + D*N,)


def l2_err(est_vec, truth_vec):
    est = (est_vec / est_vec.sum()).ravel()
    tru = (truth_vec / truth_vec.sum()).ravel()
    return float(np.linalg.norm(est - tru))


@pytest.mark.parametrize("Estimator", [JansenEstimator, HommaEstimator, JanonEstimator, SaltelliEstimator])
def test_analytic_linear_total_indices(Estimator):
    nb_dim = 4
    nb_design = 2048
    w = np.array([1.0, 2.0, 0.5, 3.0], dtype=np.float32)  # anisotropic weights
    st_true = (w**2) / np.sum(w**2)

    # fake low-res masks: shape (B,H,W,1) with H*W=nb_dim
    grid = int(np.sqrt(nb_dim))
    assert grid*grid == nb_dim
    masks = tf.ones([1, grid, grid, 1], tf.float32)

    outputs = make_abc_outputs_linear(nb_design, nb_dim, w, seed=7)
    est = Estimator()
    st = est(masks, outputs, nb_design)    # shape (H,W) or (H,W,1)

    st = tf.reshape(st, [-1]).numpy()
    np.testing.assert_allclose(st / st.sum(), st_true / st_true.sum(), rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("Estimator", [JansenEstimator, HommaEstimator, JanonEstimator, SaltelliEstimator])
def test_affine_invariance(Estimator):
    nb_dim, nb_design = 4, 1024
    w = np.array([1.0, 2.0, 0.5, 3.0], np.float32)
    masks = tf.ones([1, 2, 2, 1], tf.float32)
    outputs = make_abc_outputs_linear(nb_design, nb_dim, w, seed=11)

    est = Estimator()
    base = tf.reshape(est(masks, outputs, nb_design), [-1])

    a, b = 3.7, -1.2
    affined = a * outputs + b
    aff = tf.reshape(est(masks, affined, nb_design), [-1])

    # Indices are scale/shift invariant (up to numerical noise)
    np.testing.assert_allclose(base.numpy(), aff.numpy(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("Estimator", [JansenEstimator, HommaEstimator, JanonEstimator, SaltelliEstimator])
def test_permutation_consistency(Estimator):
    nb_dim, nb_design = 4, 1024
    w = np.array([1.0, 2.0, 0.5, 3.0], np.float32)
    grid = 2
    masks = tf.ones([1, grid, grid, 1], tf.float32)

    outputs = make_abc_outputs_linear(nb_design, nb_dim, w, seed=5)

    est = Estimator()
    base = tf.reshape(est(masks, outputs, nb_design), [-1]).numpy()

    # Permute dimensions in outputs' C-blocks
    perm = np.array([2, 0, 3, 1], dtype=np.int32)
    A = outputs[:nb_design]
    B = outputs[nb_design:2*nb_design]
    C = tf.reshape(outputs[2*nb_design:], [nb_dim, nb_design])
    C_perm = tf.gather(C, perm, axis=0)

    outputs_perm = tf.concat([A, B, tf.reshape(C_perm, [-1])], axis=0)
    masks_perm = tf.reshape(tf.gather(tf.reshape(masks[0, ..., 0], [-1]), perm), [grid, grid])[None, ..., None]

    permuted = tf.reshape(est(masks_perm, outputs_perm, nb_design), [-1]).numpy()
    np.testing.assert_allclose(permuted, base[perm], rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("Estimator", [JansenEstimator, HommaEstimator, JanonEstimator, SaltelliEstimator])
def test_convergence(Estimator):
    D = 4
    w = np.array([1.0, 2.0, 0.5, 3.0], np.float32)
    st_true = (w**2) / np.sum(w**2)
    masks = tf.ones([1, 2, 2, 1], tf.float32)

    Ns = [512, 1024, 2048]
    S = 10  # seeds to average over for stability

    errs = np.zeros((S, len(Ns)), dtype=np.float64)

    for s in range(S):
        Nmax = max(Ns)
        # Common random numbers (stateless) → nested prefixes for each N
        A_full = tf.random.stateless_normal([Nmax, D], seed=[s, 123])
        B_full = tf.random.stateless_normal([Nmax, D], seed=[s, 456])
        est = Estimator()

        for j, N in enumerate(Ns):
            outputs = outputs_from_AB(A_full[:N], B_full[:N], w)
            st = tf.reshape(est(masks, outputs, N), [-1]).numpy()
            errs[s, j] = l2_err(st, st_true)

    mean_err = errs.mean(axis=0)
    # Monotone decrease on the **average** error
    assert mean_err[2] <= mean_err[1] + 1e-6 and mean_err[1] <= mean_err[0] + 1e-6


@pytest.mark.parametrize("Estimator", [JansenEstimator, HommaEstimator, JanonEstimator, SaltelliEstimator])
def test_small_nb_design_and_constant_outputs(Estimator):
    nb_dim, nb_design = 4, 2  # smallest allowed by ddof=1
    masks = tf.ones([1, 2, 2, 1], tf.float32)
    # A, B, C all constants => variance ~ 0 → indices should be ~0 (with your EPS guard)
    const = tf.zeros([nb_design*(2+nb_dim)], tf.float32)

    est = Estimator()
    st = tf.reshape(est(masks, const, nb_design), [-1]).numpy()
    assert np.all(np.isfinite(st))
    assert np.all(st >= -1e-6)  # allow tiny negatives from numerical noise
