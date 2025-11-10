import pytest
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from xplique.attributions import HsicAttributionMethod
from xplique.attributions.global_sensitivity_analysis import (
    SobolevEstimator,
    BinaryEstimator,
    RbfEstimator,
)
from xplique.attributions.global_sensitivity_analysis.kernels import Kernel
from ..utils import generate_data, generate_model, almost_equal


def test_hsic_kernels_shape():
    """
    Test if the kernels are correctly computed.
    """
    input_shapes = [(20, 20, 1), (20, 20, 3)]
    nb_labels = 10

    for input_shape in input_shapes:
        x, y = generate_data(input_shape, nb_labels, 10)
        model = generate_model(input_shape, nb_labels)

        method = HsicAttributionMethod(model, grid_size=2, nb_design=8)
        stis_maps = method.explain(x, y)

        assert x.shape[:-1] == stis_maps.shape[:-1]


def test_estimators():
    """Ensure every proposed estimator is working"""

    estimators = [SobolevEstimator(), BinaryEstimator(), RbfEstimator()]
    input_shapes = [(20, 20, 1), (20, 20, 3)]
    nb_labels = 10

    for input_shape in input_shapes:
        x, y = generate_data(input_shape, nb_labels, 10)
        model = generate_model(input_shape, nb_labels)

        for estimator in estimators:
            method = HsicAttributionMethod(
                model, grid_size=2, nb_design=8, estimator=estimator
            )
            stis_maps = method.explain(x, y)

            assert x.shape[:-1] == stis_maps.shape[:-1]

def test_estimator_batch_size():
    """Ensure estimator batch size is correctly used"""
    input_shape = (20, 20, 1)
    nb_labels = 10
    x, y = generate_data(input_shape, nb_labels, 10)
    model = generate_model(input_shape, nb_labels)

    for estimator_batch_size in [None, 1, 4, 7, 23, 37, 103]:
        method = HsicAttributionMethod(
            model, grid_size=5, nb_design=10, estimator_batch_size=estimator_batch_size
        )
        if estimator_batch_size is None:
            assert method.estimator.batch_size == 100000
        else:
            assert method.estimator.batch_size == estimator_batch_size
        
        stis_maps = method.explain(x, y)
        assert x.shape[:-1] == stis_maps.shape[:-1]


# Some tests to make sure that the HSIC estimators make sense
@pytest.mark.parametrize("Estimator", [SobolevEstimator, BinaryEstimator, RbfEstimator])
def test_hsic_zero_when_output_constant(Estimator):
    nb_design, H, W, C = 16, 3, 2, 1
    masks = tf.random.uniform((nb_design, H, W, C), minval=0., maxval=1., dtype=tf.float32)
    outputs = tf.ones((nb_design, ), dtype=tf.float32)  # constant
    est = Estimator()

    scores = est(masks, outputs, nb_design).numpy()
    assert np.allclose(scores, 0.0, atol=1e-6)


@pytest.mark.parametrize("Estimator", [SobolevEstimator, BinaryEstimator, RbfEstimator])
def test_hsic_zero_when_masks_constant(Estimator):
    nb_design, H, W, C = 12, 2, 2, 1
    base = tf.constant(  # shape (H, W, C)
        [
            [[0.3], [0.7]],
            [[0.1], [0.9]],
        ],
        dtype=tf.float32,
    )
    masks = tf.repeat(base[None, ...], repeats=nb_design, axis=0)  # (n, H, W, C)
    outputs = tf.random.uniform((nb_design,), dtype=tf.float32)
    est = Estimator()

    scores = est(masks, outputs, nb_design).numpy()
    assert np.allclose(scores, 0.0, atol=1e-6)


@pytest.mark.parametrize("Estimator", [SobolevEstimator, BinaryEstimator, RbfEstimator])
def test_hsic_invariant_to_design_permutation(Estimator):
    nb_design, H, W, C = 24, 3, 3, 1
    rng = tf.random.Generator.from_seed(123)
    masks = rng.uniform((nb_design, H, W, C), dtype=tf.float32)
    outputs = rng.uniform((nb_design,), dtype=tf.float32)

    perm = tf.random.shuffle(tf.range(nb_design), seed=123)
    masks_perm = tf.gather(masks, perm, axis=0)
    outputs_perm = tf.gather(outputs, perm, axis=0)

    est = Estimator()
    s1 = est(masks, outputs, nb_design).numpy()
    s2 = est(masks_perm, outputs_perm, nb_design).numpy()

    assert np.allclose(s1, s2, atol=1e-6)


@pytest.mark.parametrize("Estimator", [SobolevEstimator, BinaryEstimator, RbfEstimator])
def test_hsic_detects_dependent_dimension(Estimator):
    nb_design, H, W, C = 64, 2, 1, 1  # two dimensions total
    rng = tf.random.Generator.from_seed(7)
    dep_dim = rng.uniform((nb_design,), dtype=tf.float32)  # this drives Y
    indep_dim = rng.uniform((nb_design,), dtype=tf.float32)

    # masks[..., 0,0,0] = dep_dim, masks[..., 1,0,0] = indep_dim
    masks = tf.stack([dep_dim, indep_dim], axis=1)  # (nb_design, 2)
    masks = tf.reshape(masks, (nb_design, H, W, C))  # (n,2,1,1)

    outputs = tf.identity(dep_dim)
    est = Estimator()

    scores = est(masks, outputs, nb_design).numpy()  # shape (W,H,C) = (1,2,1)
    scores_hw = np.transpose(scores, (1,0,2)).reshape(H)  # back to [H] order
    assert np.argmax(scores_hw) == 0
    assert scores_hw[0] > scores_hw[1] + 1e-3


def test_output_rbf_width_tfp_matches_numpy():
    rng = np.random.default_rng(0)
    n = 25
    Y_np = rng.normal(size=(n,1)).astype(np.float32)

    w_np = np.percentile(Y_np, 50.0).astype(np.float32)
    w_tf = tf.cast(tfp.stats.percentile(tf.convert_to_tensor(Y_np), 50.0, interpolation='linear'), tf.float32)

    # Build same Gram using each width
    X = tf.convert_to_tensor(Y_np)
    G_np = Kernel.from_string("rbf")(X, tf.transpose(X), width=tf.constant(w_np))
    G_tf = Kernel.from_string("rbf")(X, tf.transpose(X), width=w_tf)

    assert np.allclose(G_np.numpy(), G_tf.numpy(), atol=1e-6)


def test_rbf_kernel_symmetry_and_range():
    rng = tf.random.Generator.from_seed(99)
    n = 20
    X = tf.expand_dims(rng.normal((n,), dtype=tf.float32), 1)  # (n,1)
    width = tf.constant(0.7, tf.float32)
    K = Kernel.from_string("rbf")(X, tf.transpose(X), width=width)  # (n,n)
    assert K.shape == (n, n)
    assert tf.reduce_all(K >= 0.0)
    assert np.allclose(K.numpy(), K.numpy().T, atol=1e-6)
    assert float(tf.linalg.diag_part(K).numpy().min()) <= 1.0 + 1e-6


def test_binary_kernel_values():
    # For x,y in {0,1}: K = 0.5 - (x - y)^2
    X = tf.constant([[0.],[0.],[1.],[1.]], tf.float32)
    K = Kernel.from_string("binary")(X, tf.transpose(X))  # (4,4)
    expected = np.array([
        [0.5, 0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5, -0.5],
        [-0.5, -0.5, 0.5, 0.5],
        [-0.5, -0.5, 0.5, 0.5],
    ], dtype=np.float32)
    assert np.allclose(K.numpy(), expected, atol=1e-6)


def test_attributions_stable_across_estimator_batch_size():
    input_shape = (20, 20, 3)
    nb_labels = 5
    x, y = generate_data(input_shape, nb_labels, samples=6)
    model = generate_model(input_shape, nb_labels)

    m_small = HsicAttributionMethod(model, grid_size=3, nb_design=12, estimator_batch_size=2)
    m_large = HsicAttributionMethod(model, grid_size=3, nb_design=12, estimator_batch_size=10**6)

    map_small = m_small.explain(x, y).numpy()
    map_large = m_large.explain(x, y).numpy()

    assert np.allclose(map_small, map_large, rtol=1e-5, atol=1e-6)


