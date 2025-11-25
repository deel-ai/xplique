import numpy as np
import tensorflow as tf
import pytest

from xplique.metrics.base import BaseComplexityMetric
from xplique.metrics import Complexity, Sparseness

from tests.utils import almost_equal

EPS = 1e-8


class _DummyComplexityMetric(BaseComplexityMetric):
    """Sum over non-batch dims as per-sample score."""

    def detailed_evaluate(self, explanations: tf.Tensor) -> np.ndarray:
        expl = explanations.numpy()
        # sum over all non-batch dimensions
        axes = tuple(range(1, expl.ndim))
        return np.sum(expl, axis=axes)


def _entropy_numpy(explanations: np.ndarray) -> np.ndarray:
    """Mirror Complexity.detailed_evaluate in NumPy, for 2D/3D inputs."""
    x = np.asarray(explanations, dtype=np.float32)

    # no channel averaging here: caller should have already done it if needed
    x = np.abs(x)
    b = x.shape[0]
    x = x.reshape(b, -1)

    norm = np.sum(x, axis=-1, keepdims=True)
    p = x / (norm + EPS)

    entropy = -np.sum(p * np.log(p + EPS), axis=-1)
    return entropy


def _entropy_numpy_4d(explanations: np.ndarray) -> np.ndarray:
    """NumPy version of Complexity for 4D (B, H, W, C) with channel averaging."""
    x = np.asarray(explanations, dtype=np.float32)
    assert x.ndim == 4
    x_mean = np.mean(x, axis=-1)  # (B, H, W)
    return _entropy_numpy(x_mean)


def _gini_numpy(explanations: np.ndarray) -> np.ndarray:
    """Mirror Sparseness.detailed_evaluate in NumPy, for 2D/3D inputs."""
    x = np.asarray(explanations, dtype=np.float32)
    x = np.abs(x)
    b = x.shape[0]
    x = x.reshape(b, -1)

    # L1-normalize
    l1 = np.sum(x, axis=-1, keepdims=True)
    x = x / (l1 + EPS)

    # ascending sort for Gini
    x_sorted = np.sort(x, axis=-1)
    n = x_sorted.shape[1]
    idx = np.arange(1, n + 1, dtype=np.float32)  # 1..n, shape (n,)

    weighted = x_sorted * idx  # broadcast over batch
    num = 2.0 * np.sum(weighted, axis=-1)        # (B,)
    den = n * np.sum(x_sorted, axis=-1) + EPS    # (B,)
    gini = num / den - (n + 1.0) / n             # (B,)

    return gini


def _gini_numpy_4d(explanations: np.ndarray) -> np.ndarray:
    """NumPy version of Sparseness for 4D (B, H, W, C) with channel averaging."""
    x = np.asarray(explanations, dtype=np.float32)
    assert x.ndim == 4
    x_mean = np.mean(x, axis=-1)  # (B, H, W)
    return _gini_numpy(x_mean)


def test_base_complexity_metric_evaluate_and_call():
    """Evaluate should aggregate batch results and __call__ is an alias."""
    explanations = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

    for batch_size in [1, 2, 4]:
        metric = _DummyComplexityMetric(batch_size=batch_size)

        score_eval = metric.evaluate(explanations)
        score_call = metric(explanations)

        # manual computation: per-sample sums, then mean
        manual_per_sample = explanations.reshape(2, -1).sum(axis=1)
        manual_mean = float(manual_per_sample.mean())

        assert isinstance(score_eval, float)
        assert isinstance(score_call, float)
        assert almost_equal(score_eval, manual_mean)
        assert almost_equal(score_call, manual_mean)


def test_complexity_entropy_matches_numpy_2d():
    """Complexity.detailed_evaluate should match NumPy entropy for 2D inputs."""
    exps = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],    # uniform
            [1.0, 2.0, 3.0, 4.0],    # non-uniform
        ],
        dtype=np.float32,
    )

    metric = Complexity(batch_size=2)
    tf_entropy = metric.detailed_evaluate(exps)
    np_entropy = _entropy_numpy(exps)

    assert tf_entropy.shape == (2,)
    np.testing.assert_allclose(tf_entropy, np_entropy, rtol=1e-6, atol=1e-6)


def test_complexity_entropy_matches_numpy_4d_channel_averaging():
    """For 4D inputs (B, H, W, C) Complexity should average channels then compute entropy."""
    B, H, W, C = 2, 3, 4, 3
    exps = np.random.rand(B, H, W, C).astype(np.float32)

    metric = Complexity(batch_size=1)
    tf_entropy = metric.detailed_evaluate(exps)
    np_entropy = _entropy_numpy_4d(exps)

    assert tf_entropy.shape == (B,)
    np.testing.assert_allclose(tf_entropy, np_entropy, rtol=1e-6, atol=1e-6)


def test_complexity_sign_invariance():
    """Complexity should be invariant to sign flips (uses abs())."""
    exps = np.random.randn(4, 5, 6).astype(np.float32)
    metric = Complexity(batch_size=2)

    s_pos = metric.detailed_evaluate(exps)
    s_neg = metric.detailed_evaluate(-exps)

    np.testing.assert_allclose(s_pos, s_neg, rtol=1e-6, atol=1e-6)


def test_complexity_uniform_has_higher_entropy_than_delta():
    """Uniform attribution has higher entropy than a highly concentrated one."""
    n = 8
    # sample 0: uniform; sample 1: concentrated on one feature
    exps = np.stack(
        [
            np.ones((n,), dtype=np.float32),
            np.concatenate([np.ones(1, dtype=np.float32), np.zeros(n - 1, dtype=np.float32)]),
        ],
        axis=0,
    )

    metric = Complexity(batch_size=2)
    entropies = metric.detailed_evaluate(exps)

    assert entropies.shape == (2,)
    # entropy(uniform) > entropy(delta)
    assert entropies[0] > entropies[1]


def test_complexity_zero_attributions_returns_finite_entropy():
    """All-zero explanations should not produce NaNs or inf; entropy should be finite."""
    exps = np.zeros((3, 5), dtype=np.float32)
    metric = Complexity()

    ent = metric.detailed_evaluate(exps)

    assert ent.shape == (3,)
    assert np.all(np.isfinite(ent))


def test_complexity_api_evaluate_and_call_types():
    """Complexity.evaluate and __call__ should accept np and tf and return floats."""
    exps_np = np.random.rand(5, 7, 3).astype(np.float32)
    exps_tf = tf.convert_to_tensor(exps_np)

    metric = Complexity(batch_size=2)

    score_np_eval = metric.evaluate(exps_np)
    score_tf_eval = metric.evaluate(exps_tf)
    score_np_call = metric(exps_np)
    score_tf_call = metric(exps_tf)

    for s in [score_np_eval, score_tf_eval, score_np_call, score_tf_call]:
        assert isinstance(s, float)


def test_complexity_batch_size_invariance():
    """Complexity.evaluate should be invariant (up to noise) to batch_size."""
    exps = np.random.rand(11, 9).astype(np.float32)

    scores = []
    for bs in [1, 2, 5]:
        metric = Complexity(batch_size=bs)
        scores.append(metric.evaluate(exps))

    for s in scores[1:]:
        assert almost_equal(scores[0], s, epsilon=1e-6)


def test_sparseness_gini_matches_numpy_2d():
    """Sparseness.detailed_evaluate should match NumPy Gini for 2D inputs."""
    exps = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],    # uniform
            [1.0, 2.0, 3.0, 4.0],    # non-uniform
        ],
        dtype=np.float32,
    )

    metric = Sparseness(batch_size=2)
    tf_gini = metric.detailed_evaluate(exps)
    np_gini = _gini_numpy(exps)

    assert tf_gini.shape == (2,)
    np.testing.assert_allclose(tf_gini, np_gini, rtol=1e-6, atol=1e-6)


def test_sparseness_gini_matches_numpy_4d_channel_averaging():
    """For 4D inputs (B, H, W, C), Sparseness should average channels then compute Gini."""
    B, H, W, C = 3, 4, 5, 2
    exps = np.random.rand(B, H, W, C).astype(np.float32)

    metric = Sparseness(batch_size=1)
    tf_gini = metric.detailed_evaluate(exps)
    np_gini = _gini_numpy_4d(exps)

    assert tf_gini.shape == (B,)
    np.testing.assert_allclose(tf_gini, np_gini, rtol=1e-6, atol=1e-6)


def test_sparseness_sign_invariance():
    """Sparseness should be invariant to sign (uses abs())."""
    exps = np.random.randn(4, 6).astype(np.float32)
    metric = Sparseness(batch_size=2)

    g_pos = metric.detailed_evaluate(exps)
    g_neg = metric.detailed_evaluate(-exps)

    np.testing.assert_allclose(g_pos, g_neg, rtol=1e-6, atol=1e-6)


def test_sparseness_scale_invariance():
    """Sparseness (Gini) should be invariant to positive scaling (L1-normalized)."""
    exps = np.random.rand(3, 10).astype(np.float32)
    metric = Sparseness(batch_size=1)

    g1 = metric.detailed_evaluate(exps)
    g2 = metric.detailed_evaluate(3.7 * exps)  # positive scaling

    np.testing.assert_allclose(g1, g2, rtol=1e-6, atol=1e-6)


def test_sparseness_uniform_vs_one_hot():
    """
    For a uniform distribution, Gini ~ 0; for a one-hot distribution,
    Gini ~ (n-1)/n.
    """
    n = 8

    # uniform attribution
    uniform = np.ones((1, n), dtype=np.float32)
    # one-hot attribution (all mass on last feature)
    one_hot = np.zeros((1, n), dtype=np.float32)
    one_hot[0, -1] = 1.0

    metric = Sparseness(batch_size=1)

    g_uniform = metric.detailed_evaluate(uniform)[0]
    g_one_hot = metric.detailed_evaluate(one_hot)[0]

    # uniform ~ 0
    assert abs(g_uniform) < 1e-5

    # one-hot ~ (n-1)/n
    expected = (n - 1.0) / n
    assert abs(g_one_hot - expected) < 1e-5

    # and one-hot should be strictly sparser than uniform
    assert g_one_hot > g_uniform


def test_sparseness_range_for_positive_attributions():
    """For positive attributions, Gini should lie in [0, 1]."""
    exps = np.random.rand(5, 12).astype(np.float32)
    metric = Sparseness(batch_size=3)

    g = metric.detailed_evaluate(exps)

    assert g.shape == (5,)
    assert np.all(g >= -1e-6)
    assert np.all(g <= 1.0 + 1e-6)


def test_sparseness_api_evaluate_and_call_types():
    """Sparseness.evaluate and __call__ should accept np and tf and return floats."""
    exps_np = np.random.rand(7, 5, 3).astype(np.float32)
    exps_tf = tf.convert_to_tensor(exps_np)

    metric = Sparseness(batch_size=4)

    score_np_eval = metric.evaluate(exps_np)
    score_tf_eval = metric.evaluate(exps_tf)
    score_np_call = metric(exps_np)
    score_tf_call = metric(exps_tf)

    for s in [score_np_eval, score_tf_eval, score_np_call, score_tf_call]:
        assert isinstance(s, float)


def test_sparseness_batch_size_invariance():
    """Sparseness.evaluate should be invariant (up to numerical noise) to batch_size."""
    exps = np.random.rand(9, 11).astype(np.float32)

    scores = []
    for bs in [1, 2, 4]:
        metric = Sparseness(batch_size=bs)
        scores.append(metric.evaluate(exps))

    for s in scores[1:]:
        assert almost_equal(scores[0], s, epsilon=1e-6)
