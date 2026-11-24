"""Analytical and integration tests for downstream reconstruction agreement."""

import numpy as np
import pytest
from scipy.special import softmax

from xplique.metrics.concepts import ClassificationPredictionFidelity, ReconstructionR2


def test_classification_probabilities_direction_and_zero_support():
    metric = ClassificationPredictionFidelity(prediction_format="probabilities")
    original = [[1.0, 0.0], [0.5, 0.5]]
    reconstructed = [[0.25, 0.75], [0.5, 0.5]]
    report = metric(original, reconstructed)
    assert report == {
        "top1_agreement": 0.5,
        "mean_kl_divergence": pytest.approx(-np.log(0.25) / 2),
        "n_agreeing": 1,
        "n_examples": 2,
    }
    assert np.isinf(metric(reconstructed, original)["mean_kl_divergence"])
    assert metric(original, original)["mean_kl_divergence"] == 0
    with pytest.raises(ValueError, match="Probabilities"):
        metric(original, [[0.2, 0.2], [0.5, 0.5]])


@pytest.mark.parametrize("size", [None, 1, 2])
def test_classification_logits_streaming_and_validation(size):
    left = np.array([[1000, -1000], [2, 1], [1, 2], [-1, 2]])
    right = np.array([[-1000, 1000], [2, 1], [3, 0], [4, 6]])
    metric = ClassificationPredictionFidelity(prediction_format="logits", batch_size=size)
    full = metric(left, right)
    assert metric.evaluate_batches(iter([(left[:3], right[:3]), (left[3:], right[3:])])) == (
        pytest.approx(full)
    )
    assert full["top1_agreement"] == 0.5
    assert np.isfinite(full["mean_kl_divergence"])
    assert metric(left + 512, right - 256) == pytest.approx(full)
    probabilities = ClassificationPredictionFidelity(prediction_format="probabilities")
    assert metric(left[1:], right[1:]) == pytest.approx(
        probabilities(softmax(left[1:], axis=1), softmax(right[1:], axis=1))
    )
    with pytest.raises(ValueError, match="same shape"):
        metric(left, right[:1])
    with pytest.raises(ValueError, match="Class count"):
        metric.evaluate_batches([(left, right), (np.ones((1, 3)), np.ones((1, 3)))])


def test_classification_weights_a_gap_that_exceeds_float64_range():
    metric = ClassificationPredictionFidelity(prediction_format="logits")
    report = metric([[0.0, 0.0]], [[1e308, -1e308]])
    assert report["mean_kl_divergence"] == pytest.approx(1e308)
    assert np.isinf(metric([[-1e308, 1e308]], [[1e308, -1e308]])["mean_kl_divergence"])


class _AffineFactorizer:
    """Keep one varying coordinate and reconstruct the other from a decoder bias."""

    def encode(self, activations):
        return activations[:, :1]

    def decode(self, codes):
        return np.concatenate((codes, np.full_like(codes, -0.01)), axis=1)


@pytest.mark.parametrize("backend", ["tensorflow", "torch"])
def test_paired_frozen_head_with_affine_decoder(backend):
    factorizer = _AffineFactorizer()
    x = np.array([[1.0, 0.01], [100.0, -0.02]], dtype=np.float32)
    reconstructed = factorizer.decode(factorizer.encode(x))
    weights = np.array([[0, 0], [1, -1]], dtype=np.float32)
    if backend == "tensorflow":
        tf = pytest.importorskip("tensorflow")
        head = tf.keras.layers.Dense(2, use_bias=False, trainable=False)
        head.build(x.shape)
        head.set_weights([weights])
        original_pred = head(x, training=False).numpy()
        identity_pred = head(x.copy(), training=False).numpy()
        reconstructed_pred = head(reconstructed, training=False).numpy()
    else:
        torch = pytest.importorskip("torch")
        head = torch.nn.Linear(2, 2, bias=False).eval()
        with torch.no_grad():
            head.weight.copy_(torch.from_numpy(weights.T))
            original_pred = head(torch.from_numpy(x)).numpy()
            identity_pred = head(torch.from_numpy(x.copy())).numpy()
            reconstructed_pred = head(torch.from_numpy(reconstructed)).numpy()
        assert not head.training
    metric = ClassificationPredictionFidelity(prediction_format="logits")
    assert ReconstructionR2()(x, reconstructed) > 0.999
    assert metric(original_pred, identity_pred)["top1_agreement"] == 1
    assert metric(original_pred, reconstructed_pred)["top1_agreement"] == 0.5
    assert metric(original_pred, reconstructed_pred)["mean_kl_divergence"] > 0


def test_classification_constructor_validation():
    with pytest.raises(ValueError, match="prediction_format"):
        ClassificationPredictionFidelity(prediction_format="softmax")


def test_classification_validation_recovery_and_non_mutation():
    clf = ClassificationPredictionFidelity(prediction_format="logits", batch_size=1)
    logits = np.array([[2.0, 0.0], [0.0, 2.0]])
    logits_copy = logits.copy()
    with pytest.raises(ValueError, match="same shape"):
        clf(np.ones((2, 1)), np.ones((2, 1)))
    with pytest.raises(ValueError, match="finite"):
        clf(np.array([[np.nan, 0.0]]), np.array([[0.0, 1.0]]))
    with pytest.raises(ValueError, match="iterable of pairs"):
        clf.evaluate_batches((logits, logits))
    with pytest.raises(ValueError, match="predictions"):
        clf.evaluate_batches(iter(()))
    assert clf(logits, logits)["top1_agreement"] == 1.0
    np.testing.assert_array_equal(logits, logits_copy)
