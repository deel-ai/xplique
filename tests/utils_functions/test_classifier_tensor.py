"""Tests for classifier tensor wrappers across TensorFlow and PyTorch."""

import numpy as np
import pytest
import tensorflow as tf

from xplique.utils_functions.classification.tf.classifier_tensor import TfClassifierTensor

try:
    import torch

    from xplique.utils_functions.classification.torch.classifier_tensor import TorchClassifierTensor

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def test_tf_classifier_tensor_validates_rank_and_returns_tf_tensor_batch():
    with pytest.raises(ValueError, match="rank 1 or 2"):
        TfClassifierTensor(tf.ones((1, 2, 3)))

    classifier = TfClassifierTensor(np.array([0.1, 0.9], dtype=np.float32))
    batched = classifier.to_batched_tensor()
    assert isinstance(batched, tf.Tensor)
    assert batched.shape == (1, 2)


def test_tf_classifier_tensor_len_deprecation_and_properties():
    classifier = TfClassifierTensor(tf.constant([[0.1, 0.9], [0.3, 0.7]], dtype=tf.float32))

    with pytest.warns(DeprecationWarning):
        assert len(classifier) == 2

    assert classifier.num_classes == 2
    assert classifier.batch_size == 2
    assert not classifier.is_empty


def test_tf_classifier_tensor_empty_reports_is_empty():
    classifier = TfClassifierTensor(tf.zeros((0, 3), dtype=tf.float32))
    assert classifier.batch_size == 0
    assert classifier.is_empty


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_torch_classifier_tensor_validates_rank_and_properties():
    with pytest.raises(ValueError, match="rank 1 or 2"):
        TorchClassifierTensor.from_predictions(torch.ones((1, 2, 3)))

    classifier = TorchClassifierTensor.from_predictions(torch.tensor([[0.1, 0.9], [0.3, 0.7]]))
    with pytest.warns(DeprecationWarning):
        assert len(classifier) == 2

    assert classifier.num_classes == 2
    assert classifier.batch_size == 2
    assert not classifier.is_empty


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_torch_classifier_tensor_empty_reports_is_empty():
    classifier = TorchClassifierTensor.from_predictions(torch.zeros((0, 3)))
    assert classifier.batch_size == 0
    assert classifier.is_empty


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_tf_and_torch_classifier_tensor_properties_match_on_same_input():
    values = np.array([[0.1, 0.2, 0.7], [0.4, 0.5, 0.1]], dtype=np.float32)
    tf_classifier = TfClassifierTensor(values)
    torch_classifier = TorchClassifierTensor.from_predictions(values)

    assert tf_classifier.num_classes == torch_classifier.num_classes
    assert tf_classifier.batch_size == torch_classifier.batch_size
