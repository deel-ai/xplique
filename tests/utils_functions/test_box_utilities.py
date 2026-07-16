"""Focused tests for object-detection tensor and formatter utilities."""

import pytest
import tensorflow as tf

from xplique.utils_functions.object_detection.base.box_formatter import BaseBoxFormatter
from xplique.utils_functions.object_detection.base.box_manager import BoxFormat, BoxType
from xplique.utils_functions.object_detection.tf.box_formatter import TfBaseBoxFormatter
from xplique.utils_functions.object_detection.tf.multi_box_tensor import TfMultiBoxTensor

try:
    import torch

    from xplique.utils_functions.object_detection.torch.box_formatter import TorchBaseBoxFormatter

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class _TfFormatter(TfBaseBoxFormatter):
    def forward(self, predictions):
        return predictions


class _ForwardOnlyFormatter(BaseBoxFormatter):
    def forward(self, predictions):
        return predictions


def test_tf_multibox_tensor_length_is_detection_count():
    assert len(TfMultiBoxTensor(tf.zeros((0, 7)))) == 0
    assert len(TfMultiBoxTensor(tf.zeros((3, 7)))) == 3


def test_base_formatter_only_requires_forward():
    formatter = _ForwardOnlyFormatter(BoxType(BoxFormat.XYXY, is_normalized=True))
    assert formatter({"predictions": "ok"}) == {"predictions": "ok"}

    with pytest.raises(NotImplementedError):
        formatter.format_predictions({"predictions": "ok"})


def test_tf_formatter_expands_rank_one_scores():
    formatter = _TfFormatter(BoxType(BoxFormat.XYXY, is_normalized=True))
    formatted = formatter.format_predictions(
        {
            "boxes": tf.constant([[0.1, 0.2, 0.3, 0.4]]),
            "scores": tf.constant([0.9]),
            "probas": tf.constant([[0.2, 0.8]]),
        }
    )

    assert formatted.shape == (1, 7)
    tf.debugging.assert_near(formatted.scores(), [0.9])


if HAS_TORCH:

    class _TorchFormatter(TorchBaseBoxFormatter):
        def forward(self, predictions):
            return predictions


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_torch_formatter_expands_rank_one_scores():
    formatter = _TorchFormatter(BoxType(BoxFormat.XYXY, is_normalized=True))
    formatted = formatter.format_predictions(
        {
            "boxes": torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
            "scores": torch.tensor([0.9]),
            "probas": torch.tensor([[0.2, 0.8]]),
        }
    )

    assert formatted.shape == (1, 7)
    assert torch.allclose(formatted.scores(), torch.tensor([0.9]))
