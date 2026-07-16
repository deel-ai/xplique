"""
Tests for BoxManager and BoxCoordinatesTranslator across NumPy, PyTorch, and TensorFlow backends.
"""

import numpy as np
import pytest

from xplique.utils_functions.object_detection.base.box_manager import (
    BoxFormat,
    BoxType,
    NumpyBoxCoordinatesTranslator,
    NumpyBoxManager,
)

try:
    import torch

    from xplique.utils_functions.object_detection.torch.box_manager import (
        TorchBoxCoordinatesTranslator,
        TorchBoxManager,
    )

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import tensorflow as tf

from xplique.utils_functions.object_detection.tf.box_manager import (
    TfBoxCoordinatesTranslator,
    TfBoxManager,
)


class BaseBoxManagerTests:
    """Shared tests for all backend BoxManager implementations."""

    box_manager_cls = None
    translator_cls = None

    def make_tensor(self, data):
        raise NotImplementedError

    def make_image_size(self, size):
        """Return a backend-appropriate representation of a square (size x size) image."""
        raise NotImplementedError

    def allclose(self, a, b):
        raise NotImplementedError

    # --- Unit tests ---

    def test_normalize_boxes(self):
        raw_boxes = self.make_tensor([[50, 50, 100, 100], [30, 30, 60, 60]])
        result = self.box_manager_cls.normalize_boxes(raw_boxes, self.make_image_size(200))
        expected = self.make_tensor([[0.25, 0.25, 0.5, 0.5], [0.15, 0.15, 0.3, 0.3]])
        assert self.allclose(result, expected)

    def test_box_cxcywh_to_xyxy(self):
        boxes = self.make_tensor([[50, 50, 20, 20], [0.3, 0.3, 0.1, 0.1]])
        result = self.box_manager_cls.box_cxcywh_to_xyxy(boxes)
        expected = self.make_tensor([[40, 40, 60, 60], [0.25, 0.25, 0.35, 0.35]])
        assert self.allclose(result, expected)

    def test_box_xyxy_to_cxcywh(self):
        boxes = self.make_tensor([[40, 40, 60, 60], [0.25, 0.25, 0.35, 0.35]])
        result = self.box_manager_cls.box_xyxy_to_cxcywh(boxes)
        expected = self.make_tensor([[50, 50, 20, 20], [0.3, 0.3, 0.1, 0.1]])
        assert self.allclose(result, expected)

    def test_box_xywh_to_xyxy(self):
        boxes = self.make_tensor([[40, 40, 20, 20], [0.25, 0.25, 0.1, 0.1]])
        result = self.box_manager_cls.box_xywh_to_xyxy(boxes)
        expected = self.make_tensor([[40, 40, 60, 60], [0.25, 0.25, 0.35, 0.35]])
        assert self.allclose(result, expected)

    def test_box_conversion_preserves_trailing_columns(self):
        boxes = self.make_tensor([[40, 40, 20, 20, 0.8, 1.0]])
        result = self.box_manager_cls.box_xywh_to_xyxy(boxes)
        expected = self.make_tensor([[40, 40, 60, 60, 0.8, 1.0]])
        assert self.allclose(result, expected)

    def test_box_xyxy_to_xywh(self):
        boxes = self.make_tensor([[40, 40, 60, 60], [0.25, 0.25, 0.35, 0.35]])
        result = self.box_manager_cls.box_xyxy_to_xywh(boxes)
        expected = self.make_tensor([[40, 40, 20, 20], [0.25, 0.25, 0.1, 0.1]])
        assert self.allclose(result, expected)

    def test_denormalize_boxes(self):
        boxes = self.make_tensor([[0.25, 0.25, 0.5, 0.5], [0.15, 0.15, 0.3, 0.3]])
        result = self.box_manager_cls.denormalize_boxes(boxes, self.make_image_size(400))
        expected = self.make_tensor([[100, 100, 200, 200], [60, 60, 120, 120]])
        assert self.allclose(result, expected)

    def test_denormalize_boxes_with_tuple(self):
        """TF-specific: denormalize accepts a plain Python tuple (like PIL Image.size)."""
        boxes = self.make_tensor([[0.25, 0.25, 0.5, 0.5]])
        result = self.box_manager_cls.denormalize_boxes(boxes, (640, 480))
        expected = self.make_tensor([[160, 120, 320, 240]])
        assert self.allclose(result, expected)

    # --- Integration tests for BoxCoordinatesTranslator ---

    def test_translator_normalized_cxcywh_to_normalized_xyxy(self):
        """DETR typical case: normalized CXCYWH -> normalized XYXY."""
        translator = self.translator_cls(
            input_box_type=BoxType(BoxFormat.CXCYWH, is_normalized=True),
            output_box_type=BoxType(BoxFormat.XYXY, is_normalized=True),
        )
        boxes = self.make_tensor([[0.3, 0.4, 0.1, 0.2]])
        result = translator.translate(boxes)
        expected = self.make_tensor([[0.25, 0.3, 0.35, 0.5]])
        assert self.allclose(result, expected)

    def test_translator_pixel_xyxy_to_normalized_xyxy(self):
        """FCOS typical case: pixel XYXY -> normalized XYXY."""
        translator = self.translator_cls(
            input_box_type=BoxType(BoxFormat.XYXY, is_normalized=False),
            output_box_type=BoxType(BoxFormat.XYXY, is_normalized=True),
        )
        boxes = self.make_tensor([[50, 50, 100, 100]])
        result = translator.translate(boxes, image_size=self.make_image_size(200))
        expected = self.make_tensor([[0.25, 0.25, 0.5, 0.5]])
        assert self.allclose(result, expected)

    def test_translator_normalized_xyxy_to_normalized_cxcywh(self):
        translator = self.translator_cls(
            input_box_type=BoxType(BoxFormat.XYXY, is_normalized=True),
            output_box_type=BoxType(BoxFormat.CXCYWH, is_normalized=True),
        )
        boxes = self.make_tensor([[0.25, 0.3, 0.35, 0.5]])
        result = translator.translate(boxes)
        expected = self.make_tensor([[0.3, 0.4, 0.1, 0.2]])
        assert self.allclose(result, expected)

    def test_translator_normalized_xyxy_to_pixel_xyxy(self):
        translator = self.translator_cls(
            input_box_type=BoxType(BoxFormat.XYXY, is_normalized=True),
            output_box_type=BoxType(BoxFormat.XYXY, is_normalized=False),
        )
        boxes = self.make_tensor([[0.25, 0.25, 0.5, 0.5]])
        result = translator.translate(boxes, image_size=self.make_image_size(400))
        expected = self.make_tensor([[100, 100, 200, 200]])
        assert self.allclose(result, expected)


class TestNumpyBoxManager(BaseBoxManagerTests):
    box_manager_cls = NumpyBoxManager
    translator_cls = NumpyBoxCoordinatesTranslator

    def make_tensor(self, data):
        return np.array(data, dtype=np.float32)

    def make_image_size(self, size):
        return (size, size)

    def allclose(self, a, b):
        return np.allclose(a, b)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestTorchBoxManager(BaseBoxManagerTests):
    box_manager_cls = TorchBoxManager if HAS_TORCH else None
    translator_cls = TorchBoxCoordinatesTranslator if HAS_TORCH else None

    def make_tensor(self, data):
        return torch.tensor(data, dtype=torch.float32)

    def make_image_size(self, size):
        return torch.Size([size, size])

    def allclose(self, a, b):
        return torch.allclose(a, b)


class TestTfBoxManager(BaseBoxManagerTests):
    box_manager_cls = TfBoxManager
    translator_cls = TfBoxCoordinatesTranslator

    def make_tensor(self, data):
        return tf.constant(data, dtype=tf.float32)

    def make_image_size(self, size):
        return tf.constant([size, size], dtype=tf.float32)

    def allclose(self, a, b):
        return bool(tf.reduce_all(tf.abs(a - b) < 1e-5))


def test_numpy_box_manager_handles_integer_boxes_and_metadata():
    boxes = np.array([[50, 25, 100, 50, 1, 2]], dtype=np.int32)

    normalized = NumpyBoxManager.normalize_boxes(boxes, (200, 100))
    denormalized = NumpyBoxManager.denormalize_boxes(normalized, (200, 100))

    assert np.issubdtype(normalized.dtype, np.floating)
    np.testing.assert_allclose(normalized, [[0.25, 0.25, 0.5, 0.5, 1.0, 2.0]])
    np.testing.assert_allclose(denormalized, [[50, 25, 100, 50, 1.0, 2.0]])


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_torch_box_manager_handles_integer_boxes_and_metadata():
    boxes = torch.tensor([[50, 25, 100, 50, 1, 2]], dtype=torch.int64)

    normalized = TorchBoxManager.normalize_boxes(boxes, (200, 100))
    denormalized = TorchBoxManager.denormalize_boxes(normalized, (200, 100))

    assert normalized.dtype == torch.float32
    assert torch.allclose(normalized, torch.tensor([[0.25, 0.25, 0.5, 0.5, 1.0, 2.0]]))
    assert torch.allclose(denormalized, torch.tensor([[50, 25, 100, 50, 1.0, 2.0]]))


def test_tf_box_manager_handles_integer_boxes_and_metadata_in_a_graph():
    @tf.function
    def normalize_and_denormalize(boxes):
        normalized = TfBoxManager.normalize_boxes(boxes, (200, 100))
        return normalized, TfBoxManager.denormalize_boxes(normalized, (200, 100))

    boxes = tf.constant([[50, 25, 100, 50, 1, 2]], dtype=tf.int32)
    normalized, denormalized = normalize_and_denormalize(boxes)

    assert normalized.dtype == tf.float32
    np.testing.assert_allclose(normalized, [[0.25, 0.25, 0.5, 0.5, 1.0, 2.0]])
    np.testing.assert_allclose(denormalized, [[50, 25, 100, 50, 1.0, 2.0]])


def test_numpy_translator_uses_enum_members_for_format_checks():
    translator = NumpyBoxCoordinatesTranslator(
        input_box_type=BoxType(BoxFormat.CXCYWH, is_normalized=True),
        output_box_type=BoxType(BoxFormat.XYXY, is_normalized=True),
    )

    translator.input_box_type.format = BoxFormat.XYWH.value
    boxes = np.array([[0.3, 0.4, 0.1, 0.2]], dtype=np.float32)
    result = translator.translate(boxes)

    np.testing.assert_allclose(result, boxes)
    assert not np.allclose(result, [[0.25, 0.3, 0.35, 0.5]])
