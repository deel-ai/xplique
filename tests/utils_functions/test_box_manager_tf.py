"""
Tests for TensorFlow BoxManager and BoxCoordinatesTranslator.
"""
import tensorflow as tf
from xplique.utils_functions.object_detection.base.box_manager import (
    BoxFormat, BoxType
)
from xplique.utils_functions.object_detection.tf.box_manager import (
    TfBoxManager, TfBoxCoordinatesTranslator
)

# Tests for methods used by TfBoxCoordinatesTranslator

def test_normalize_boxes():
    """Test normalizing boxes from pixel coordinates to [0, 1] range."""
    raw_boxes = tf.constant([[50, 50, 100, 100], [30, 30, 60, 60]], dtype=tf.float32)
    image_source_size = tf.constant([200, 200], dtype=tf.float32)
    normalized_boxes = TfBoxManager.normalize_boxes(raw_boxes, image_source_size)
    expected_boxes = tf.constant(
        [[0.25, 0.25, 0.5, 0.5], [0.15, 0.15, 0.3, 0.3]], dtype=tf.float32
    )
    assert tf.reduce_all(tf.abs(normalized_boxes - expected_boxes) < 1e-6)

def test_box_cxcywh_to_xyxy():
    """Test converting boxes from CXCYWH format to XYXY format."""
    cxcywh_boxes = tf.constant([[50, 50, 20, 20], [0.3, 0.3, 0.1, 0.1]], dtype=tf.float32)
    xyxy_boxes = TfBoxManager.box_cxcywh_to_xyxy(cxcywh_boxes)
    expected_boxes = tf.constant(
        [[40, 40, 60, 60], [0.25, 0.25, 0.35, 0.35]], dtype=tf.float32
    )
    assert tf.reduce_all(tf.abs(xyxy_boxes - expected_boxes) < 1e-6)

def test_box_xyxy_to_cxcywh():
    """Test converting boxes from XYXY format to CXCYWH format."""
    xyxy_boxes = tf.constant([[40, 40, 60, 60], [0.25, 0.25, 0.35, 0.35]], dtype=tf.float32)
    cxcywh_boxes = TfBoxManager.box_xyxy_to_cxcywh(xyxy_boxes)
    expected_boxes = tf.constant([[50, 50, 20, 20], [0.3, 0.3, 0.1, 0.1]], dtype=tf.float32)
    assert tf.reduce_all(tf.abs(cxcywh_boxes - expected_boxes) < 1e-6)

def test_box_xywh_to_xyxy():
    """Test converting boxes from XYWH format to XYXY format."""
    xywh_boxes = tf.constant(
        [[40, 40, 20, 20], [0.25, 0.25, 0.1, 0.1]], dtype=tf.float32
    )
    xyxy_boxes = TfBoxManager.box_xywh_to_xyxy(xywh_boxes)
    expected_boxes = tf.constant(
        [[40, 40, 60, 60], [0.25, 0.25, 0.35, 0.35]], dtype=tf.float32
    )
    assert tf.reduce_all(tf.abs(xyxy_boxes - expected_boxes) < 1e-6)

def test_denormalize_boxes():
    """Test denormalizing boxes from [0, 1] range to pixel coordinates."""
    normalized_boxes = tf.constant(
        [[0.25, 0.25, 0.5, 0.5], [0.15, 0.15, 0.3, 0.3]], dtype=tf.float32
    )
    image_target_size = tf.constant([400, 400], dtype=tf.float32)
    denormalized_boxes = TfBoxManager.denormalize_boxes(normalized_boxes, image_target_size)
    expected_boxes = tf.constant(
        [[100, 100, 200, 200], [60, 60, 120, 120]], dtype=tf.float32
    )
    assert tf.reduce_all(tf.abs(denormalized_boxes - expected_boxes) < 1e-5)

def test_denormalize_boxes_with_tuple():
    """Test denormalizing boxes with Python tuple input (like PIL Image.size)."""
    normalized_boxes = tf.constant([[0.25, 0.25, 0.5, 0.5]], dtype=tf.float32)
    image_size = (640, 480)  # Python tuple like PIL Image.size
    denormalized_boxes = TfBoxManager.denormalize_boxes(normalized_boxes, image_size)
    expected_boxes = tf.constant([[160, 120, 320, 240]], dtype=tf.float32)
    assert tf.reduce_all(tf.abs(denormalized_boxes - expected_boxes) < 1e-5)

# Integration tests for TfBoxCoordinatesTranslator

def test_translator_detr_normalized_cxcywh_to_normalized_xyxy():
    """Test DETR typical case: normalized CXCYWH -> normalized XYXY."""
    translator = TfBoxCoordinatesTranslator(
        input_box_type=BoxType(BoxFormat.CXCYWH, is_normalized=True),
        output_box_type=BoxType(BoxFormat.XYXY, is_normalized=True)
    )
    input_boxes = tf.constant([[0.3, 0.4, 0.1, 0.2]], dtype=tf.float32)
    output_boxes = translator.translate(input_boxes)
    # CXCYWH [0.3, 0.4, 0.1, 0.2] -> XYXY [0.25, 0.3, 0.35, 0.5]
    expected_boxes = tf.constant([[0.25, 0.3, 0.35, 0.5]], dtype=tf.float32)
    assert tf.reduce_all(tf.abs(output_boxes - expected_boxes) < 1e-6)

def test_translator_fcos_pixel_xyxy_to_normalized_xyxy():
    """Test FCOS typical case: pixel XYXY -> normalized XYXY."""
    translator = TfBoxCoordinatesTranslator(
        input_box_type=BoxType(BoxFormat.XYXY, is_normalized=False),
        output_box_type=BoxType(BoxFormat.XYXY, is_normalized=True)
    )
    input_boxes = tf.constant([[50, 50, 100, 100]], dtype=tf.float32)
    image_size = tf.constant([200, 200], dtype=tf.float32)
    output_boxes = translator.translate(input_boxes, input_image_size=image_size)
    # Pixel XYXY [50, 50, 100, 100] with image 200x200 -> normalized [0.25, 0.25, 0.5, 0.5]
    expected_boxes = tf.constant([[0.25, 0.25, 0.5, 0.5]], dtype=tf.float32)
    assert tf.reduce_all(tf.abs(output_boxes - expected_boxes) < 1e-6)

def test_translator_normalized_xyxy_to_normalized_cxcywh():
    """Test converting normalized XYXY to normalized CXCYWH."""
    translator = TfBoxCoordinatesTranslator(
        input_box_type=BoxType(BoxFormat.XYXY, is_normalized=True),
        output_box_type=BoxType(BoxFormat.CXCYWH, is_normalized=True)
    )
    input_boxes = tf.constant([[0.25, 0.3, 0.35, 0.5]], dtype=tf.float32)
    output_boxes = translator.translate(input_boxes)
    # XYXY [0.25, 0.3, 0.35, 0.5] -> CXCYWH [0.3, 0.4, 0.1, 0.2]
    expected_boxes = tf.constant([[0.3, 0.4, 0.1, 0.2]], dtype=tf.float32)
    assert tf.reduce_all(tf.abs(output_boxes - expected_boxes) < 1e-6)

def test_translator_normalized_xyxy_to_pixel_xyxy():
    """Test converting normalized XYXY to pixel XYXY."""
    translator = TfBoxCoordinatesTranslator(
        input_box_type=BoxType(BoxFormat.XYXY, is_normalized=True),
        output_box_type=BoxType(BoxFormat.XYXY, is_normalized=False)
    )
    input_boxes = tf.constant([[0.25, 0.25, 0.5, 0.5]], dtype=tf.float32)
    image_size = tf.constant([400, 400], dtype=tf.float32)
    output_boxes = translator.translate(input_boxes, output_image_size=image_size)
    # Normalized [0.25, 0.25, 0.5, 0.5] with image 400x400 -> pixel [100, 100, 200, 200]
    expected_boxes = tf.constant([[100, 100, 200, 200]], dtype=tf.float32)
    assert tf.reduce_all(tf.abs(output_boxes - expected_boxes) < 1e-6)
