import torch
from xplique.utils_functions.object_detection.base.box_manager import BoxManager, BoxFormat, BoxType
from xplique.utils_functions.object_detection.torch.box_manager import TorchBoxManager, TorchBoxCoordinatesTranslator

# Tests for methods used by TorchBoxCoordinatesTranslator

def test_normalize_boxes():
    """Test normalizing boxes from pixel coordinates to [0, 1] range."""
    raw_boxes = torch.tensor([[50, 50, 100, 100], [30, 30, 60, 60]], dtype=torch.float32)
    image_source_size = torch.Size([200, 200])
    normalized_boxes = TorchBoxManager.normalize_boxes(raw_boxes.clone(), image_source_size)
    expected_boxes = torch.tensor([[0.25, 0.25, 0.5, 0.5], [0.15, 0.15, 0.3, 0.3]], dtype=torch.float32)
    assert torch.allclose(normalized_boxes, expected_boxes)

def test_box_cxcywh_to_xyxy():
    """Test converting boxes from CXCYWH format to XYXY format."""
    cxcywh_boxes = torch.tensor([[50, 50, 20, 20], [0.3, 0.3, 0.1, 0.1]], dtype=torch.float32)
    xyxy_boxes = TorchBoxManager.box_cxcywh_to_xyxy(cxcywh_boxes)
    expected_boxes = torch.tensor([[40, 40, 60, 60], [0.25, 0.25, 0.35, 0.35]], dtype=torch.float32)
    assert torch.allclose(xyxy_boxes, expected_boxes)

def test_box_xyxy_to_cxcywh():
    """Test converting boxes from XYXY format to CXCYWH format."""
    xyxy_boxes = torch.tensor([[40, 40, 60, 60], [0.25, 0.25, 0.35, 0.35]], dtype=torch.float32)
    cxcywh_boxes = TorchBoxManager.box_xyxy_to_cxcywh(xyxy_boxes)
    expected_boxes = torch.tensor([[50, 50, 20, 20], [0.3, 0.3, 0.1, 0.1]], dtype=torch.float32)
    assert torch.allclose(cxcywh_boxes, expected_boxes)

def test_box_xywh_to_xyxy():
    """Test converting boxes from XYWH format to XYXY format."""
    xywh_boxes = torch.tensor([[40, 40, 20, 20], [0.25, 0.25, 0.1, 0.1]], dtype=torch.float32)
    xyxy_boxes = TorchBoxManager.box_xywh_to_xyxy(xywh_boxes)
    expected_boxes = torch.tensor([[40, 40, 60, 60], [0.25, 0.25, 0.35, 0.35]], dtype=torch.float32)
    assert torch.allclose(xyxy_boxes, expected_boxes)

def test_denormalize_boxes():
    """Test denormalizing boxes from [0, 1] range to pixel coordinates."""
    normalized_boxes = torch.tensor([[0.25, 0.25, 0.5, 0.5], [0.15, 0.15, 0.3, 0.3]], dtype=torch.float32)
    image_target_size = torch.Size([400, 400])
    denormalized_boxes = TorchBoxManager.denormalize_boxes(normalized_boxes, image_target_size)
    expected_boxes = torch.tensor([[100, 100, 200, 200], [60, 60, 120, 120]], dtype=torch.float32)
    assert torch.allclose(denormalized_boxes, expected_boxes)

# Integration tests for TorchBoxCoordinatesTranslator

def test_translator_detr_normalized_cxcywh_to_normalized_xyxy():
    """Test DETR typical case: normalized CXCYWH -> normalized XYXY."""
    translator = TorchBoxCoordinatesTranslator(
        input_box_type=BoxType(BoxFormat.CXCYWH, is_normalized=True),
        output_box_type=BoxType(BoxFormat.XYXY, is_normalized=True)
    )
    input_boxes = torch.tensor([[0.3, 0.4, 0.1, 0.2]], dtype=torch.float32)
    output_boxes = translator.translate(input_boxes)
    # CXCYWH [0.3, 0.4, 0.1, 0.2] -> XYXY [0.25, 0.3, 0.35, 0.5]
    expected_boxes = torch.tensor([[0.25, 0.3, 0.35, 0.5]], dtype=torch.float32)
    assert torch.allclose(output_boxes, expected_boxes)

def test_translator_fcos_pixel_xyxy_to_normalized_xyxy():
    """Test FCOS typical case: pixel XYXY -> normalized XYXY."""
    translator = TorchBoxCoordinatesTranslator(
        input_box_type=BoxType(BoxFormat.XYXY, is_normalized=False),
        output_box_type=BoxType(BoxFormat.XYXY, is_normalized=True)
    )
    input_boxes = torch.tensor([[50, 50, 100, 100]], dtype=torch.float32)
    image_size = torch.Size([200, 200])
    output_boxes = translator.translate(input_boxes, image_size)
    # Pixel XYXY [50, 50, 100, 100] with image 200x200 -> normalized [0.25, 0.25, 0.5, 0.5]
    expected_boxes = torch.tensor([[0.25, 0.25, 0.5, 0.5]], dtype=torch.float32)
    assert torch.allclose(output_boxes, expected_boxes)

def test_translator_normalized_xyxy_to_normalized_cxcywh():
    """Test converting normalized XYXY to normalized CXCYWH."""
    translator = TorchBoxCoordinatesTranslator(
        input_box_type=BoxType(BoxFormat.XYXY, is_normalized=True),
        output_box_type=BoxType(BoxFormat.CXCYWH, is_normalized=True)
    )
    input_boxes = torch.tensor([[0.25, 0.3, 0.35, 0.5]], dtype=torch.float32)
    output_boxes = translator.translate(input_boxes)
    # XYXY [0.25, 0.3, 0.35, 0.5] -> CXCYWH [0.3, 0.4, 0.1, 0.2]
    expected_boxes = torch.tensor([[0.3, 0.4, 0.1, 0.2]], dtype=torch.float32)
    assert torch.allclose(output_boxes, expected_boxes)

def test_translator_normalized_xyxy_to_pixel_xyxy():
    """Test converting normalized XYXY to pixel XYXY."""
    translator = TorchBoxCoordinatesTranslator(
        input_box_type=BoxType(BoxFormat.XYXY, is_normalized=True),
        output_box_type=BoxType(BoxFormat.XYXY, is_normalized=False)
    )
    input_boxes = torch.tensor([[0.25, 0.25, 0.5, 0.5]], dtype=torch.float32)
    image_size = torch.Size([400, 400])
    output_boxes = translator.translate(input_boxes, image_size)
    # Normalized [0.25, 0.25, 0.5, 0.5] with image 400x400 -> pixel [100, 100, 200, 200]
    expected_boxes = torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)
    assert torch.allclose(output_boxes, expected_boxes)
