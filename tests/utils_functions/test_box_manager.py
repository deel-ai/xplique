import torch
from xplique.utils_functions.object_detection.common.box_manager import BoxManager, BoxFormat
from xplique.utils_functions.object_detection.torch.box_manager import TorchBoxManager

def test_normalize_boxes():
    raw_boxes = torch.tensor([[50, 50, 100, 100], [30, 30, 60, 60]], dtype=torch.float32)
    image_source_size = torch.Size([200, 200])
    normalized_boxes = TorchBoxManager.normalize_boxes(raw_boxes.clone(), image_source_size)
    expected_boxes = torch.tensor([[0.25, 0.25, 0.5, 0.5], [0.15, 0.15, 0.3, 0.3]], dtype=torch.float32)
    assert torch.allclose(normalized_boxes, expected_boxes)

def test_box_cxcywh_to_xyxy():
    normalized_boxes = torch.tensor([[50, 50, 20, 20], [0.3, 0.3, 0.1, 0.1]], dtype=torch.float32)
    converted_boxes = TorchBoxManager.box_cxcywh_to_xyxy(normalized_boxes)
    expected_boxes = torch.tensor([[40, 40, 60, 60], [0.25, 0.25, 0.35, 0.35]], dtype=torch.float32)
    assert torch.allclose(converted_boxes, expected_boxes)

def test_rescale_bboxes():
    normalized_boxes = torch.tensor([[0.25, 0.25, 0.5, 0.5], [0.15, 0.15, 0.3, 0.3]], dtype=torch.float32)
    image_target_size = torch.Size([400, 400])
    rescaled_boxes = TorchBoxManager.rescale_bboxes(normalized_boxes, image_target_size)
    expected_boxes = torch.tensor([[100, 100, 200, 200], [60, 60, 120, 120]], dtype=torch.float32)
    assert torch.allclose(rescaled_boxes, expected_boxes)

# Tests of resize() for specific model types (Detr, Fcos)

def test_resize_Detr():
    # DETR boxes are in CXCYWH format, already normalized -> will not be normalized
    box_manager = TorchBoxManager(format=BoxFormat.CXCYWH, normalize=False)
    raw_boxes = torch.tensor([[0.3, 0.4, 0.1, 0.2]], dtype=torch.float32)
    image_source_size = torch.Size([100, 100])
    image_target_size = torch.Size([200, 200])
    resized_boxes = box_manager.resize(raw_boxes.clone(), image_source_size, image_target_size)
    expected_boxes = torch.tensor([[ 50,  60,  70, 100]], dtype=torch.float32)
    assert torch.allclose(resized_boxes, expected_boxes)

def test_resize_Fcos():
    # FCOS boxes are in XYXY format, not normalized -> will be normalized
    box_manager = TorchBoxManager(format=BoxFormat.XYXY, normalize=True)
    raw_boxes = torch.tensor([[50, 50, 100, 100], [30, 30, 60, 60]], dtype=torch.float32)
    image_source_size = torch.Size([100, 100])
    image_target_size = torch.Size([200, 200])
    resized_boxes = box_manager.resize(raw_boxes.clone(), image_source_size, image_target_size)
    expected_boxes = torch.tensor([[100, 100, 200, 200], [ 60,  60, 120, 120]], dtype=torch.float32)
    assert torch.allclose(resized_boxes, expected_boxes)
