import os
import pprint

import pytest
import numpy as np
from functools import partial
import torch
import tensorflow as tf
from torchvision.transforms.functional import to_pil_image, to_tensor, resize
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

import xplique
from xplique.attributions import Saliency
from xplique.attributions.gradient_input import GradientInput
from xplique.plots import plot_attributions
from xplique.utils_functions.object_detection.torch.gradients_check import check_model_gradients
from xplique.utils_functions.object_detection.base.box_manager import BoxFormat
from xplique.utils_functions.object_detection.torch.box_manager import TorchBoxManager
from xplique.concepts.torch.latent_data_detr import LatentDataDetr, DetrExtractorBuilder
from xplique.concepts import HolisticCraftTorch as Craft

from xplique.plots.display_image_with_boxes import display_image_with_boxes
from xplique.wrappers import TorchWrapper

pp = pprint.PrettyPrinter(indent=4) # Set Pretty Print Indentation
print(torch.__version__) # Check the version of torch used

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
test_dir = os.path.dirname(os.path.abspath(__file__))

@pytest.fixture(scope="session", params=["cpu", "cuda"])
def device_param(request):
    device_str = request.param
    if device_str == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device(device_str)
    print(f"Using device: {device}")
    return device

@pytest.fixture(scope="session")
def image_data(device_param):
    # Check if 'img.jpg' exists, if not, download it
    if not os.path.exists("img.jpg"):
        print("File 'img.jpg' not found. Downloading...")
        os.system('wget -O img.jpg "https://unsplash.com/photos/MXvcHk-zCIs/download?force=true&w=640"')

    # Load and preprocess the image
    raw_image = Image.open("img.jpg")

    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.Resize((800, 800)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # preprocess and batch the image
    input_tensor = transform(raw_image).unsqueeze(0).to(device_param)

    return raw_image, input_tensor

def test_image_size(image_data):
    image, _ = image_data
    expected_size = (640, 462)
    assert image.size == expected_size


@pytest.fixture(scope="session")
def model_data(image_data, device_param):
    _, input_tensor = image_data
    model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True).to(device_param)
    model.eval()
    processed_results = model(input_tensor)
    return model, processed_results

def test_model_outputs(model_data):
    _, processed_results = model_data
    assert list(processed_results.keys()) == ['pred_logits', 'pred_boxes']

def test_gradients_model_original(image_data, model_data):
    image, input_tensor = image_data
    model, _ = model_data
    check = check_model_gradients(model, input_tensor)
    assert check == False, "Model gradients should be not computed successfully when calling the default Detr model."

def test_multibox_tensor_to_batched():
    from xplique.utils_functions.object_detection.torch.multi_box_tensor import MultiBoxTensor
    
    # Create a mock MultiBoxTensor with shape (num_boxes, features)
    tensor_data = torch.rand(5, 25)  # 5 boxes, 25 features (4+1+20)
    nbc = MultiBoxTensor(tensor_data)
    
    # Test to_batched_tensor()
    batched = nbc.to_batched_tensor()
    assert isinstance(batched, torch.Tensor)
    assert batched.shape == (1, 5, 25)  # (1, num_boxes, features)
    assert torch.allclose(batched[0], tensor_data)

def test_box_model_wrapper():
    from xplique.utils_functions.object_detection.torch.box_model_wrapper import TorchBoxesModelWrapper
    from xplique.utils_functions.object_detection.torch.box_formatter import TorchvisionBoxFormatter
    
    # Create a simple mock model that returns a list of predictions
    class MockTorchvisionModel(torch.nn.Module):
        def forward(self, x):
            batch_size = x.shape[0]
            # Return list of dicts (Torchvision format)
            predictions = []
            for _ in range(batch_size):
                pred = {
                    'boxes': torch.rand(5, 4),  # 5 boxes, 4 coords
                    'scores': torch.rand(5),     # 5 scores
                    'labels': torch.randint(0, 20, (5,))  # 5 labels (20 classes)
                }
                predictions.append(pred)
            return predictions
    
    model = MockTorchvisionModel()
    formatter = TorchvisionBoxFormatter(nb_classes=20)
    wrapper = TorchBoxesModelWrapper(model, formatter)
    
    # Test with batch_size=2
    input_tensor = torch.randn(2, 3, 224, 224)
    
    # Test output_as_list mode (default)
    assert wrapper.output_as_list is True
    output_list = wrapper(input_tensor)
    assert isinstance(output_list, list)
    assert len(output_list) == 2
    
    # Test output_as_tensor mode
    wrapper.set_output_as_tensor()
    assert wrapper.output_as_list is False
    output_tensor = wrapper(input_tensor)
    assert isinstance(output_tensor, torch.Tensor)
    assert output_tensor.shape == (2, 5, 25)  # (batch, num_boxes, features)

@pytest.fixture(scope="session")
def dataset_classes():
    # COCO classes
    CLASSES = [
        'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
        'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
        'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    nb_classes = len(CLASSES)
    label_to_color = {'person': 'r',
                'bicycle': 'b',
                'car': 'g',
                'motorcycle': 'y',
                'truck': 'orange'}
    return CLASSES, nb_classes, label_to_color

@pytest.fixture(scope="session")
def latent_extractor_data(dataset_classes, model_data, device_param):
    classes_names, nb_classes, label_to_color = dataset_classes
    model, _ = model_data

    latent_extractor = DetrExtractorBuilder.build(model, device=str(device_param))
    return latent_extractor

def test_latent_extractor(image_data, dataset_classes, latent_extractor_data):
    image, input_tensor = image_data
    classes_names, nb_classes, label_to_color = dataset_classes
    latent_extractor = latent_extractor_data

    results = latent_extractor(input_tensor)
    print("Latent Data Detr:", results)
    assert isinstance(results, list), "Results should be a list of MultiBoxTensor objects"
    assert len(results) == 1, "Should have one result per batch item"
    assert results[0].shape == torch.Size([100, 96]), "MultiBoxTensor shape should be [100, 96]."

    filtered_results = results[0].filter(accuracy=0.85)
    box_manager = TorchBoxManager(BoxFormat.XYXY, normalized=True)
    display_image_with_boxes(image, filtered_results, box_manager, classes_names, label_to_color)

def test_latent_extractor_gradients(image_data, latent_extractor_data):
    image, input_tensor = image_data
    latent_extractor = latent_extractor_data

    check = check_model_gradients(latent_extractor, input_tensor)
    assert check, "Latent extractor gradients should be computed successfully."

def test_latent_extractor_saliency(image_data, dataset_classes, latent_extractor_data, device_param):
    image, input_tensor = image_data
    classes_names, nb_classes, label_to_color = dataset_classes
    latent_extractor = latent_extractor_data

    targets = latent_extractor(input_tensor)
    filtered_targets = targets[0].filter(accuracy=0.9, class_id=classes_names.index('car'))
    box_to_explain = np.expand_dims(filtered_targets.detach().cpu().numpy(), axis=0)

    latent_extractor.set_output_as_tensor()
    input_tensor_tf_dim = input_tensor.detach().cpu().numpy().transpose(0, 2, 3, 1)
    
    torch_wrapped_model = TorchWrapper(latent_extractor, device=device_param, is_channel_first=True)
    
    explainer = Saliency(torch_wrapped_model, operator=xplique.Tasks.OBJECT_DETECTION, batch_size=1)
    explanation = explainer.explain(input_tensor_tf_dim, targets=box_to_explain)
 
    plot_attributions(explanation, [np.array(image)], img_size=6.,
                    cmap='jet', alpha=0.3, absolute_value=False, clip_percentile=0.5)

@pytest.fixture(scope="session")
def craft_data(image_data, latent_extractor_data, device_param):
    image, input_tensor = image_data
    latent_extractor = latent_extractor_data

    craft = Craft(latent_extractor = latent_extractor,
                number_of_concepts = 10,
                device = str(device_param))
    craft.fit(input_tensor)
    return craft

def test_craft_reencode(image_data, dataset_classes, craft_data):
    image, input_tensor = image_data
    classes_names, nb_classes, label_to_color = dataset_classes
    craft = craft_data

    # The encode method now returns a list of tuples [(latent_data, coeffs_u), ...]
    latent_data_coeffs_u_list = craft.encode(input_tensor)
    
    # Should have one tuple per image in the batch
    assert len(latent_data_coeffs_u_list) == input_tensor.shape[0], f"Expected {input_tensor.shape[0]} tuples, got {len(latent_data_coeffs_u_list)}"
    
    # Get the first tuple (since we're only testing with one image)
    latent_data, coeffs_u = latent_data_coeffs_u_list[0]
    
    print(coeffs_u.shape)
    assert coeffs_u.shape == (1, 25, 25, 10), "Latent data shape should be (1, 25, 25, 10)."

    result = craft.decode(latent_data, coeffs_u)
    from xplique.utils_functions.object_detection.torch.multi_box_tensor import MultiBoxTensor
    assert isinstance(result, MultiBoxTensor), "Decoded result should be an MultiBoxTensor directly"
    print(result.shape)
    assert result.shape == torch.Size([100, 96]), "Decoded MultiBoxTensor shape should be [100, 96]."

    filtered_result = result.filter(accuracy=0.85)
    box_manager = TorchBoxManager(BoxFormat.XYXY, normalized=True)
    display_image_with_boxes(image, filtered_result, box_manager, classes_names, label_to_color)

def test_craft_decoder_modes(image_data, dataset_classes, craft_data, device_param):
    image, input_tensor = image_data
    classes_names, nb_classes, label_to_color = dataset_classes
    craft = craft_data

    encoded_data = craft.encode(input_tensor)
    latent_data, coeffs_u = encoded_data[0]
    decoder = craft.make_concept_decoder(latent_data)
    
    # Decoder should always return tensor (unified behavior with TensorFlow)
    output_tensor = decoder(coeffs_u)
    assert hasattr(output_tensor, 'shape'), "Decoder should always return a tensor"
    assert output_tensor.shape == torch.Size([1, 100, 96]), f"Expected shape [1, 100, 96], got {output_tensor.shape}"

    # For filtering, use decode directly to get MultiBoxTensor
    nbc_tensor = craft.decode(latent_data, coeffs_u)
    assert hasattr(nbc_tensor, 'filter'), "decode should return MultiBoxTensor with filter method"


def test_craft_concepts(image_data, craft_data):
    image, input_tensor = image_data
    craft = craft_data
    craft.display_images_per_concept(input_tensor, order=None, filter_percentile=80, clip_percentile=5)

def test_craft_gradient_input(image_data, dataset_classes, craft_data):
    image, input_tensor = image_data
    classes_names, nb_classes, label_to_color = dataset_classes
    craft = craft_data

    # Test compute_gradient_input
    operator = xplique.Tasks.OBJECT_DETECTION
    class_id = classes_names.index("person")
    explainer_partial = partial(
                GradientInput,
                operator=operator,
                harmonize=False,
            )
    explanation = craft.compute_explanation_per_concept(input_tensor, class_id=class_id, accuracy=0.6, explainer_partial=explainer_partial)

    # Verify explanation shape
    assert explanation.shape[0] == 1, "Should have one explanation per image"
    assert explanation.shape[1:3] == (25, 25), "Should match coeffs_u spatial dimensions"
    assert explanation.shape[3] == 10, "Should match number of concepts"
    
    importances_gi = craft.estimate_importance(input_tensor, operator, class_id, accuracy=0.6, method='gradient_input')

    order = importances_gi.argsort()[::-1]
    print(f"Concept importances order for 'person': {order}")
    assert np.all(order == np.array([2, 6, 5, 4, 7, 9, 8, 0, 3, 1])), "Concepts order is not as expected"
    craft.display_images_per_concept(input_tensor, order=order, filter_percentile=80, clip_percentile=5)

def test_craft_encode_differentiable_gradients(image_data, craft_data):
    """Test that encode(differentiable=True) preserves gradients through the encoding pipeline."""
    image, input_tensor = image_data
    craft = craft_data
    
    # Ensure input has gradients enabled
    input_with_grad = input_tensor.clone().detach().requires_grad_(True)
    
    # Test differentiable encoding
    encoded_data = craft.encode(input_with_grad, differentiable=True)
    latent_data, coeffs_u = encoded_data[0]
    
    # Verify coeffs_u is a tensor with gradients
    assert isinstance(coeffs_u, torch.Tensor), "coeffs_u should be a torch.Tensor in differentiable mode"
    assert coeffs_u.requires_grad, "coeffs_u should have gradients enabled"
    
    # Create a simple loss and check gradient flow
    loss = coeffs_u.sum()
    loss.backward()
    gradients = input_with_grad.grad

    # Verify gradients flowed back to input
    assert gradients is not None, "Gradients should flow back to input"
    assert gradients.abs().sum() > 0, "Gradients should be non-zero"
