"""Tests for HolisticCraftTorch on object detection tasks with PyTorch."""
# pylint: disable=unused-variable,unused-argument,redefined-outer-name
import os
import pprint
from functools import partial

import numpy as np
import pytest
import torch
from PIL import Image
import torchvision.transforms as T

import xplique
from xplique.attributions import Saliency
from xplique.attributions.gradient_input import GradientInput
from xplique.plots import plot_attributions
from xplique.utils_functions.common.torch.gradients_check import check_model_gradients
from xplique.utils_functions.object_detection.base.box_manager import BoxFormat, BoxType
from xplique.utils_functions.object_detection.torch.box_manager import TorchBoxManager
from xplique.utils_functions.object_detection.torch.multi_box_tensor import MultiBoxTensor
from xplique.utils_functions.object_detection.torch.box_model_wrapper import (
    TorchBoxesModelWrapper
)
from xplique.concepts import HolisticCraftTorch as Craft
from xplique.concepts.torch.latent_extractor import TorchLatentExtractor

from xplique.plots.display_image_with_boxes import display_image_with_boxes
from xplique.wrappers import TorchWrapper
from xplique.concepts.latent_extractor import LatentData

pp = pprint.PrettyPrinter(indent=4) # Set Pretty Print Indentation
print(torch.__version__) # Check the version of torch used

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
test_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# Mock Classes - Shared across tests
# ============================================================================

class MockTorchvisionModel(torch.nn.Module):
    """Mock Torchvision-style model that returns list of predictions with gradient flow."""
    def forward(self, x):
        """Build method for the mock extractor builder."""
        batch_size = x.shape[0]
        device = x.device

        # Create a simple linear layer to ensure gradient flow from input
        # Use mean pooling to reduce spatial dimensions
        pooled = torch.mean(x, dim=[2, 3])  # (batch, channels)

        # Return list of dicts (Torchvision format)
        predictions = []
        for i in range(batch_size):
            # Create predictions that depend on input via pooled features
            # Use pooled features to create a scaling factor
            pooled_mean = pooled[i].mean().abs() + 1.0  # Add 1.0 to ensure decent scale

            # boxes: modulated by input features
            boxes_base = torch.rand(5, 4, device=device)
            boxes = boxes_base * pooled_mean * 100  # Scale to reasonable box sizes

            # scores: ensure some boxes have high confidence for filtering tests
            # First 2 boxes have high confidence (>0.9), rest have lower
            # Multiply by a small gradient from pooled to maintain gradient flow
            gradient_factor = (pooled_mean - 1.0) * 0.01  # Small variation from input
            scores = torch.tensor([0.95, 0.92, 0.5, 0.3, 0.1], device=device) + gradient_factor
            scores = torch.clamp(scores, 0.0, 1.0)  # Ensure valid range

            # labels: ensure first box is 'car' (index 3 in COCO) for saliency test filtering
            # This ensures at least one high-confidence box has the 'car' class
            labels = torch.randint(0, 20, (5,), device=device)
            labels[0] = 3  # 'car' class (COCO index)

            pred = {
                'boxes': boxes,
                'scores': scores,
                'labels': labels
            }
            predictions.append(pred)
        return predictions


class MockTorchvisionBoxFormatter:
    """Mock formatter for Torchvision-style predictions."""
    def __init__(self, nb_classes=20):
        self.nb_classes = nb_classes
        self.input_box_type = BoxType(BoxFormat.XYXY, is_normalized=False)
        self.output_box_type = BoxType(BoxFormat.XYXY, is_normalized=False)

    def __call__(self, model_outputs):
        """Make formatter callable."""
        return self.format_outputs(model_outputs)

    def format_outputs(self, model_outputs):
        """Convert Torchvision outputs (list of dicts) to list of MultiBoxTensor."""
        results = []

        for pred in model_outputs:
            boxes = pred['boxes']  # (num_boxes, 4)
            scores = pred['scores']  # (num_boxes,)
            labels = pred['labels']  # (num_boxes,)

            # Create one-hot encoded class probabilities on the same device as boxes
            num_boxes = boxes.shape[0]
            device = boxes.device
            class_probs = torch.zeros(num_boxes, self.nb_classes, device=device)
            class_probs.scatter_(1, labels.unsqueeze(1), 1.0)
            class_probs = class_probs * scores.unsqueeze(1)

            # Concatenate: [boxes (4), confidence (1), class_probs (nb_classes)]
            combined = torch.cat([
                boxes,
                scores.unsqueeze(1),
                class_probs
            ], dim=1)

            results.append(MultiBoxTensor(combined))

        return results

class MockTorchvisionLatentData(LatentData):
    """Mock latent data with fixed reproducible activations in (N, H, W, C) format."""

    def __init__(self, batch_size=1, device='cpu'):
        """
        Parameters
        ----------
        batch_size : int
            Batch size
        device : str or torch.device
            Device for tensors
        """
        self.batch_size = batch_size
        self.device = device
        # Fixed spatial dimensions for reproducibility
        self.height = 10
        self.width = 10
        self.channels = 64

        # Create fixed reproducible activations (N, H, W, C)
        # Use ReLU-activated values (non-negative for NMF)
        torch.manual_seed(42)
        activations = torch.randn(batch_size, self.height, self.width, self.channels, device=device)
        self.activations = torch.relu(activations)

    def __len__(self):
        """Return batch size."""
        return self.batch_size

    def detach(self):
        """Detach all tensors from computation graph."""
        self.activations = self.activations.detach()
        return self

    def get_activations(self, as_numpy=True, keep_gradients=False):
        """
        Get activations in (N, H, W, C) format.

        Returns
        -------
        activations : np.ndarray or torch.Tensor
            Shape (batch, H, W, C)
        """
        activations = self.activations

        if not keep_gradients:
            activations = activations.detach()

        if as_numpy:
            activations = activations.cpu().numpy()

        return activations

    def set_activations(self, values):
        """
        Set activations from (N, H, W, C) format.

        Parameters
        ----------
        values : np.ndarray or torch.Tensor
            Shape (batch, H, W, C)
        """
        # Convert numpy to tensor if needed
        if isinstance(values, np.ndarray):
            values = torch.from_numpy(values).to(self.device)

        self.activations = values

    def to(self, device):
        """Move all data to specified device."""
        self.device = device
        self.activations = self.activations.to(device)
        return self


class MockExtractorBuilder:
    """
    Builder for creating LatentExtractor instances for mock Torchvision-style models.

    This class encapsulates the logic for creating a TorchLatentExtractor configured
    for mock object detection models that follow the Torchvision output format.
    """

    @classmethod
    def build(cls, model, device='cpu', batch_size=1):
        """
        Build a LatentExtractor for a mock Torchvision-style model.

        Creates custom input_to_latent and latent_to_logit functions that split
        the model's forward pass into feature extraction and prediction.

        Parameters
        ----------
        model : torch.nn.Module
            Mock Torchvision-style object detection model
        device : str or torch.device
            Device to run computations on. Default is 'cpu'.
        batch_size : int
            Batch size for processing. Default is 1.

        Returns
        -------
        latent_extractor : TorchLatentExtractor
            Configured TorchLatentExtractor instance for the mock model.
        """
        def input_to_latent_fn(inputs):
            """Simple mock latent extractor - just pool input to fixed size."""
            batch_size = inputs.shape[0]
            device = inputs.device

            # Pool to fixed spatial size (10x10) and apply ReLU for NMF
            pooled = torch.nn.functional.adaptive_avg_pool2d(inputs, (10, 10))  # (N, C, 10, 10)
            activations = torch.relu(pooled.permute(0, 2, 3, 1))  # (N, 10, 10, C) with ReLU

            # Pad/truncate channels to 64 for consistency
            if activations.shape[-1] != 64:
                activations = torch.nn.functional.pad(activations, (0, 64 - activations.shape[-1]))

            latent_data = MockTorchvisionLatentData(batch_size=batch_size, device=device)
            latent_data.set_activations(activations)
            return latent_data

        def latent_to_logit_fn(latent_data):
            """Simple mock decoder - reuse the existing MockTorchvisionModel logic."""
            activations = latent_data.get_activations(as_numpy=False, keep_gradients=True)
            # Convert back to channel-first for model: (N, 10, 10, C) -> (N, C, 10, 10)
            inputs_chw = activations.permute(0, 3, 1, 2)
            return model(inputs_chw)

        # Create mock Torchvision formatter
        formatter = MockTorchvisionBoxFormatter(nb_classes=20)

        # Create TorchLatentExtractor with simplified mock functions
        latent_extractor = TorchLatentExtractor(
            model=model,
            input_to_latent_model=input_to_latent_fn,
            latent_to_logit_model=latent_to_logit_fn,
            latent_data_class=MockTorchvisionLatentData,
            output_formatter=formatter,
            batch_size=batch_size,
            device=str(device)
        )

        return latent_extractor


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture(scope="session", params=["cpu", "cuda"])
def device_param(request):
    """Fixture providing the device parameter for tests."""
    device_str = request.param
    if device_str == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device(device_str)
    print(f"Using device: {device}")
    return device

@pytest.fixture(scope="session")
def image_data(device_param):
    """Fixture providing image data for tests."""
    # Check if 'img.jpg' exists, if not, download it
    if not os.path.exists("img.jpg"):
        print("File 'img.jpg' not found. Downloading...")
        url = "https://unsplash.com/photos/MXvcHk-zCIs/download?force=true&w=640"
        os.system(f'wget -O img.jpg "{url}"')

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
    """Fixture providing a mock model for tests."""
    image, _ = image_data
    expected_size = (640, 462)
    assert image.size == expected_size


@pytest.fixture(scope="session")
def model_data(image_data, device_param):
    """Create a mock Torchvision-style object detection model."""
    _, input_tensor = image_data

    model = MockTorchvisionModel().to(device_param)
    model.eval()
    processed_results = model(input_tensor)
    return model, processed_results

def test_model_outputs(model_data):
    """Fixture providing model data including model, formatter, and device."""
    _, processed_results = model_data
    # Torchvision format: list of dicts with 'boxes', 'scores', 'labels'
    assert isinstance(processed_results, list)
    assert len(processed_results) > 0
    assert 'boxes' in processed_results[0]
    assert 'scores' in processed_results[0]
    assert 'labels' in processed_results[0]

def test_gradients_model_original(image_data, model_data):
    """Fixture providing a mock latent extractor for tests."""
    image, input_tensor = image_data
    model, _ = model_data
    check = check_model_gradients(model, input_tensor)
    assert check, "Mock model gradients should be computed successfully."

def test_multibox_tensor_to_batched():
    """Test MultiBoxTensor to_batched_tensor method."""
    # Create a mock MultiBoxTensor with shape (num_boxes, features)
    tensor_data = torch.rand(5, 25)  # 5 boxes, 25 features (4+1+20)
    nbc = MultiBoxTensor(tensor_data)

    # Test to_batched_tensor()
    batched = nbc.to_batched_tensor()
    assert isinstance(batched, torch.Tensor)
    assert batched.shape == (1, 5, 25)  # (1, num_boxes, features)
    assert torch.allclose(batched[0], tensor_data)

def test_box_model_wrapper():
    """Test TorchBoxesModelWrapper initialization and functionality."""
    model = MockTorchvisionModel()
    formatter = MockTorchvisionBoxFormatter(nb_classes=20)
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
    """Fixture providing COCO dataset classes and metadata."""
    # COCO classes
    classes = [
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
    nb_classes = len(classes)
    label_to_color = {'person': 'r',
                'bicycle': 'b',
                'car': 'g',
                'motorcycle': 'y',
                'truck': 'orange'}
    return classes, nb_classes, label_to_color

@pytest.fixture(scope="session")
def latent_extractor_data(dataset_classes, model_data, device_param):
    """Create latent extractor using MockExtractorBuilder."""
    classes_names, nb_classes, label_to_color = dataset_classes
    model, _ = model_data

    # Use the builder to create the latent extractor
    latent_extractor = MockExtractorBuilder.build(
        model=model,
        device=str(device_param),
        batch_size=1
    )

    return latent_extractor

def test_latent_extractor(image_data, dataset_classes, latent_extractor_data):
    """Fixture providing latent extractor with model data."""
    image, input_tensor = image_data
    classes_names, nb_classes, label_to_color = dataset_classes
    latent_extractor = latent_extractor_data

    results = latent_extractor(input_tensor)
    print("Latent Data Torchvision:", results)
    assert isinstance(results, list), "Results should be a list of MultiBoxTensor objects"
    assert len(results) == 1, "Should have one result per batch item"
    # Torchvision model: 5 boxes, each with 4+1+20 features (boxes, confidence, class_probs)
    expected_shape = torch.Size([5, 25])
    error_msg = (f"MultiBoxTensor shape should be {expected_shape}, "
                 f"got {results[0].shape}")
    assert results[0].shape == expected_shape, error_msg

    filtered_results = results[0].filter(confidence=0.5)
    box_manager = TorchBoxManager(BoxFormat.XYXY, normalized=False)
    display_image_with_boxes(image, filtered_results, box_manager,
                             classes_names, label_to_color)

def test_latent_extractor_gradients(image_data, latent_extractor_data):
    """Fixture providing crop data for tests."""
    image, input_tensor = image_data
    latent_extractor = latent_extractor_data

    check = check_model_gradients(latent_extractor, input_tensor)
    assert check, "Latent extractor gradients should be computed successfully."

def test_latent_extractor_saliency(image_data, dataset_classes,
                                   latent_extractor_data, device_param):
    """Test latent extractor saliency computation."""
    image, input_tensor = image_data
    classes_names, nb_classes, label_to_color = dataset_classes
    latent_extractor = latent_extractor_data

    targets = latent_extractor(input_tensor)
    class_id_car = classes_names.index('car')
    filtered_targets = targets[0].filter(confidence=0.9, class_id=class_id_car)
    box_to_explain = np.expand_dims(filtered_targets.detach().cpu().numpy(), axis=0)

    latent_extractor.set_output_as_tensor()
    input_tensor_tf_dim = input_tensor.detach().cpu().numpy().transpose(0, 2, 3, 1)

    torch_wrapped_model = TorchWrapper(latent_extractor, device=device_param,
                                       is_channel_first=True)

    operator = xplique.Tasks.OBJECT_DETECTION
    explainer = Saliency(torch_wrapped_model, operator=operator, batch_size=1)
    explanation = explainer.explain(input_tensor_tf_dim, targets=box_to_explain)

    plot_attributions(explanation, [np.array(image)], img_size=6.,
                      cmap='jet', alpha=0.3, absolute_value=False,
                      clip_percentile=0.5)

@pytest.fixture(scope="session")
def craft_data(image_data, latent_extractor_data, device_param):
    """Fixture providing CRAFT instance for tests."""
    image, input_tensor = image_data
    latent_extractor = latent_extractor_data

    craft = Craft(latent_extractor = latent_extractor,
                number_of_concepts = 10,
                device = str(device_param))
    craft.fit(input_tensor)
    return craft

def test_craft_reencode(image_data, dataset_classes, craft_data):
    """Fixture providing crop selection data for tests."""
    image, input_tensor = image_data
    classes_names, nb_classes, label_to_color = dataset_classes
    craft = craft_data

    # The encode method now returns a list of tuples [(latent_data, coeffs_u), ...]
    latent_data_coeffs_u_list = craft.encode(input_tensor)

    # Should have one tuple per image in the batch
    expected_len = input_tensor.shape[0]
    actual_len = len(latent_data_coeffs_u_list)
    error_msg = f"Expected {expected_len} tuples, got {actual_len}"
    assert len(latent_data_coeffs_u_list) == expected_len, error_msg

    # Get the first tuple (since we're only testing with one image)
    latent_data, coeffs_u = latent_data_coeffs_u_list[0]

    print(coeffs_u.shape)
    # Torchvision mock uses spatial feature maps (H=10, W=10) with 10 concepts
    expected_shape = (1, 10, 10, 10)
    error_msg = f"Latent data shape should be {expected_shape}, got {coeffs_u.shape}"
    assert coeffs_u.shape == expected_shape, error_msg

    result = craft.decode(latent_data, coeffs_u)
    assert isinstance(result, MultiBoxTensor), "Decoded result should be an MultiBoxTensor directly"
    print(result.shape)
    # Result should have 5 boxes, 25 features each (4 box coords + 1 confidence + 20 class probs)
    expected_shape = torch.Size([5, 25])
    error_msg = f"Decoded MultiBoxTensor shape should be [5, 25], got {result.shape}"
    assert result.shape == expected_shape, error_msg

    filtered_result = result.filter(confidence=0.5)
    box_manager = TorchBoxManager(BoxFormat.XYXY, normalized=False)
    display_image_with_boxes(image, filtered_result, box_manager,
                             classes_names, label_to_color)

def test_craft_decoder_modes(image_data, dataset_classes, craft_data, device_param):
    """Test CRAFT importance estimation functionality."""
    image, input_tensor = image_data
    classes_names, nb_classes, label_to_color = dataset_classes
    craft = craft_data

    encoded_data = craft.encode(input_tensor)
    latent_data, coeffs_u = encoded_data[0]
    decoder = craft.make_concept_decoder(latent_data)

    # Decoder should always return tensor (unified behavior with TensorFlow)
    output_tensor = decoder(coeffs_u)
    assert hasattr(output_tensor, 'shape'), "Decoder should always return a tensor"
    # Torchvision model: (1, num_boxes, features) = (1, 5, 25)
    expected_shape = torch.Size([1, 5, 25])
    error_msg = f"Expected shape {expected_shape}, got {output_tensor.shape}"
    assert output_tensor.shape == expected_shape, error_msg

    # For filtering, use decode directly to get MultiBoxTensor
    nbc_tensor = craft.decode(latent_data, coeffs_u)
    error_msg = "decode should return MultiBoxTensor with filter method"
    assert hasattr(nbc_tensor, 'filter'), error_msg


def test_craft_concepts(image_data, craft_data):
    """Test CRAFT patch counting functionality."""
    image, input_tensor = image_data
    craft = craft_data
    craft.display_images_per_concept(input_tensor, order=None,
                                     filter_percentile=80, clip_percentile=5)

def test_craft_gradient_input(image_data, dataset_classes, craft_data):
    """Test CRAFT display functionality."""
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
    explanation = craft.compute_explanation_per_concept(
        input_tensor, class_id=class_id, confidence=0.6,
        explainer_partial=explainer_partial
    )

    # Verify explanation shape - Torchvision mock produces 4D explanations
    # Shape: (batch, H, W, num_concepts) = (1, 10, 10, 10)
    assert explanation.shape[0] == 1, "Should have one explanation per image"
    assert explanation.shape[1] == 10, "Should match feature map height"
    assert explanation.shape[2] == 10, "Should match feature map width"
    assert explanation.shape[3] == 10, "Should match number of concepts"

    importances_gi = craft.estimate_importance(
        input_tensor, operator, class_id, confidence=0.6, method='gradient_input'
    )

    # Verify importances are computed and have correct shape
    assert importances_gi.shape[0] == 10, "Should have importance for each concept"
    assert np.all(np.isfinite(importances_gi)), "All importances should be finite"

    order = importances_gi.argsort()[::-1]
    print(f"Concept importances order for 'person': {order}")

    # For mock data, we just verify we got a valid ordering (not checking specific order)
    # Real Torchvision models would produce different orderings based on actual features
    assert len(order) == 10, "Should have ordered all 10 concepts"
    assert len(np.unique(order)) == 10, "All concepts should have unique ordering"
    # craft.display_images_per_concept(input_tensor, order=order,
    #                                   filter_percentile=80, clip_percentile=5)

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
    expected_type_msg = "coeffs_u should be a torch.Tensor in differentiable mode"
    assert isinstance(coeffs_u, torch.Tensor), expected_type_msg
    assert coeffs_u.requires_grad, "coeffs_u should have gradients enabled"

    # Create a simple loss and check gradient flow
    loss = coeffs_u.sum()
    loss.backward()
    gradients = input_with_grad.grad

    # Verify gradients flowed back to input
    assert gradients is not None, "Gradients should flow back to input"
    assert gradients.abs().sum() > 0, "Gradients should be non-zero"
