"""Tests for HolisticCraftTorch on classification tasks with PyTorch."""
# pylint: disable=redefined-outer-name
import os
import pprint
from functools import partial

import pytest
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision import models

import xplique
from xplique.attributions import Saliency
from xplique.attributions.gradient_input import GradientInput
from xplique.concepts.torch.latent_data_layered_model import LayeredModelExtractorBuilder
from xplique.concepts import HolisticCraftTorch as Craft
from xplique.utils_functions.classification.torch.classifier_tensor import ClassifierTensor
from xplique.utils_functions.common.torch.gradients_check import check_model_gradients
from xplique.wrappers import TorchWrapper

pp = pprint.PrettyPrinter(indent=4)  # Set Pretty Print Indentation
print(torch.__version__)  # Check the version of torch used

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
test_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(params=["cpu", "cuda"])
def device_param(request):
    """Pytest fixture to provide device parameter (cpu or cuda)."""
    device_str = request.param
    if device_str == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device(device_str)
    print(f"Using device: {device}")
    return device


@pytest.fixture(scope="function")
def image_data(device_param):
    """Pytest fixture to load and preprocess test image data."""
    # Check if 'img.jpg' exists, if not, download it
    if not os.path.exists("img.jpg"):
        print("File 'img.jpg' not found. Downloading...")
        url = 'https://unsplash.com/photos/MXvcHk-zCIs/download?force=true&w=640'
        os.system(f'wget -O img.jpg "{url}"')

    # Load and preprocess the image
    raw_image = Image.open("img.jpg")

    # Standard ImageNet normalization
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Preprocess and batch the image
    input_tensor = transform(raw_image).unsqueeze(0).to(device_param)

    return raw_image, input_tensor


def test_image_size(image_data):
    """Test that loaded image has expected dimensions."""
    image, _ = image_data
    expected_size = (640, 462)
    assert image.size == expected_size


@pytest.fixture(scope="function")
def model_data(image_data, device_param):
    """Pytest fixture to load pretrained ResNet50 model and run predictions."""
    _, input_tensor = image_data
    # Load pretrained ResNet50
    model = models.resnet50(pretrained=True).to(device_param)
    model.eval()

    with torch.no_grad():
        predictions = model(input_tensor)

    return model, predictions


def test_model_outputs(model_data):
    """Test that model outputs have expected shape for ImageNet classes."""
    _, predictions = model_data
    # ResNet50 outputs 1000 ImageNet classes
    assert predictions.shape == (1, 1000), f"Expected shape (1, 1000), got {predictions.shape}"


@pytest.fixture(scope="module")
def imagenet_classes():
    """Pytest fixture providing ImageNet class labels for testing."""
    # Top-5 most common ImageNet classes for testing
    # In a real scenario, you would load all 1000 classes
    classes = [
        'tench', 'goldfish', 'great_white_shark', 'tiger_shark', 'hammerhead',
        'electric_ray', 'stingray', 'cock', 'hen', 'ostrich'
    ]
    return classes


@pytest.fixture(scope="function")
def latent_extractor_data(model_data, device_param):
    """Pytest fixture to create latent extractor from ResNet50 model."""
    model, _ = model_data

    # Split ResNet50 at layer 4 (before the final classifier layer)
    # ResNet structure: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
    latent_extractor = LayeredModelExtractorBuilder.build(
        model=model,
        split_layer=8,  # Split at avgpool (index 8), before fc layer
        device=str(device_param),
        batch_size=1
    )
    return latent_extractor


def test_latent_extractor(image_data, latent_extractor_data):
    """Test that latent extractor returns ClassifierTensor with correct shape."""
    _, input_tensor = image_data
    latent_extractor = latent_extractor_data

    # Test the latent extractor output
    with torch.no_grad():
        results = latent_extractor(input_tensor)

    # Should return ClassifierTensor with shape (batch, num_classes)
    assert isinstance(results, ClassifierTensor), "Results should be a ClassifierTensor"
    assert results.shape == (1, 1000), f"Expected shape (1, 1000), got {results.shape}"


def test_latent_extractor_gradients(image_data, latent_extractor_data):
    """Test that gradients flow correctly through the latent extractor."""
    _, input_tensor = image_data
    latent_extractor = latent_extractor_data

    # Use check_model_gradients to verify gradient flow
    check = check_model_gradients(latent_extractor, input_tensor)
    assert check, "Latent extractor gradients should be computed successfully."


def test_latent_extractor_saliency(image_data, latent_extractor_data, device_param):
    """Test saliency attribution method on latent extractor."""
    _, input_tensor = image_data
    latent_extractor = latent_extractor_data

    # Get predictions
    with torch.no_grad():
        predictions = latent_extractor(input_tensor)

    # Get top prediction
    top_class = predictions.argmax(dim=1).item()

    # Convert input for TensorFlow format (Xplique expects channels last)
    input_tensor_tf_dim = input_tensor.detach().cpu().numpy().transpose(0, 2, 3, 1)

    # Wrap model for Xplique
    torch_wrapped_model = TorchWrapper(
        latent_extractor, device=device_param, is_channel_first=True
    )

    # Create explainer
    explainer = Saliency(
        torch_wrapped_model, operator=xplique.Tasks.CLASSIFICATION, batch_size=1
    )
    explanation = explainer.explain(input_tensor_tf_dim, targets=np.array([top_class]))

    # Check explanation shape
    expected_shape = (1, 224, 224, 1)
    assert explanation.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {explanation.shape}"
    )

    # Visualize (would need raw_image in real scenario)
    # plot_attributions(explanation, [np.array(raw_image.resize((224, 224)))],
    #                   img_size=6., cmap='jet', alpha=0.3,
    #                   absolute_value=False, clip_percentile=0.5)


@pytest.fixture(scope="function")
def craft_data(image_data, latent_extractor_data, device_param):
    """Pytest fixture to create and fit CRAFT instance."""
    _, input_tensor = image_data
    latent_extractor = latent_extractor_data

    # Create CRAFT instance
    craft = Craft(
        latent_extractor=latent_extractor,
        number_of_concepts=10,
        device=str(device_param)
    )

    # Fit on the input (in real scenario, use multiple images)
    craft.fit(input_tensor)

    return craft


def test_craft_reencode(image_data, craft_data):
    """Test CRAFT encode and decode operations."""
    _, input_tensor = image_data
    craft = craft_data

    # Encode the input
    encoded_data = craft.encode(input_tensor)

    # Should have one tuple per image in the batch
    expected_tuples = input_tensor.shape[0]
    assert len(encoded_data) == expected_tuples, (
        f"Expected {expected_tuples} tuples, got {len(encoded_data)}"
    )

    # Get the first tuple
    latent_data, coeffs_u = encoded_data[0]

    print(f"coeffs_u shape: {coeffs_u.shape}")
    # For ResNet with avgpool output, coeffs_u should have spatial dimensions
    assert len(coeffs_u.shape) == 4, "coeffs_u should be 4D (batch, height, width, concepts)"
    assert coeffs_u.shape[0] == 1, "Batch dimension should be 1"
    assert coeffs_u.shape[3] == 10, "Should have 10 concepts"

    # Decode back
    result = craft.decode(latent_data, coeffs_u)
    assert isinstance(result, ClassifierTensor), "Decoded result should be a ClassifierTensor"
    assert result.shape == (1, 1000), f"Expected shape (1, 1000), got {result.shape}"


def test_craft_decoder_modes(image_data, craft_data):
    """Test CRAFT concept decoder functionality."""
    _, input_tensor = image_data
    craft = craft_data

    # Encode
    encoded_data = craft.encode(input_tensor)
    latent_data, coeffs_u = encoded_data[0]

    # Create decoder
    decoder = craft.make_concept_decoder(latent_data)

    # Decoder should return tensor
    output_tensor = decoder(coeffs_u)
    assert hasattr(output_tensor, 'shape'), "Decoder should return a tensor"
    assert output_tensor.shape == (1, 1000), f"Expected shape (1, 1000), got {output_tensor.shape}"


def test_craft_concepts(image_data, craft_data):
    """Test CRAFT concept visualization."""
    _, input_tensor = image_data
    craft = craft_data

    # Display concepts (should work without error)
    craft.display_images_per_concept(
        input_tensor, order=None, filter_percentile=80, clip_percentile=5
    )


def test_craft_gradient_input(image_data, craft_data):
    """Test CRAFT gradient-based importance estimation."""
    _, input_tensor = image_data
    craft = craft_data

    # Use a specific class for testing (e.g., class 281 is 'tabby cat')
    class_id = 281

    # Test compute_explanation_per_concept
    operator = xplique.Tasks.CLASSIFICATION
    explainer_partial = partial(
        GradientInput,
        operator=operator,
        harmonize=False,
    )
    explanation = craft.compute_explanation_per_concept(
        input_tensor, class_id=class_id, explainer_partial=explainer_partial
    )

    # Verify explanation shape
    assert explanation.shape[0] == 1, "Should have one explanation per image"
    assert explanation.shape[3] == 10, "Should match number of concepts"

    # Test estimate_importance
    importances_gi = craft.estimate_importance(
        input_tensor, operator, class_id, method='gradient_input'
    )

    # Verify importance scores
    assert importances_gi.shape == (10,), (
        f"Expected shape (10,), got {importances_gi.shape}"
    )
    assert np.all(np.isfinite(importances_gi)), "All importances should be finite"

    order = importances_gi.argsort()[::-1]
    print(f"Concept importances order for class {class_id}: {order}")
    print(f"Concept importances: {importances_gi}")

    # Display top concepts
    craft.display_images_per_concept(
        input_tensor, order=order, filter_percentile=80, clip_percentile=5
    )


def test_craft_encode_differentiable_gradients(image_data, craft_data):
    """Test that encode(differentiable=True) preserves gradients."""
    _, input_tensor = image_data
    craft = craft_data

    # Ensure input has gradients enabled
    input_with_grad = input_tensor.clone().detach().requires_grad_(True)

    # Test differentiable encoding
    encoded_data = craft.encode(input_with_grad, differentiable=True)
    _, coeffs_u = encoded_data[0]

    # Verify coeffs_u is a tensor with gradients
    expected_msg = "coeffs_u should be a torch.Tensor in differentiable mode"
    assert isinstance(coeffs_u, torch.Tensor), expected_msg
    assert coeffs_u.requires_grad, "coeffs_u should have gradients enabled"

    # Create a simple loss and check gradient flow
    loss = coeffs_u.sum()
    loss.backward()
    gradients = input_with_grad.grad

    # Verify gradients flowed back to input
    assert gradients is not None, "Gradients should flow back to input"
    assert gradients.abs().sum() > 0, "Gradients should be non-zero"


def test_craft_sobol_importance(image_data, craft_data):
    """Test Sobol importance estimation for concepts."""
    _, input_tensor = image_data
    craft = craft_data

    # Use a specific class for testing
    class_id = 281  # 'tabby cat'
    operator = xplique.Tasks.CLASSIFICATION

    # Estimate importance using Sobol method (this may take longer)
    importances_sobol = craft.estimate_importance(
        input_tensor,
        operator,
        class_id,
        method='sobol',
        nb_design=32  # Reduced for faster testing
    )

    # Verify importance scores
    assert importances_sobol.shape == (10,), (
        f"Expected shape (10,), got {importances_sobol.shape}"
    )
    assert np.all(np.isfinite(importances_sobol)), (
        "All importances should be finite"
    )
    assert np.all(importances_sobol >= 0), (
        "Sobol importances should be non-negative"
    )

    order = importances_sobol.argsort()[::-1]
    print(f"Concept importances (Sobol) order for class {class_id}: {order}")
    print(f"Concept importances (Sobol): {importances_sobol}")
