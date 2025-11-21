import os
import pprint

import pytest
import numpy as np
from functools import partial
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

import xplique
from xplique.attributions import Saliency
from xplique.attributions.gradient_input import GradientInput
from xplique.plots import plot_attributions
from xplique.concepts.tf.latent_data_layered_model import LayeredModelExtractorBuilder
from xplique.concepts import HolisticCraftTf as Craft
from xplique.utils_functions.classification.tf.classifier_tensor import ClassifierTensor

pp = pprint.PrettyPrinter(indent=4)
print(tf.__version__)

test_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(params=["cpu", "gpu"])
def device_param(request):
    if request.param == "gpu" and not tf.config.list_physical_devices('GPU'):
        pytest.skip("GPU not available")
    return request.param


@pytest.fixture(scope="function")
def image_data(device_param):
    device_name = f'/{device_param.upper()}:0'
    with tf.device(device_name):
        # Check if 'img.jpg' exists, if not, download it
        if not os.path.exists("img.jpg"):
            print("File 'img.jpg' not found. Downloading...")
            os.system('wget -O img.jpg "https://unsplash.com/photos/MXvcHk-zCIs/download?force=true&w=640"')

        # Load and preprocess the image
        raw_image = Image.open("img.jpg")
        
        # Resize to 224x224 for ImageNet models
        resized_image = raw_image.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_np = np.array(resized_image, dtype=np.float32)
        
        # Standard ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406]) * 255.0
        std = np.array([0.229, 0.224, 0.225]) * 255.0
        img_np = (img_np - mean) / std
        
        # Add batch dimension
        input_tensor = tf.expand_dims(img_np, axis=0)
        
        return raw_image, input_tensor


def test_image_size(image_data):
    image, _ = image_data
    expected_size = (640, 462)
    assert image.size == expected_size


@pytest.fixture(scope="function")
def model_data(image_data, device_param):
    device_name = f'/{device_param.upper()}:0'
    with tf.device(device_name):
        _, input_tensor = image_data
        # Load pretrained ResNet50
        model = tf.keras.applications.ResNet50(weights='imagenet')
        predictions = model.predict(input_tensor, verbose=0)
        
        return model, predictions


def test_model_outputs(model_data):
    _, predictions = model_data
    # ResNet50 outputs 1000 ImageNet classes
    assert predictions.shape == (1, 1000), f"Expected shape (1, 1000), got {predictions.shape}"


@pytest.fixture(scope="module")
def imagenet_classes():
    # Top-5 most common ImageNet classes for testing
    # In a real scenario, you would load all 1000 classes
    CLASSES = [
        'tench', 'goldfish', 'great_white_shark', 'tiger_shark', 'hammerhead',
        'electric_ray', 'stingray', 'cock', 'hen', 'ostrich'
    ]
    return CLASSES


@pytest.fixture(scope="function")
def latent_extractor_data(model_data, device_param):
    device_name = f'/{device_param.upper()}:0'
    with tf.device(device_name):
        model, _ = model_data
        
        # Split ResNet50 at layer -3 (before GlobalAveragePooling2D)
        # This preserves spatial dimensions needed for CRAFT
        # ResNet structure: ... -> conv5_block3_out (7x7x2048) -> avg_pool (2048) -> predictions (1000)
        latent_extractor = LayeredModelExtractorBuilder.build(
            model=model,
            split_layer=-3,  # Split before avg_pool to preserve spatial dimensions
            batch_size=1
        )
        return latent_extractor


def test_latent_extractor(image_data, latent_extractor_data):
    image, input_tensor = image_data
    latent_extractor = latent_extractor_data

    # Test the latent extractor output
    results = latent_extractor(input_tensor)
    
    # Should return ClassifierTensor with shape (batch, num_classes)
    assert isinstance(results, ClassifierTensor), "Results should be a ClassifierTensor"
    assert results.shape == (1, 1000), f"Expected shape (1, 1000), got {results.shape}"


def test_latent_extractor_gradients(image_data, latent_extractor_data):
    image, input_tensor = image_data
    latent_extractor = latent_extractor_data
    
    # Enable gradients with GradientTape
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        output = latent_extractor(input_tensor)
        loss = tf.reduce_sum(output)
    
    # Compute gradients
    gradients = tape.gradient(loss, input_tensor)
    
    # Check that gradients exist
    assert gradients is not None, "Gradients should be computed"
    assert tf.reduce_sum(tf.abs(gradients)) > 0, "Gradients should be non-zero"


def test_latent_extractor_saliency(image_data, imagenet_classes, latent_extractor_data):
    image, input_tensor = image_data
    latent_extractor = latent_extractor_data
    
    # Get predictions
    predictions = latent_extractor(input_tensor)
    
    # Get top prediction
    top_class = tf.argmax(predictions, axis=1).numpy()[0]
    
    # Create explainer
    explainer = Saliency(latent_extractor, operator=xplique.Tasks.CLASSIFICATION, batch_size=1)
    explanation = explainer.explain(input_tensor, targets=np.array([top_class]))

    # Check explanation shape
    assert explanation.shape == (1, 224, 224, 1), f"Expected shape (1, 224, 224, 1), got {explanation.shape}"

    # Visualize
    plot_attributions(explanation, [np.array(image.resize((224, 224)))], img_size=6.,
                    cmap='jet', alpha=0.3, absolute_value=False, clip_percentile=0.5)


@pytest.fixture(scope="function")
def craft_data(image_data, latent_extractor_data, device_param):
    device_name = f'/{device_param.upper()}:0'
    with tf.device(device_name):
        image, input_tensor = image_data
        latent_extractor = latent_extractor_data
        
        # Create CRAFT instance
        craft = Craft(
            latent_extractor=latent_extractor,
            number_of_concepts=10
        )
        
        # Fit on the input (in real scenario, use multiple images)
        craft.fit(input_tensor)
        
        return craft


def test_craft_reencode(image_data, craft_data):
    image, input_tensor = image_data
    craft = craft_data
    
    # Encode the input
    encoded_data = craft.encode(input_tensor)
    
    # Should have one tuple per image in the batch
    assert len(encoded_data) == input_tensor.shape[0], f"Expected {input_tensor.shape[0]} tuples, got {len(encoded_data)}"
    
    # Get the first tuple
    latent_data, coeffs_u = encoded_data[0]
    
    print(f"coeffs_u shape: {coeffs_u.shape}")
    # For ResNet with GlobalAveragePooling output, coeffs_u should have spatial dimensions
    assert len(coeffs_u.shape) == 4, "coeffs_u should be 4D (batch, height, width, concepts)"
    assert coeffs_u.shape[0] == 1, "Batch dimension should be 1"
    assert coeffs_u.shape[3] == 10, "Should have 10 concepts"
    
    # Decode back
    result = craft.decode(latent_data, coeffs_u)
    assert isinstance(result, ClassifierTensor), "Decoded result should be a ClassifierTensor"
    assert result.shape == (1, 1000), f"Expected shape (1, 1000), got {result.shape}"


def test_craft_decoder_modes(image_data, craft_data):
    image, input_tensor = image_data
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
    image, input_tensor = image_data
    craft = craft_data
    
    # Display concepts (should work without error)
    craft.display_images_per_concept(input_tensor, order=None, filter_percentile=80, clip_percentile=5)


def test_craft_gradient_input(image_data, imagenet_classes, craft_data):
    image, input_tensor = image_data
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
    explanation = craft.compute_explanation_per_concept(input_tensor, class_id=class_id, explainer_partial=explainer_partial)
    
    # Verify explanation shape
    assert explanation.shape[0] == 1, "Should have one explanation per image"
    assert explanation.shape[3] == 10, "Should match number of concepts"
    
    # Test estimate_importance
    importances_gi = craft.estimate_importance(input_tensor, operator, class_id, method='gradient_input')
    
    # Verify importance scores
    assert importances_gi.shape == (10,), f"Expected shape (10,), got {importances_gi.shape}"
    assert np.all(np.isfinite(importances_gi)), "All importances should be finite"
    
    order = importances_gi.argsort()[::-1]
    print(f"Concept importances order for class {class_id}: {order}")
    print(f"Concept importances: {importances_gi}")
    
    # Display top concepts
    craft.display_images_per_concept(input_tensor, order=order, filter_percentile=80, clip_percentile=5)


def test_craft_encode_differentiable_gradients(image_data, craft_data):
    """Test that encode(differentiable=True) preserves gradients through the encoding pipeline."""
    image, input_tensor = image_data
    craft = craft_data
    
    # Test differentiable encoding with gradient tape
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        
        # Encode with differentiable mode
        encoded_data = craft.encode(input_tensor, differentiable=True)
        latent_data, coeffs_u = encoded_data[0]
        
        # Verify coeffs_u is a tensor
        assert isinstance(coeffs_u, tf.Tensor), "coeffs_u should be a tf.Tensor in differentiable mode"
        
        # Create a simple loss
        loss = tf.reduce_sum(coeffs_u)
    
    # Check gradient flow
    gradients = tape.gradient(loss, input_tensor)
    
    # Verify gradients flowed back to input
    assert gradients is not None, "Gradients should flow back to input"
    assert tf.reduce_sum(tf.abs(gradients)) > 0, "Gradients should be non-zero"


def test_craft_sobol_importance(image_data, craft_data):
    """Test Sobol importance estimation for concepts."""
    image, input_tensor = image_data
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
    assert importances_sobol.shape == (10,), f"Expected shape (10,), got {importances_sobol.shape}"
    assert np.all(np.isfinite(importances_sobol)), "All importances should be finite"
    assert np.all(importances_sobol >= 0), "Sobol importances should be non-negative"
    
    order = importances_sobol.argsort()[::-1]
    print(f"Concept importances (Sobol) order for class {class_id}: {order}")
    print(f"Concept importances (Sobol): {importances_sobol}")
