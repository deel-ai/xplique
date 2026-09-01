"""Tests for HolisticCraftTf on classification tasks with TensorFlow."""
# pylint: disable=redefined-outer-name

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

import xplique
from xplique.attributions import Rise, Saliency, SobolAttributionMethod
from xplique.attributions.gradient_input import GradientInput
from xplique.concepts import HolisticCraftTf as Craft
from xplique.concepts.holistic_craft import PartialExplainer
from xplique.concepts.tf.latent_extractor import TfLatentExtractor
from xplique.concepts.tf.layered_model_latent_extractor import (
    LayeredLatentData,
    LayeredModelExtractorBuilder,
)
from xplique.utils_functions.classification.tf.classifier_tensor import TfClassifierTensor
from xplique.utils_functions.common.tf.gradients_check import check_model_gradients


class _IdentityFactorizer:
    is_fitted = False
    requires_positive_activations = False

    def fit(self, activations):
        self.is_fitted = True
        return np.eye(2, dtype=np.float32), np.asarray(activations, dtype=np.float32)

    def encode(self, activations):
        return np.asarray(activations, dtype=np.float32)

    def encode_differentiable(self, activations):
        return activations


def test_classifier_tensor_targets_class_and_preserves_batch_shape():
    predictions = TfClassifierTensor(tf.constant([[0.1, 0.2, 0.7], [0.3, 0.6, 0.1]]))

    targets = predictions.to_attribution_target(class_id=1)

    np.testing.assert_array_equal(targets.tensor.numpy(), [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    assert targets.to_batched_tensor().shape == (2, 3)

    with pytest.raises(ValueError):
        predictions.to_attribution_target(class_id=3)


@pytest.fixture(params=["cpu", "gpu"])
def device_param(request):
    """Pytest fixture to provide device parameter (cpu or gpu)."""
    if request.param == "gpu" and not tf.config.list_physical_devices('GPU'):
        pytest.skip("GPU not available")
    return request.param


@pytest.fixture(scope="function")
def image_data(device_param):
    """Pytest fixture to create a fake image for testing."""
    device_name = f'/{device_param.upper()}:0'
    with tf.device(device_name):
        rng = np.random.default_rng(seed=42)
        raw_image = Image.fromarray(rng.integers(0, 256, (462, 640, 3), dtype=np.uint8))

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
    """Test that loaded image has expected dimensions."""
    image, _ = image_data
    expected_size = (640, 462)
    assert image.size == expected_size


@pytest.fixture(scope="function")
def model_data(image_data, device_param):
    """Pytest fixture to load a local ResNet50 model and run predictions."""
    device_name = f'/{device_param.upper()}:0'
    with tf.device(device_name):
        _, input_tensor = image_data
        # Do not download ImageNet weights during tests.
        model = tf.keras.applications.ResNet50(weights=None)
        predictions = model.predict(input_tensor, verbose=0)

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
    device_name = f'/{device_param.upper()}:0'
    with tf.device(device_name):
        model, _ = model_data

        # Split ResNet50 at layer -3 (before GlobalAveragePooling2D)
        # This preserves spatial dimensions needed for CRAFT
        # ResNet structure: ... -> conv5_block3_out (7x7x2048)
        # -> avg_pool (2048) -> predictions (1000)
        latent_extractor = LayeredModelExtractorBuilder.build(
            model=model,
            split_layer=-3,  # Split before avg_pool to preserve spatial dimensions
            batch_size=1
        )
        return latent_extractor


def test_latent_extractor(image_data, latent_extractor_data):
    """Test that latent extractor returns ClassifierTensor with correct shape."""
    _, input_tensor = image_data
    latent_extractor = latent_extractor_data

    # Test the latent extractor output
    results = latent_extractor(input_tensor)

    # Should return TfClassifierTensor with shape (batch, num_classes)
    assert isinstance(results, TfClassifierTensor), "Results should be a TfClassifierTensor"
    assert results.shape == (1, 1000), f"Expected shape (1, 1000), got {results.shape}"


def test_layered_latent_data_getitem():
    """Test that LayeredLatentData supports integer and slice indexing."""
    activations = tf.random.normal((4, 7, 7, 16))
    latent_data = LayeredLatentData(activations)

    sliced = latent_data[1:3]
    assert isinstance(sliced, LayeredLatentData)
    assert sliced.activations.shape[0] == 2

    item = latent_data[0]
    assert isinstance(item, LayeredLatentData)


def test_latent_extractor_gradients(image_data, latent_extractor_data):
    """Test that gradients flow correctly through the latent extractor."""
    _, input_tensor = image_data
    latent_extractor = latent_extractor_data

    # Use check_model_gradients to verify gradient flow
    check = check_model_gradients(latent_extractor, input_tensor)
    assert check, "Latent extractor gradients should be computed successfully."


def test_latent_extractor_saliency(image_data, latent_extractor_data):
    """Test saliency attribution method on latent extractor."""
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
    expected_shape = (1, 224, 224, 1)
    assert explanation.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {explanation.shape}"
    )


@pytest.fixture(scope="function")
def craft_data(image_data, latent_extractor_data, device_param):
    """Pytest fixture to create and fit CRAFT instance."""
    device_name = f'/{device_param.upper()}:0'
    with tf.device(device_name):
        _, input_tensor = image_data
        latent_extractor = latent_extractor_data

        # Create CRAFT instance
        craft = Craft(
            latent_extractor=latent_extractor,
            number_of_concepts=10
        )

        # Fit on the input (in real scenario, use multiple images)
        craft.fit(input_tensor)

        return craft


@pytest.fixture
def tiny_craft_data():
    """Create a deterministic identity CRAFT pipeline for localization tests."""
    values = np.arange(2 * 4 * 4 * 2, dtype=np.float32).reshape(2, 4, 4, 2)
    extractor = TfLatentExtractor(
        model=lambda inputs: inputs,
        input_to_latent_model=lambda inputs: LayeredLatentData(inputs),
        latent_to_logit_model=lambda latent_data: tf.reduce_mean(
            latent_data.activations, axis=(1, 2)
        ),
        batch_size=2,
    )
    craft = Craft(extractor, number_of_concepts=2, factorizer=_IdentityFactorizer())
    craft.fit(tf.constant(values))
    return craft, values


def test_craft_reencode(image_data, craft_data):
    """Test CRAFT encode and decode operations."""
    _, input_tensor = image_data
    craft = craft_data

    # Encode the input
    encoded_data = craft.encode(input_tensor)

    # Should have one tuple per image in the batch
    expected_len = input_tensor.shape[0]
    assert len(encoded_data) == expected_len, (
        f"Expected {expected_len} tuples, got {len(encoded_data)}"
    )

    # Get the first tuple
    latent_data, coeffs_u = encoded_data[0]

    # For ResNet with GlobalAveragePooling output, coeffs_u should have spatial dimensions
    assert len(coeffs_u.shape) == 4, "coeffs_u should be 4D (batch, height, width, concepts)"
    assert coeffs_u.shape[0] == 1, "Batch dimension should be 1"
    assert coeffs_u.shape[3] == 10, "Should have 10 concepts"

    # Decode back
    result = craft.decode(latent_data, coeffs_u)
    assert isinstance(result, TfClassifierTensor), "Decoded result should be a TfClassifierTensor"
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


def test_craft_gradient_input(image_data, craft_data):
    """Test CRAFT gradient-based importance estimation."""
    _, input_tensor = image_data
    craft = craft_data

    # Use a specific class for testing (e.g., class 281 is 'tabby cat')
    class_id = 281

    # Test compute_explanation_per_concept
    operator = xplique.Tasks.CLASSIFICATION
    partial_explainer = PartialExplainer(
        GradientInput,
        operator=operator,
        reducer=None,
    )
    explanation = craft.compute_explanation_per_concept(
        input_tensor, class_id=class_id, partial_explainer=partial_explainer
    )

    # Verify explanation shape
    assert explanation.shape[0] == 1, "Should have one explanation per image"
    assert explanation.shape[3] == 10, "Should match number of concepts"

    # Test estimate_importance
    importances_gi = craft.estimate_importance(
        input_tensor, operator, class_id, method='gradient_input'
    )

    # Verify importance scores
    assert importances_gi.shape == (10,), f"Expected shape (10,), got {importances_gi.shape}"
    assert np.all(np.isfinite(importances_gi)), "All importances should be finite"

    order = importances_gi.argsort()[::-1]
    assert order.shape == (10,), "Should have ordering for all concepts"
    assert len(np.unique(order)) == 10, "All concepts should have unique ordering"


def test_craft_encode_differentiable_gradients(image_data, craft_data):
    """Test that encode(differentiable=True) preserves gradients."""
    _, input_tensor = image_data
    craft = craft_data

    # Test differentiable encoding with gradient tape
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)

        # Encode with differentiable mode
        encoded_data = craft.encode(input_tensor, differentiable=True)
        _, coeffs_u = encoded_data[0]

        # Verify coeffs_u is a tensor
        assert isinstance(coeffs_u, tf.Tensor), (
            "coeffs_u should be a tf.Tensor in differentiable mode"
        )
        assert tf.reduce_all(tf.math.is_finite(coeffs_u)), "coeffs_u should be finite"
        assert tf.reduce_min(coeffs_u) >= -1e-6, "coeffs_u should remain non-negative"

        # Create a simple loss
        loss = tf.reduce_sum(coeffs_u)

    # Check gradient flow
    gradients = tape.gradient(loss, input_tensor)

    # Verify gradients flowed back to input
    assert gradients is not None, "Gradients should flow back to input"
    assert tf.reduce_sum(tf.abs(gradients)) > 0, "Gradients should be non-zero"


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
        grid_size=4,  # Reduced for faster testing
        nb_design=4  # Reduced for faster testing
    )

    # Verify importance scores
    assert importances_sobol.shape == (10,), f"Expected shape (10,), got {importances_sobol.shape}"
    assert np.all(np.isfinite(importances_sobol)), "All importances should be finite"
    assert np.all(importances_sobol >= 0), "Sobol importances should be non-negative"

    order = importances_sobol.argsort()[::-1]
    assert order.shape == (10,), "Should have ordering for all concepts"
    assert len(np.unique(order)) == 10, "All concepts should have unique ordering"


def test_craft_make_concept_localizer_matches_reduced_transform(tiny_craft_data):
    craft, images = tiny_craft_data

    localizer = craft.make_concept_localizer("mean")
    scores = localizer(images).numpy()
    coeffs_u = craft.transform(images)
    expected = np.mean(coeffs_u, axis=(1, 2))

    assert scores.shape == (2, craft.number_of_concepts)
    assert scores.dtype == np.float32
    assert np.all(np.isfinite(scores))
    np.testing.assert_allclose(scores, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("method", ["rise", "sobol"])
def test_craft_compute_concept_attributions_black_box_smoke(tiny_craft_data, method):
    craft, images = tiny_craft_data
    tf.random.set_seed(1)
    if method == "rise":
        explainer = PartialExplainer(
            Rise,
            nb_samples=8,
            grid_size=2,
            preservation_probability=0.5,
        )
    else:
        explainer = PartialExplainer(
            SobolAttributionMethod,
            grid_size=2,
            nb_design=2,
            perturbation_function="inpainting",
        )

    maps = craft.compute_concept_attributions(
        images[:1],
        partial_explainer=explainer,
        concept_ids=[0],
    )

    assert maps.shape == (1, 4, 4, craft.number_of_concepts)
    assert maps.dtype == np.float32
    assert np.all(np.isfinite(maps[..., 0]))
    assert np.all(np.isnan(np.delete(maps, 0, axis=-1)))
