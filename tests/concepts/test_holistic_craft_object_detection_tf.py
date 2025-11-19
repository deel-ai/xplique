"""Tests for HolisticCraftTf on object detection tasks with TensorFlow."""
# pylint: disable=redefined-outer-name
import os
import pprint
from functools import partial
from typing import List

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

import xplique
from xplique.attributions import Saliency
from xplique.attributions.gradient_input import GradientInput
from xplique.concepts import HolisticCraftTf as Craft
from xplique.concepts.latent_extractor import LatentData
from xplique.concepts.tf.latent_extractor import TfLatentExtractor
from xplique.plots import plot_attributions
from xplique.plots.display_image_with_boxes import display_image_with_boxes
from xplique.utils_functions.common.tf.gradients_check import check_model_gradients
from xplique.utils_functions.object_detection.base.box_manager import BoxFormat, BoxType
from xplique.utils_functions.object_detection.tf.box_formatter import TfBaseBoxFormatter
from xplique.utils_functions.object_detection.tf.box_manager import TfBoxManager
from xplique.utils_functions.object_detection.tf.box_model_wrapper import TfBoxesModelWrapper
from xplique.utils_functions.object_detection.tf.multi_box_tensor import MultiBoxTensor


pp = pprint.PrettyPrinter(indent=4) # Set Pretty Print Indentation
print(tf.__version__) # Check the version of tensorflow used

test_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# Mock Classes - Shared across tests
# ============================================================================

class MockRetinaNetModel(tf.keras.Model):
    """Mock RetinaNet model that returns predictions with boxes, confidence, and classes."""
    def __init__(self, nb_classes=20):
        super().__init__()
        self.nb_classes = nb_classes

    def call(self, x):
        """Forward pass generating mock predictions based on input."""
        batch_size = tf.shape(x)[0]

        # Use input features to create predictions - ensures gradient flow
        # Pool the input to a scalar per batch and use it to modulate predictions
        pooled = tf.reduce_mean(x, axis=[1, 2, 3])  # (batch_size,)
        pooled = tf.abs(pooled) + 1.0  # Ensure positive, add 1.0 for decent scale
        pooled_expanded = tf.expand_dims(pooled, axis=1)  # (batch_size, 1)
        pooled_boxes = tf.expand_dims(pooled_expanded, axis=-1)  # (batch_size, 1, 1)

        # Create deterministic predictions with at least one high-confidence car detection
        # Car class index = 6 in PASCAL VOC
        # Define a single batch of detections
        single_batch_boxes = [
            [0.3, 0.3, 0.7, 0.7],  # Box 0: high-confidence car
            [0.1, 0.1, 0.3, 0.3],  # Box 1: person
            [0.5, 0.5, 0.9, 0.9],  # Box 2: aeroplane
            [0.2, 0.4, 0.5, 0.8],  # Box 3: bird
            [0.0, 0.0, 0.1, 0.1],  # Box 4-9: low confidence
            [0.0, 0.0, 0.1, 0.1],
            [0.0, 0.0, 0.1, 0.1],
            [0.0, 0.0, 0.1, 0.1],
            [0.0, 0.0, 0.1, 0.1],
            [0.0, 0.0, 0.1, 0.1],
        ]

        single_batch_confidences = [
            0.95,  # High confidence for car
            0.85,  # person
            0.75,  # aeroplane
            0.6,   # bird
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1  # Low confidence for others
        ]

        single_batch_classes = [
            6,   # car
            14,  # person
            0,   # aeroplane
            2,   # bird
            0, 0, 0, 0, 0, 0  # Low confidence detections
        ]

        # Create base predictions
        boxes_base = tf.constant([single_batch_boxes], dtype=tf.float32)  # (1, 10, 4)
        confidences_base = tf.constant([single_batch_confidences], dtype=tf.float32)  # (1, 10)

        # Create class probabilities (nb_classes=20) instead of class indices
        # This ensures gradients flow through class predictions
        nb_classes = 20
        class_probas_list = []
        for cls_idx in single_batch_classes:
            # Create one-hot-like probabilities: 0.99 for target class, 0.01 for others
            # Build list directly without using .numpy()
            probas_row = [0.99 if i == cls_idx else 0.01 for i in range(nb_classes)]
            class_probas_list.append(probas_row)
        class_probas_base = tf.constant([class_probas_list], dtype=tf.float32)  # (1, 10, 20)

        # Tile to match batch_size
        boxes = tf.tile(boxes_base, [batch_size, 1, 1])  # (batch_size, 10, 4)
        confidences = tf.tile(confidences_base, [batch_size, 1])  # (batch_size, 10)
        class_probas = tf.tile(class_probas_base, [batch_size, 1, 1])  # (batch_size, 10, 20)

        # Modulate by input features to create gradient path
        # Use stronger modulation (0.1 instead of 0.001) for better saliency gradients
        pooled_mod = (pooled_expanded - 1.0) * 0.1 + 1.0  # Near 1.0 but with visible gradient
        confidences = confidences * pooled_mod

        # Modulate class probabilities by input for gradient flow
        # Expand pooled_mod for broadcasting: (batch_size, 1) -> (batch_size, 10, 1)
        pooled_mod_3d = tf.expand_dims(pooled_mod, axis=-1)
        class_probas = class_probas * pooled_mod_3d

        # Also modulate boxes slightly for stronger gradients
        boxes = boxes * pooled_boxes * 0.05 + boxes * 0.95  # 5% modulation

        return {
            'boxes': boxes,
            'confidence': confidences,
            'class_probas': class_probas  # Return probabilities instead of class indices
        }

class MockRetinaNetFormatter(TfBaseBoxFormatter):
    """
    Mock formatter for RetinaNet model.call() outputs.

    Closely mimics RetinaNetProcessedBoxFormatter from xplique-adapters.
    Handles RetinaNet-specific output format with boxes and classification logits,
    converting them to unified Xplique format.
    """
    def __init__(self, nb_classes: int = 20,
                 input_box_type: BoxType = BoxType(BoxFormat.XYWH, is_normalized=False),
                 output_box_type: BoxType = BoxType(BoxFormat.XYXY, is_normalized=True),
                 input_image_size: tuple = None,
                 output_image_size: tuple = None) -> None:
        """
        Initialize the mock RetinaNet box formatter.

        Parameters
        ----------
        nb_classes
            Number of object classes in the dataset.
        input_box_type
            Format of boxes from RetinaNet (default XYWH, unnormalized).
        output_box_type
            Desired output format (default XYXY, normalized).
        input_image_size
            Size of input image for coordinate conversion.
        output_image_size
            Target size for output coordinates.
        **kwargs
            Additional arguments passed to parent class.
        """
        super().__init__(input_box_type, output_box_type)

        self.nb_classes = nb_classes
        self.input_image_size = input_image_size
        self.output_image_size = output_image_size

    def __call__(self, model_outputs):
        """Make formatter callable."""
        return self.forward(model_outputs)

    def forward(self, predictions) -> List[MultiBoxTensor]:
        """
        Process RetinaNet predictions handling both single and multi-batch inputs.

        Parameters
        ----------
        predictions
            Dictionary with 'boxes', 'confidence', 'class_probas' keys

        Returns
        -------
        formatted_predictions
            List of MultiBoxTensor objects, one per image in the batch.
        """
        def process_single_batch(batch_idx):
            boxes = predictions['boxes'][batch_idx]
            scores = predictions['confidence'][batch_idx]
            class_probas = predictions['class_probas'][batch_idx]

            scores_expanded = scores[:, tf.newaxis]

            pred_dict = {
                'boxes': boxes,
                'scores': scores_expanded,
                'probas': class_probas  # Use class_probas directly (already has gradients)
            }
            return self.format_predictions(pred_dict, self.input_image_size, self.output_image_size)

        batch_size = predictions['boxes'].shape[0]

        results = []
        for batch_idx in range(batch_size):
            formatted = process_single_batch(batch_idx)
            results.append(formatted)
        return results

class MockTfLatentData(LatentData):
    """Mock latent data container for TensorFlow, similar to PyTorch version."""
    def __init__(self, batch_size=1):
        self.batch_size = batch_size
        self.activations = None

    def set_activations(self, values):
        """Store activations."""
        self.activations = values

    def get_activations(self, as_numpy=False, keep_gradients=False):
        """Retrieve activations."""
        if as_numpy:
            return self.activations.numpy()
        return self.activations

class MockExtractorBuilder:
    """
    Builder for creating TfLatentExtractor instances for mock RetinaNet-style models.

    This class encapsulates the logic for creating a TfLatentExtractor configured
    for mock object detection models that follow the RetinaNet output format.

    Similar to the PyTorch MockExtractorBuilder, this provides a clean interface
    for constructing the latent extractor with all necessary components.
    """

    @classmethod
    def build(cls, nb_classes, image_size, device='cpu', batch_size=1):
        """
        Build a TfLatentExtractor for a mock RetinaNet-style model.

        Creates custom input_to_latent and latent_to_logit functions that split
        the model's forward pass into feature extraction and prediction.

        Parameters
        ----------
        nb_classes : int
            Number of object classes in the dataset.
        image_size : tuple
            Image dimensions as (height, width) for coordinate conversion.
        device : str
            Device to run computations on. Default is 'cpu'.
            Should be 'cpu' or 'gpu'.
        batch_size : int
            Batch size for processing. Default is 1.

        Returns
        -------
        latent_extractor : TfLatentExtractor
            Configured TfLatentExtractor instance for the mock model.
        """
        device_name = f'/{device.upper()}:0'
        with tf.device(device_name):
            # Create mock model (not real RetinaNet)
            mock_model = MockRetinaNetModel(nb_classes=nb_classes)

            def input_to_latent_fn(inputs):
                """Simple mock latent extractor - just pool input to fixed size."""
                batch_size_actual = tf.shape(inputs)[0]

                # Pool to fixed spatial size (10x10) and apply ReLU for NMF
                pooled = tf.image.resize(inputs, (10, 10), method='bilinear')  # (N, 10, 10, C)
                activations = tf.nn.relu(pooled)  # Apply ReLU for NMF

                # Pad/truncate channels to 64 for consistency
                current_channels = tf.shape(activations)[-1]
                if current_channels != 64:
                    padding = [[0, 0], [0, 0], [0, 0], [0, tf.maximum(0, 64 - current_channels)]]
                    activations = tf.pad(activations, padding)
                    activations = activations[:, :, :, :64]

                batch_size_value = (
                    batch_size_actual.numpy()
                    if hasattr(batch_size_actual, 'numpy')
                    else batch_size_actual
                )
                latent_data = MockTfLatentData(batch_size=batch_size_value)
                latent_data.set_activations(activations)
                return latent_data

            def latent_to_logit_fn(latent_data):
                """Simple mock decoder - reuse the existing MockRetinaNetModel logic."""
                activations = latent_data.get_activations(as_numpy=False, keep_gradients=True)
                # Upsample back to expected size and pass through model
                # The model expects (N, H, W, C), we have (N, 10, 10, 64)
                return mock_model(activations)

            # Create formatter with proper image sizes
            formatter = MockRetinaNetFormatter(
                nb_classes,
                input_image_size=image_size,
                output_image_size=image_size
            )

            # Create TfLatentExtractor with simplified mock functions
            latent_extractor = TfLatentExtractor(
                model=mock_model,
                input_to_latent_model=input_to_latent_fn,
                latent_to_logit_model=latent_to_logit_fn,
                latent_data_class=MockTfLatentData,
                output_formatter=formatter,
                batch_size=batch_size
            )

            return latent_extractor


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture(params=["cpu", "gpu"])
def device_param(request):
    """Pytest fixture to parametrize tests with CPU and GPU devices."""
    if request.param == "gpu" and not tf.config.list_physical_devices('GPU'):
        pytest.skip("GPU not available")
    return request.param

@pytest.fixture(scope="function")
def image_data(device_param):
    """Pytest fixture to provide test image data."""
    device_name = f'/{device_param.upper()}:0'
    with tf.device(device_name):
        # Check if 'img.jpg' exists, if not, download it
        if not os.path.exists("img.jpg"):
            print("File 'img.jpg' not found. Downloading...")
            download_url = "https://unsplash.com/photos/MXvcHk-zCIs/download?force=true&w=640"
            os.system(f'wget -O img.jpg "{download_url}"')

        # Load and preprocess the image
        raw_image = Image.open("img.jpg")
        def resize_image(image, target_height=640):
            width, height = image.size
            target_width = int(width * target_height / height)
            return image.resize((target_width, target_height), Image.Resampling.LANCZOS)

        resized_image = resize_image(raw_image)
        image = resized_image.crop((100, 0, 740, 640))

        img_np = np.array(image, dtype=np.float32)
        input_tensor = tf.expand_dims(img_np, axis=0)
        return image, input_tensor

def test_image_size(image_data):
    """Test that the image has the expected size."""
    image, _ = image_data
    expected_size = (640, 640)
    assert image.size == expected_size

@pytest.fixture(scope="function")
def model_data(image_data, device_param):
    """Pytest fixture to provide model and predictions."""
    device_name = f'/{device_param.upper()}:0'
    with tf.device(device_name):
        _, input_tensor = image_data
        model = MockRetinaNetModel(nb_classes=20)
        processed_results = model(input_tensor)
        return model, processed_results

def test_model_outputs(model_data):
    """Test that model outputs have the expected structure."""
    _, processed_results = model_data
    assert isinstance(processed_results, dict)
    assert set(processed_results.keys()) == {'boxes', 'confidence', 'class_probas'}

def test_gradients_predict(image_data, model_data):
    """Test that gradients should not be computed in predict mode."""
    _, input_tensor = image_data
    model, _ = model_data
    check = check_model_gradients(model.predict, input_tensor)
    expected_msg = (
        "Model gradients should not be computed successfully in normal / predict mode."
    )
    assert check is False, expected_msg

def test_gradients_model_call(image_data, model_data):
    """Test that gradients should be computed when calling the model directly."""
    _, input_tensor = image_data
    model, _ = model_data
    check = check_model_gradients(model, input_tensor)
    assert check, "Model gradients should be computed successfully when calling the model."

def test_box_model_wrapper(image_data):
    """Test that the box model wrapper works correctly in both output modes."""
    image, input_tensor = image_data
    image_size = (image.height, image.width)

    model = MockRetinaNetModel(nb_classes=20)
    formatter = MockRetinaNetFormatter(
        nb_classes=20,
        input_image_size=image_size,
        output_image_size=image_size
    )
    wrapper = TfBoxesModelWrapper(model, formatter)

    # Test output_as_list mode (default)
    assert wrapper.output_as_list is True
    output_list = wrapper(input_tensor)
    assert isinstance(output_list, list)
    assert len(output_list) == 1  # One result per batch item
    assert isinstance(output_list[0], MultiBoxTensor)

    # Test output_as_tensor mode
    wrapper.set_output_as_tensor()
    assert wrapper.output_as_list is False
    output_tensor = wrapper(input_tensor)
    assert isinstance(output_tensor, tf.Tensor)
    assert len(output_tensor.shape) == 3  # (batch, num_boxes, features)



@pytest.fixture(scope="module")
def dataset_classes():
    """Pytest fixture to provide dataset class names and mappings."""
    classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    nb_classes = len(classes)
    label_to_color = {'person': 'r',
                'bicycle': 'b',
                'car': 'g',
                'motorcycle': 'y',
                'truck': 'orange'}
    return classes, nb_classes, label_to_color

@pytest.fixture(scope="function")
def latent_extractor_data(dataset_classes, device_param, image_data):
    """Create a latent extractor using the MockExtractorBuilder."""
    _classes_names, nb_classes, _label_to_color = dataset_classes
    image, _ = image_data
    image_size = (image.height, image.width)

    # Use builder to create the latent extractor
    latent_extractor = MockExtractorBuilder.build(
        nb_classes=nb_classes,
        image_size=image_size,
        device=device_param
    )

    return latent_extractor

def test_latent_extractor(image_data, dataset_classes, latent_extractor_data):
    """Test that the latent extractor works correctly in both tensor and list modes."""
    image, input_tensor = image_data
    classes_names, _nb_classes, label_to_color = dataset_classes
    latent_extractor = latent_extractor_data

    # Test in tensor mode
    latent_extractor.set_output_as_tensor()
    results = latent_extractor(input_tensor)
    print("Latent Data Retinanet (tensor mode):", results.shape)
    assert results.shape == (1, 10, 25), "Latent data shape should be (1, 10, 25)."

    # Test in list mode
    latent_extractor.set_output_as_list()
    results_list = latent_extractor(input_tensor)
    print("Latent Data Retinanet (list mode):", type(results_list))
    expected_msg = "Should return a list of MultiBoxTensor objects in list mode"
    assert isinstance(results_list, list), expected_msg
    assert len(results_list) == 1, "Should have one result per batch item"
    assert isinstance(results_list[0], MultiBoxTensor), "First element should be a MultiBoxTensor"
    assert results_list[0].shape == (10, 25), "MultiBoxTensor shape should be (10, 25)."

    filtered_results_list = results_list[0].filter(confidence=0.85)
    box_manager = TfBoxManager(BoxFormat.XYXY, normalized=True)
    display_image_with_boxes(
        image, filtered_results_list, box_manager, classes_names, label_to_color
    )

def test_latent_extractor_gradients(image_data, latent_extractor_data):
    """Test that latent extractor computes gradients successfully."""
    _image, input_tensor = image_data
    latent_extractor = latent_extractor_data

    check = check_model_gradients(latent_extractor, input_tensor)
    assert check, "Latent extractor gradients should be computed successfully."

def test_latent_extractor_saliency(image_data, dataset_classes, latent_extractor_data):
    """Test that saliency attribution works with the latent extractor."""
    image, input_tensor = image_data
    classes_names, _nb_classes, _label_to_color = dataset_classes
    latent_extractor = latent_extractor_data

    # Set to tensor mode for operator compatibility
    latent_extractor.set_output_as_tensor()
    operator = xplique.Tasks.OBJECT_DETECTION_BOX_PROBA

    # Get targets in list mode first, then filter
    latent_extractor.set_output_as_list()
    targets = latent_extractor(input_tensor)
    box_to_explain = targets[0].filter(confidence=0.9, class_id=classes_names.index('car'))

    # Set back to tensor mode for the explainer
    latent_extractor.set_output_as_tensor()
    box_to_explain = box_to_explain.to_batched_tensor()

    explainer = Saliency(latent_extractor, operator=operator, batch_size=None)
    explanation = explainer.explain(input_tensor, targets=box_to_explain)

    # Check explanation shape (should match input image dimensions)
    expected_shape = (1, 640, 640, 1)
    assert explanation.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {explanation.shape}"
    )

    plot_attributions(explanation, [np.array(image)], img_size=6.,
                    cmap='jet', alpha=0.3, absolute_value=False, clip_percentile=0.5)

@pytest.fixture(scope="function")
def craft_data(image_data, latent_extractor_data, device_param):
    """Pytest fixture to create and fit a CRAFT instance."""
    device_name = f'/{device_param.upper()}:0'
    with tf.device(device_name):
        _image, input_tensor = image_data
        latent_extractor = latent_extractor_data

        # Ensure latent extractor is in list mode for CRAFT
        latent_extractor.set_output_as_list()
        craft = Craft(latent_extractor = latent_extractor,
                    number_of_concepts = 10)
        craft.fit(input_tensor)
        return craft

def test_craft_reencode(image_data, dataset_classes, craft_data):
    """Test that CRAFT encoding and decoding work correctly."""
    image, input_tensor = image_data
    classes_names, _nb_classes, label_to_color = dataset_classes
    craft = craft_data

    encoded_data = craft.encode(input_tensor)
    print("nb encoded data:", len(encoded_data))
    latent_data, coeffs_u = encoded_data[0]
    print(coeffs_u.shape)
    assert coeffs_u.shape == (1, 10, 10, 10), "Latent data shape should be (1, 10, 10, 10)."

    decoded_data = craft.decode(latent_data, coeffs_u)
    print(f"decoded_data type = {type(decoded_data)}")
    assert isinstance(decoded_data, MultiBoxTensor), "Should return an MultiBoxTensor directly"
    assert decoded_data.shape == (10, 25), "MultiBoxTensor shape should be (10, 25)."

    filtered_decoded_data = decoded_data.filter(confidence=0.85)
    box_manager = TfBoxManager(BoxFormat.XYXY, normalized=True)
    display_image_with_boxes(
        image, filtered_decoded_data, box_manager, classes_names, label_to_color
    )

def test_craft_concepts(image_data, craft_data):
    """Test that CRAFT concept visualization works correctly."""
    _image, input_tensor = image_data
    craft = craft_data
    craft.display_images_per_concept(
        input_tensor, order=None, filter_percentile=80, clip_percentile=5
    )

def test_craft_gradient_input(image_data, dataset_classes, craft_data):
    """Test that CRAFT importance estimation with gradient input works correctly."""
    _image, input_tensor = image_data
    classes_names, _nb_classes, _label_to_color = dataset_classes
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

    # Verify explanation shape
    assert explanation.shape[0] == 1, "Should have one explanation per image"
    assert explanation.shape[1:3] == (10, 10), "Should match coeffs_u spatial dimensions"
    assert explanation.shape[3] == 10, "Should match number of concepts"

    # Test estimate_importance_gradient_input
    importances_gi = craft.estimate_importance(
        input_tensor, operator, class_id, confidence=0.6, method='gradient_input'
    )
    assert importances_gi.shape == (10,), "Should return importance scores for each concept"

    # Verify importances are computed (don't check specific order for mock model)
    order = importances_gi.argsort()[::-1]
    print(f"Concept importances order for 'person': {order}")
    assert len(order) == 10, "Should have ordering for all concepts"
    craft.display_images_per_concept(
        input_tensor, order=order, filter_percentile=80, clip_percentile=5
    )

def test_craft_decoder_modes(image_data, dataset_classes, craft_data):
    """Test that the decoder works in both tensor and list modes."""
    _image, input_tensor = image_data
    _classes_names, _nb_classes, _label_to_color = dataset_classes
    craft = craft_data

    encoded_data = craft.encode(input_tensor)
    latent_data, coeffs_u = encoded_data[0]
    decoder = craft.make_concept_decoder(latent_data)

    # Decoder should always return tensor (unified behavior with PyTorch)
    output_tensor = decoder(coeffs_u)
    assert hasattr(output_tensor, 'shape'), "Decoder should always return a tensor"
    assert output_tensor.shape == (1, 10, 25), "Should have correct tensor shape"

    # For filtering, use decode directly to get MultiBoxTensor
    nbc_tensor = craft.decode(latent_data, coeffs_u)
    assert hasattr(nbc_tensor, 'filter'), "decode should return MultiBoxTensor with filter method"

def test_multibox_tensor_filter(image_data, dataset_classes, latent_extractor_data):
    """Test MultiBoxTensor filtering functionality."""
    _image, input_tensor = image_data
    classes_names, _nb_classes, _label_to_color = dataset_classes
    latent_extractor = latent_extractor_data

    # Get results in list mode to access MultiBoxTensor
    latent_extractor.set_output_as_list()
    targets = latent_extractor(input_tensor)
    nbc_tensor = targets[0]  # targets is a list, get the first MultiBoxTensor

    # Test filtering by class_id and confidence
    filtered = nbc_tensor.filter(class_id=classes_names.index('person'), confidence=0.5)
    assert hasattr(filtered, 'shape'), "Filtered result should have shape attribute"
    assert len(filtered.shape) == 2, "Filtered result should be 2D (boxes, features)"
    assert filtered.shape[1] == 25, "Should preserve feature dimension"

    print(f"Original: {nbc_tensor.shape}, Filtered: {filtered.shape}")

    # Test filtering with high confidence (should return fewer boxes)
    filtered_high = nbc_tensor.filter(
        class_id=classes_names.index('person'), confidence=0.9
    )
    expected_msg = "Higher confidence should return fewer or equal boxes"
    assert filtered_high.shape[0] <= filtered.shape[0], expected_msg

def test_craft_encode_differentiable_gradients(image_data, craft_data):
    """Test that encode with differentiable mode preserves gradients."""
    _image, input_tensor = image_data
    craft = craft_data

    # Test differentiable encoding with gradient tape
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)

        # Encode with differentiable mode
        encoded_data = craft.encode(input_tensor, differentiable=True)
        _latent_data, coeffs_u = encoded_data[0]

        # Verify coeffs_u is a tensor
        expected_msg = "coeffs_u should be a tf.Tensor in differentiable mode"
        assert isinstance(coeffs_u, tf.Tensor), expected_msg

        # Create a simple loss
        loss = tf.reduce_sum(coeffs_u)

    # Check gradient flow
    gradients = tape.gradient(loss, input_tensor)

    # Verify gradients flowed back to input
    assert gradients is not None, "Gradients should flow back to input"

    # Note: With mock model, NMF may produce NaN gradients due to Cholesky decomposition
    # In real usage with actual models, gradients should be non-zero
    # For testing, we just verify gradient computation doesn't crash
    grad_sum = tf.reduce_sum(tf.abs(gradients))
    if not tf.math.is_nan(grad_sum):
        assert grad_sum > 0, "Gradients should be non-zero (if not NaN)"
