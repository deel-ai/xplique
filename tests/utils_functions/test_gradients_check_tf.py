"""
Tests for TensorFlow gradient checking utilities.

This module tests the check_model_gradients function with various output formats
to ensure it correctly detects gradient flow in different model architectures.
"""
# pylint: disable=redefined-outer-name

import pytest
import tensorflow as tf

from xplique.utils_functions.common.tf.gradients_check import check_model_gradients
from xplique.utils_functions.object_detection.tf.multi_box_tensor import (
    TfMultiBoxTensor as MultiBoxTensor,
)


class SimpleTensorModel(tf.keras.Model):
    """Model that returns a single tensor."""

    def __init__(self, has_gradient=True):
        super().__init__()
        self.has_gradient = has_gradient
        self.dense = tf.keras.layers.Dense(5)

    def call(self, x):
        """Forward pass of the model."""
        if self.has_gradient:
            # Process input to create gradient path
            pooled = tf.reduce_mean(x, axis=[1, 2])  # (batch, channels)
            return self.dense(pooled)
        # No gradient path - return constant
        return tf.ones([tf.shape(x)[0], 5])


class DictOutputModel(tf.keras.Model):
    """Model that returns a dict of tensors."""

    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(5)

    def call(self, x):
        """Forward pass returning dict."""
        pooled = tf.reduce_mean(x, axis=[1, 2])
        out = self.dense(pooled)
        return {"scores": out, "features": out * 2}


class ListOutputModel(tf.keras.Model):
    """Model that returns a list of tensors."""

    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(5)

    def call(self, x):
        """Forward pass returning list."""
        pooled = tf.reduce_mean(x, axis=[1, 2])
        out = self.dense(pooled)
        return [out, out * 2, out * 3]


class NestedDictModel(tf.keras.Model):
    """Model that returns nested dict structure."""

    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(5)

    def call(self, x):
        """Forward pass returning nested dict."""
        pooled = tf.reduce_mean(x, axis=[1, 2])
        out = self.dense(pooled)
        return {"predictions": {"scores": out, "features": out * 2}, "meta": out * 3}


class DictWithListModel(tf.keras.Model):
    """Model that returns dict containing lists of tensors."""

    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(5)

    def call(self, x):
        """Forward pass returning dict with list values."""
        pooled = tf.reduce_mean(x, axis=[1, 2])
        out = self.dense(pooled)
        return {"predictions": [out, out * 2], "meta": out * 3}


class MixedListModel(tf.keras.Model):
    """Model that returns list with both tensors and dicts."""

    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(5)

    def call(self, x):
        """Forward pass returning mixed list."""
        pooled = tf.reduce_mean(x, axis=[1, 2])
        out = self.dense(pooled)
        return [out, {"scores": out * 2, "features": out * 3}, out * 4]


class MultiBoxModel(tf.keras.Model):
    """Model that returns a MultiBoxTensor object."""

    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(85)  # 4 boxes + 1 score + 80 classes

    def call(self, x):
        """Forward pass returning MultiBoxTensor."""
        pooled = tf.reduce_mean(x, axis=[1, 2])  # (batch, channels)
        out = self.dense(pooled)

        # Reshape to (batch * num_boxes, features)
        reshaped = tf.reshape(out, [-1, 85])

        # Create MultiBoxTensor (wraps the tensor with .tensor attribute)
        return MultiBoxTensor(reshaped)


@pytest.fixture
def input_tensor():  # pylint: disable=redefined-outer-name
    """Create a simple input tensor."""
    return tf.random.normal([2, 32, 32, 10])


def test_simple_tensor_with_gradient(input_tensor):
    """Test gradient checking with a simple tensor output."""
    model = SimpleTensorModel(has_gradient=True)
    result = check_model_gradients(model, input_tensor)
    assert result, "Gradients should flow through simple tensor model"


def test_simple_tensor_without_gradient(input_tensor):
    """Test gradient checking when no gradient flows."""
    model = SimpleTensorModel(has_gradient=False)
    result = check_model_gradients(model, input_tensor)
    assert not result, "Should detect when no gradients flow"


def test_dict_output(input_tensor):
    """Test gradient checking with dict output."""
    model = DictOutputModel()
    result = check_model_gradients(model, input_tensor)
    assert result, "Gradients should flow through dict output model"


def test_list_output(input_tensor):
    """Test gradient checking with list output."""
    model = ListOutputModel()
    result = check_model_gradients(model, input_tensor)
    assert result, "Gradients should flow through list output model"


def test_nested_dict(input_tensor):
    """Test gradient checking with nested dict structure."""
    model = NestedDictModel()
    result = check_model_gradients(model, input_tensor)
    assert result, "Gradients should flow through nested dict model"


def test_dict_with_list(input_tensor):
    """Test gradient checking with dict containing lists."""
    model = DictWithListModel()
    result = check_model_gradients(model, input_tensor)
    assert result, "Gradients should flow through dict with list model"


def test_mixed_list(input_tensor):
    """Test gradient checking with mixed list (tensors and dicts)."""
    model = MixedListModel()
    result = check_model_gradients(model, input_tensor)
    assert result, "Gradients should flow through mixed list model"


def test_callable_function(input_tensor):
    """Test gradient checking with a callable function instead of Model."""

    def simple_func(x):
        return tf.reduce_mean(x, axis=[1, 2])

    result = check_model_gradients(simple_func, input_tensor)
    assert result, "Gradients should flow through simple callable"


def test_empty_dict(input_tensor):
    """Test gradient checking with model returning empty dict."""

    class EmptyDictModel(tf.keras.Model):
        """Model returning empty dict."""

        def call(self, _):
            """Forward pass returning empty dict."""
            return {}

    model = EmptyDictModel()
    result = check_model_gradients(model, input_tensor)
    assert not result, "Should detect no tensors in empty dict"


def test_empty_list(input_tensor):
    """Test gradient checking with model returning empty list."""

    class EmptyListModel(tf.keras.Model):
        """Model returning empty list."""

        def call(self, _):
            """Forward pass returning empty list."""
            return []

    model = EmptyListModel()
    result = check_model_gradients(model, input_tensor)
    assert not result, "Should detect no tensors in empty list"


def test_multibox_tensor_output(input_tensor):
    """Test gradient checking with model that returns MultiBoxTensor."""
    model = MultiBoxModel()
    result = check_model_gradients(model, input_tensor)
    assert result, "Gradients should flow through MultiBoxTensor model"


def test_dict_with_non_tensor_values(input_tensor):
    """Test gradient checking with dict containing non-tensor values."""

    class MixedDictModel(tf.keras.Model):
        """Model returning dict with mixed value types."""

        def __init__(self):
            super().__init__()
            self.dense = tf.keras.layers.Dense(5)

        def call(self, x):
            """Forward pass returning dict with mixed values."""
            pooled = tf.reduce_mean(x, axis=[1, 2])
            out = self.dense(pooled)
            return {"scores": out, "metadata": "some string", "count": 42}

    model = MixedDictModel()
    result = check_model_gradients(model, input_tensor)
    assert result, "Should still detect gradients from tensor values"


def test_list_with_non_tensor_values(input_tensor):
    """Test gradient checking with list containing non-tensor values."""

    class MixedListWithNonTensorsModel(tf.keras.Model):
        """Model returning list with mixed value types."""

        def __init__(self):
            super().__init__()
            self.dense = tf.keras.layers.Dense(5)

        def call(self, x):
            """Forward pass returning list with mixed values."""
            pooled = tf.reduce_mean(x, axis=[1, 2])
            out = self.dense(pooled)
            return [out, "metadata", 42, out * 2]

    model = MixedListWithNonTensorsModel()
    result = check_model_gradients(model, input_tensor)
    assert result, "Gradients should flow through mixed list model"


def test_model_with_input_preprocessing_breaking_gradients(input_tensor):
    """Test gradient checking with model that breaks gradients during preprocessing.

    This replicates DETR's issue: inputs are preprocessed into a new tensor format
    where the gradient connection to the original inputs is severed.

    Pattern (like DETR's nested_tensor_from_tensor_list):
    1. Input comes in with gradient tracking enabled (for attribution)
    2. Preprocessing uses tf.stop_gradient to break gradient connection
    3. Model processes the gradient-disconnected tensor
    4. Gradient path is broken - backward pass can't reach original inputs

    Note: In TensorFlow, we explicitly use tf.stop_gradient to break the gradient
    flow, simulating what happens in PyTorch when torch.zeros() creates a tensor
    with requires_grad=False.
    """

    class PreprocessingModel(tf.keras.Model):
        """Model with preprocessing that breaks gradients."""

        def __init__(self):
            super().__init__()
            self.conv = tf.keras.layers.Conv2D(5, 3, padding="same")

        def call(self, x):
            """Forward pass with gradient-breaking preprocessing."""
            # Break gradient connection at preprocessing step
            x_no_grad = tf.stop_gradient(x)
            out = self.conv(x_no_grad)
            return tf.reduce_mean(out, axis=[1, 2])

    model = PreprocessingModel()
    result = check_model_gradients(model, input_tensor)
    assert not result, (
        "Should detect broken gradients due to input preprocessing with tf.stop_gradient"
    )


def test_model_with_backward_error(input_tensor):
    """Test gradient checking with model that raises error during backward.

    This simulates cases where gradient computation fails due to model architecture
    issues, numerical problems, or other errors during backpropagation.

    The check_model_gradients function should return False and catch the error.
    """

    class BackwardErrorModel(tf.keras.Model):
        """Model that breaks gradients during backward."""

        def __init__(self):
            super().__init__()
            self.dense = tf.keras.layers.Dense(5)

        def call(self, x):
            """Forward pass with detached output."""
            pooled = tf.reduce_mean(x, axis=[1, 2])
            out = self.dense(pooled)

            # Detach from gradient computation (TF equivalent of PyTorch's .detach())
            out_detached = tf.stop_gradient(out)

            # Return a tensor that has no gradient connection to input
            return out_detached

    model = BackwardErrorModel()
    # Should detect that no gradients flow through (detached tensor)
    result = check_model_gradients(model, input_tensor)
    assert not result, "Should detect no gradients due to backward error"


def test_model_with_nan_output(input_tensor):
    """Test gradient checking with model that produces NaN outputs.

    This tests whether the function handles numerical instability gracefully.
    """

    class NaNModel(tf.keras.Model):
        """Model that produces NaN outputs."""

        def call(self, x):
            """Forward pass producing NaN."""
            # Create NaN by invalid operation
            return tf.zeros([tf.shape(x)[0], tf.shape(x)[3]]) / 0.0

    model = NaNModel()
    result = check_model_gradients(model, input_tensor)
    # The important thing is it doesn't crash
    assert not result, "Should detect no gradients due to NaN outputs"


def test_model_with_cancelling_outputs(input_tensor):
    """Test that independent VJP probes detect gradients hidden by output summation."""

    def cancelling_model(x):
        return tf.stack([x[..., 0], -x[..., 0]], axis=-1)

    assert check_model_gradients(cancelling_model, input_tensor)


def test_model_with_zero_gradient(input_tensor):
    """Test that zero gradients are rejected."""

    assert not check_model_gradients(lambda x: x * 0.0, input_tensor)


def test_model_with_finite_gradient(input_tensor):
    """Test that finite non-zero gradients are accepted."""

    assert check_model_gradients(lambda x: x * 2.0, input_tensor)


@pytest.mark.parametrize("invalid_gradient", [float("nan"), float("inf")])
def test_model_with_nonfinite_gradient(input_tensor, invalid_gradient):
    """Test that NaN and infinite gradients are rejected."""

    def invalid_gradient_model(x):
        return x * tf.cast(invalid_gradient, x.dtype)

    assert not check_model_gradients(invalid_gradient_model, input_tensor)
