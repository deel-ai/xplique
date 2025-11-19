"""
Tests for PyTorch gradient checking utilities.

This module tests the check_model_gradients function with various output formats
to ensure it correctly detects gradient flow in different model architectures.
"""
# pylint: disable=redefined-outer-name

import pytest
import torch

from xplique.utils_functions.common.torch.gradients_check import check_model_gradients
from xplique.utils_functions.object_detection.torch.multi_box_tensor import MultiBoxTensor


class SimpleTensorModel(torch.nn.Module):
    """Model that returns a single tensor."""
    def __init__(self, has_gradient=True):
        super().__init__()
        self.has_gradient = has_gradient
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        """Forward pass of the model."""
        if self.has_gradient:
            # Process input to create gradient path
            pooled = torch.mean(x, dim=[2, 3])  # (batch, channels)
            return self.linear(pooled)
        # No gradient path - return constant
        return torch.ones(x.shape[0], 5, device=x.device)


class DictOutputModel(torch.nn.Module):
    """Model that returns a dict of tensors."""
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        """Forward pass returning dict."""
        pooled = torch.mean(x, dim=[2, 3])
        out = self.linear(pooled)
        return {'scores': out, 'features': out * 2}


class ListOutputModel(torch.nn.Module):
    """Model that returns a list of tensors."""
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        """Forward pass returning list."""
        pooled = torch.mean(x, dim=[2, 3])
        out = self.linear(pooled)
        return [out, out * 2, out * 3]


class TorchvisionFormatModel(torch.nn.Module):
    """Model that returns list of dicts (Torchvision format)."""
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        """Forward pass returning list of dicts."""
        batch_size = x.shape[0]
        pooled = torch.mean(x, dim=[2, 3])

        results = []
        for i in range(batch_size):
            out = self.linear(pooled[i:i+1])
            results.append({
                'boxes': out,
                'scores': out * 2,
                'labels': out * 3
            })
        return results


class NestedDictModel(torch.nn.Module):
    """Model that returns nested dict structure.

    IMPORTANT: Only nested tensors have gradients. Top-level 'meta' is constant.
    This tests whether the function checks nested structures.
    """
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        """Forward pass returning nested dict."""
        pooled = torch.mean(x, dim=[2, 3])
        out = self.linear(pooled)
        return {
            'predictions': {
                'scores': out,
                'features': out * 2
            },
            'meta': torch.ones(x.shape[0], 5, device=x.device)  # Constant, no gradient
        }


class DictWithListModel(torch.nn.Module):
    """Model that returns dict containing lists of tensors.

    IMPORTANT: Only tensors in the list have gradients. 'meta' is constant.
    This tests whether the function checks nested lists.
    """
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        """Forward pass returning dict with list values."""
        pooled = torch.mean(x, dim=[2, 3])
        out = self.linear(pooled)
        return {
            'predictions': [out, out * 2],
            'meta': torch.ones(x.shape[0], 5, device=x.device)  # Constant, no gradient
        }


class MixedListModel(torch.nn.Module):
    """Model that returns list with both tensors and dicts.

    IMPORTANT: Only dict tensors have gradients. Direct tensors are constants.
    This tests whether the function checks ALL elements in mixed lists.
    """
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        """Forward pass returning mixed list."""
        pooled = torch.mean(x, dim=[2, 3])
        out = self.linear(pooled)
        return [
            torch.ones(x.shape[0], 5, device=x.device),  # Constant, no gradient
            {'scores': out * 2, 'features': out * 3},
            torch.zeros(x.shape[0], 5, device=x.device)  # Constant, no gradient
        ]


class NestedListInDictModel(torch.nn.Module):
    """Model that returns list of dicts containing lists.

    Only tensors in nested lists have gradients. 'meta' is constant.
    This tests whether the function checks deeply nested structures.
    """
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        """Forward pass returning list of dicts with nested lists."""
        batch_size = x.shape[0]
        pooled = torch.mean(x, dim=[2, 3])

        results = []
        for i in range(batch_size):
            out = self.linear(pooled[i:i+1])
            results.append({
                'predictions': [out, out * 2],
                'meta': torch.ones(1, 5, device=x.device)  # Constant, no gradient
            })
        return results

class MultiBoxModel(torch.nn.Module):
    """Model that returns MultiBoxTensor."""
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 85)  # 4 boxes + 1 score + 80 classes

    def forward(self, x):
        """Forward pass returning MultiBoxTensor."""
        pooled = torch.mean(x, dim=[2, 3])  # (batch, channels)
        out = self.linear(pooled)

        # Create MultiBoxTensor (9 boxes, 85 features each)
        # Reshape to (batch * num_boxes, features)
        reshaped = out.view(-1, 85)

        # Convert to MultiBoxTensor
        return MultiBoxTensor(reshaped)

@pytest.fixture
def input_tensor():
    """Create a simple input tensor."""
    return torch.randn(2, 10, 32, 32)


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


def test_torchvision_format(input_tensor):
    """Test gradient checking with Torchvision format (list of dicts)."""
    model = TorchvisionFormatModel()
    result = check_model_gradients(model, input_tensor)
    assert result, "Gradients should flow through Torchvision format model"


def test_nested_dict(input_tensor):
    """Test gradient checking with nested dict structure.

    Should FAIL with current implementation (only checks top-level 'meta' which has no gradient).
    Should PASS with recursive implementation (finds gradients in nested 'predictions' dict).
    """
    model = NestedDictModel()
    result = check_model_gradients(model, input_tensor)
    assert result, "Should find gradients in nested dict structure"


def test_dict_with_list(input_tensor):
    """Test gradient checking with dict containing lists.

    Should FAIL with current implementation (only checks 'meta' which has no gradient).
    Should PASS with recursive implementation (finds gradients in 'predictions' list).
    """
    model = DictWithListModel()
    result = check_model_gradients(model, input_tensor)
    assert result, "Should find gradients in nested list structure"


def test_mixed_list(input_tensor):
    """Test gradient checking with mixed list (tensors and dicts).

    Should FAIL with current implementation (checks direct tensors which have no gradient).
    Should PASS with recursive implementation (finds gradients in nested dict).
    """
    model = MixedListModel()
    result = check_model_gradients(model, input_tensor)
    assert result, "Should find gradients in nested dict within mixed list"


def test_nested_list_in_dict(input_tensor):
    """Test gradient checking with list of dicts containing lists.

    Should FAIL with current implementation (only checks 'meta' which has no gradient).
    Should PASS with recursive implementation (finds gradients in nested lists).
    """
    model = NestedListInDictModel()
    result = check_model_gradients(model, input_tensor)
    assert result, "Should find gradients in deeply nested list structures"


def test_multibox_tensor_output(input_tensor):
    """Test gradient checking with model that returns MultiBoxTensor."""
    model = MultiBoxModel()
    result = check_model_gradients(model, input_tensor)
    assert result, "Gradients should flow through MultiBoxTensor (it extends torch.Tensor)"


def test_invalid_input_type():
    """Test that invalid input type raises error."""
    model = SimpleTensorModel()
    invalid_input = "not a tensor"  # type: ignore
    with pytest.raises(TypeError):
        check_model_gradients(model, invalid_input)  # type: ignore


def test_callable_function(input_tensor):
    """Test gradient checking with a callable function instead of nn.Module."""
    def simple_func(x):
        return torch.mean(x, dim=[2, 3])

    result = check_model_gradients(simple_func, input_tensor)
    assert result, "Gradients should flow through simple callable"


def test_empty_dict(input_tensor):
    """Test gradient checking with model returning empty dict."""
    class EmptyDictModel(torch.nn.Module):
        """Model returning empty dict."""
        def forward(self, _):
            """Forward pass returning empty dict."""
            return {}

    model = EmptyDictModel()
    result = check_model_gradients(model, input_tensor)
    assert not result, "Should detect no tensors in empty dict"


def test_empty_list(input_tensor):
    """Test gradient checking with model returning empty list."""
    class EmptyListModel(torch.nn.Module):
        """Model returning empty list."""
        def forward(self, _):
            """Forward pass returning empty list."""
            return []

    model = EmptyListModel()
    result = check_model_gradients(model, input_tensor)
    assert not result, "Should detect no tensors in empty list"


def test_dict_with_non_tensor_values(input_tensor):
    """Test gradient checking with dict containing non-tensor values."""
    class MixedDictModel(torch.nn.Module):
        """Model returning dict with mixed value types."""
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, x):
            """Forward pass returning dict with mixed values."""
            pooled = torch.mean(x, dim=[2, 3])
            out = self.linear(pooled)
            return {
                'scores': out,
                'metadata': 'some string',
                'count': 42
            }

    model = MixedDictModel()
    result = check_model_gradients(model, input_tensor)
    assert result, "Should still detect gradients from tensor values"


def test_list_with_non_tensor_values(input_tensor):
    """Test gradient checking with list containing non-tensor values."""
    class MixedListWithNonTensorsModel(torch.nn.Module):
        """Model returning list with mixed value types."""
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, x):
            """Forward pass returning list with mixed values."""
            pooled = torch.mean(x, dim=[2, 3])
            out = self.linear(pooled)
            return [out, 'metadata', 42, out * 2]

    model = MixedListWithNonTensorsModel()
    result = check_model_gradients(model, input_tensor)
    assert result, "Should still detect gradients from tensor values in mixed list"


def test_model_with_input_preprocessing_breaking_gradients(input_tensor):
    """Test gradient checking with model that breaks gradients during preprocessing.

    This replicates DETR's issue: inputs are preprocessed into a new tensor format
    where the gradient connection to the original inputs is severed.

    Pattern (like DETR's nested_tensor_from_tensor_list):
    1. Input comes in with requires_grad=True (for attribution)
    2. Preprocessing creates new tensor with torch.zeros() (requires_grad=False)
    3. Data is copied from input to new tensor
    4. Gradient path is broken - backward pass can't reach original inputs
    """
    class PreprocessingModel(torch.nn.Module):
        """Model with preprocessing that breaks gradients."""
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(10, 5, 3, padding=1)

        def forward(self, x):
            """Forward pass with gradient-breaking preprocessing."""
            # Simulate DETR's nested tensor preprocessing
            # Create new tensor format (e.g., for padding to common size)
            batch_size, channels, height, width = x.shape

            # Create padded tensor with torch.zeros (requires_grad=False!)
            padded = torch.zeros(
                batch_size, channels, height + 10, width + 10,
                device=x.device, dtype=x.dtype
            )

            # Copy input data into new tensor (breaks gradient connection)
            for i, pad_img in enumerate(padded):
                pad_img[:, :height, :width].copy_(x[i])

            # Continue processing with the model
            out = self.conv(padded)
            return torch.mean(out, dim=[2, 3])

    model = PreprocessingModel()
    result = check_model_gradients(model, input_tensor)
    assert not result, (
        "Should detect broken gradients due to input "
        "preprocessing with torch.zeros()"
    )


def test_model_with_backward_error(input_tensor):
    """Test gradient checking with model that raises error during backward.

    This simulates cases where gradient computation fails due to model architecture
    issues, numerical problems, or other errors during backpropagation.

    The check_model_gradients function should return False and catch the error.
    """
    class BackwardErrorModel(torch.nn.Module):
        """Model that breaks gradients during backward."""
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, x):
            """Forward pass with detached output."""
            pooled = torch.mean(x, dim=[2, 3])
            out = self.linear(pooled)

            # Create a custom operation that will fail during backward
            # by using a non-differentiable operation
            out_detached = out.detach()

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
    class NaNModel(torch.nn.Module):
        """Model that produces NaN outputs."""
        def forward(self, x):
            """Forward pass producing NaN."""
            # Create NaN by invalid operation
            return torch.zeros_like(x[:, :, 0, 0]) / 0.0

    model = NaNModel()
    # Should handle NaN gracefully
    result = check_model_gradients(model, input_tensor)
    # Result could be True or False depending on whether NaN propagates
    # The important thing is it doesn't crash
    assert not result, "Should detect no gradients due to NaN outputs"
