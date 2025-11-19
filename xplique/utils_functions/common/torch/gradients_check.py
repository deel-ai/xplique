"""
Utilities for checking gradient propagation in PyTorch models.
"""
# pylint: disable=duplicate-code
from typing import Callable, Union, List, Any

import torch


def _extract_all_tensors(obj: Any) -> List[torch.Tensor]:
    """
    Recursively extract all PyTorch tensors from a nested structure.

    Handles nested combinations of dicts, lists, tuples, and tensors.

    Parameters
    ----------
    obj
        Object to extract tensors from (can be tensor, dict, list, tuple, or nested combinations).

    Returns
    -------
    tensors
        List of all tensors found in the structure.
    """
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, dict):
        tensors = []
        for value in obj.values():
            tensors.extend(_extract_all_tensors(value))
        return tensors
    if isinstance(obj, (list, tuple)):
        tensors = []
        for item in obj:
            tensors.extend(_extract_all_tensors(item))
        return tensors
    # Not a tensor or container, return empty list
    return []


def check_model_gradients(func: Union[Callable, torch.nn.Module],
                          input_tensor: torch.Tensor) -> bool:
    """
    Checks if gradients are propagated to the inputs of a PyTorch model.

    Parameters
    ----------
    func
        A PyTorch model (nn.Module) or a callable function.
    input_tensor
        The input tensor.

    Returns
    -------
    bool
        True if non-zero gradients are propagated to the input, False otherwise.
    """
    # Input verification and preparation
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError("input_tensor must be a torch.Tensor")

    # Make sure we're not in a no_grad context
    had_no_grad = torch.is_grad_enabled()
    if not had_no_grad:
        torch.set_grad_enabled(True)

    # Create a copy to avoid modifying the original and enable gradients
    x = input_tensor.detach().clone().requires_grad_(True)

    # Configure the model if it's a nn.Module
    was_training = None
    if isinstance(func, torch.nn.Module):
        # Save current training state and set to eval mode
        was_training = func.training
        func.eval()  # Use eval() to freeze weights without blocking gradients

        # Print a warning about in-place operations in ReLU, etc...
        for m in func.modules():
            if hasattr(m, "inplace") and m.inplace:
                print(
                    f"Warning: In-place operation found in {m}. "
                    f"This may cause issues with gradient computation.")

        try:
            device = next(func.parameters()).device
            x = x.to(device)
        except (StopIteration, RuntimeError):
            pass  # No parameters or other error

    try:
        # Forward pass
        outputs = func(x)

        # Extract all tensors recursively from the output structure
        tensors = _extract_all_tensors(outputs)

        if not tensors:
            print("No tensor found in outputs")
            return False

        # Calculate the loss by summing all tensors
        loss = torch.stack([t.sum() for t in tensors]).sum()

        # Backward pass
        loss.backward()

        # Check if gradients were propagated
        if x.grad is not None and x.grad.abs().sum() > 0:
            print(f"Gradients OK - sum={x.grad.abs().sum().item():.6f}")
            return True

        print("No gradients or zero gradients")
        return False

    # pylint: disable=broad-exception-caught
    except Exception as e:
        print(f"Error: {str(e)}")
        return False
    finally:
        # Restore settings
        if not had_no_grad:
            torch.set_grad_enabled(False)

        # Restore the original model mode
        if isinstance(func, torch.nn.Module) and was_training is not None:
            func.train(was_training)
