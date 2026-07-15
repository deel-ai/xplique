"""
Utilities for checking gradient propagation in PyTorch models.
"""

# pylint: disable=duplicate-code
from typing import Any, Callable, List, Union

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


def _vjp_probes(tensor: torch.Tensor) -> List[torch.Tensor]:
    """Return deterministic probes that do not reduce every output to its sum."""
    alternating = torch.arange(tensor.numel(), device=tensor.device)
    alternating = 2 * torch.remainder(alternating, 2) - 1
    return [torch.ones_like(tensor), alternating.to(tensor.dtype).reshape_as(tensor)]


def check_model_gradients(
    func: Union[Callable, torch.nn.Module], input_tensor: torch.Tensor, verbose: bool = False
) -> bool:
    """
    Checks if gradients are propagated to the inputs of a PyTorch model.

    Parameters
    ----------
    func
        A PyTorch model (nn.Module) or a callable function.
    input_tensor
        The input tensor.
    verbose
        If True, print information about gradient computation. Default is False.

    Returns
    -------
    bool
        True if non-zero gradients are propagated to the input, False otherwise.
    """
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError("input_tensor must be a torch.Tensor")
    if not (input_tensor.is_floating_point() or input_tensor.is_complex()):
        raise TypeError("input_tensor must have a floating-point or complex dtype")

    module_states = []
    in_place_warning_printed = False
    if isinstance(func, torch.nn.Module):
        module_states = [(module, module.training) for module in func.modules()]

    try:
        # This context restores the caller's grad-mode state on every exit path.
        with torch.enable_grad():
            device = None
            if isinstance(func, torch.nn.Module):
                func.eval()

                # Print a warning about in-place operations in ReLU, etc...
                for module in func.modules():
                    if hasattr(module, "inplace") and module.inplace:
                        if not in_place_warning_printed and verbose:
                            print(
                                f"Warning: In-place operation found in {type(module)}. "
                                f"This may cause issues with gradient computation."
                            )
                        in_place_warning_printed = True

                try:
                    device = next(func.parameters()).device
                except StopIteration:
                    try:
                        device = next(func.buffers()).device
                    except StopIteration:
                        pass

            # Transfer before enabling gradients on the input so it remains a leaf.
            x = input_tensor.detach()
            if device is not None:
                x = x.to(device)
            x = x.clone().requires_grad_(True)
            outputs = func(x)
            tensors = _extract_all_tensors(outputs)

            if not tensors:
                if verbose:
                    print("No tensor found in outputs")
                return False

            # Probe each output independently: summing outputs can cancel gradients,
            # for example when a model returns probabilities that sum to one.
            for tensor in tensors:
                if (
                    not (tensor.is_floating_point() or tensor.is_complex())
                    or not tensor.requires_grad
                ):
                    continue
                for probe in _vjp_probes(tensor):
                    (gradients,) = torch.autograd.grad(
                        tensor,
                        x,
                        grad_outputs=probe,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    if gradients is None:
                        continue
                    is_finite = bool(torch.all(torch.isfinite(gradients)).item())
                    is_nonzero = bool(torch.any(gradients != 0).item())
                    if is_finite and is_nonzero:
                        if verbose:
                            print(f"Gradients OK - sum={gradients.abs().sum().item():.6f}")
                        return True

            if verbose:
                print("No finite, non-zero gradients")
            return False

    # pylint: disable=broad-exception-caught
    except Exception as e:
        if verbose:
            print(f"Error: {str(e)}")
        return False
    finally:
        for module, training in module_states:
            module.training = training
