import torch
from typing import Callable, Union

def check_model_gradients(func: Union[Callable, torch.nn.Module], input_tensor: torch.Tensor) -> bool:
    """Checks if gradients are propagated to the inputs of a PyTorch model.

    Args:
        func: A PyTorch model (nn.Module) or a callable function
        input_tensor: The input tensor

    Returns:
        True if non-zero gradients are propagated to the input, False otherwise
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
    if isinstance(func, torch.nn.Module):
        # Save current training state and set to eval mode
        was_training = func.training
        func.eval()  # Use eval() to freeze weights without blocking gradients

        # Disable in-place operations in ReLU, etc. to avoid view errors
        for m in func.modules():
            if hasattr(m, "inplace") and m.inplace:
                m.inplace = False

        try:
            device = next(func.parameters()).device
            x = x.to(device)
        except (StopIteration, RuntimeError):
            pass  # No parameters or other error

    try:
        # Forward pass
        outputs = func(x)

        # Calculate the loss
        if isinstance(outputs, torch.Tensor):
            loss = outputs.sum()
        elif isinstance(outputs, dict):
            tensors = [v for v in outputs.values() if isinstance(v, torch.Tensor)]
            loss = sum(t.sum() for t in tensors) if tensors else None
        elif isinstance(outputs, (list, tuple)):
            tensors = [v for v in outputs if isinstance(v, torch.Tensor)]
            loss = sum(t.sum() for t in tensors) if tensors else None
        else:
            print(f"Unsupported output type: {type(outputs)}")
            return False
        
        if loss is None:
            print("No tensor found in outputs")
            return False
            
        # Backward pass
        loss.backward()
        
        # Check if gradients were propagated
        if x.grad is not None and x.grad.abs().sum() > 0:
            print(f"Gradients OK - sum={x.grad.abs().sum().item():.6f}")
            return True

        print("No gradients or zero gradients")
        return False
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return False
    finally:
        # Restore settings
        if not had_no_grad:
            torch.set_grad_enabled(False)
            
        # Restore the original model mode
        if isinstance(func, torch.nn.Module) and 'was_training' in locals():
            func.train(was_training)