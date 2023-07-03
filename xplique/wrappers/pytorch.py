"""
Module for having a wrapper for PyTorch's model
"""

import tensorflow as tf
import numpy as np

try:
    # use PyTorch functionality
    import torch
    from torch import nn
    from torch import from_numpy
except ImportError as exc:
    raise ImportError(
       "PyTorch is required to use this feature. Please install PyTorch using 'pip install torch'."
    ) from exc

from ..types import Union, Optional, Tuple, Callable

class TorchWrapper(tf.keras.Model):
    """
    A wrapper for PyTorch's model so that they can be used in Xplique framework
    for most attributions method

    Parameters
    ----------
    torch_model
        A nn.Module PyTorch model
    device
        If we are on GPU or CPU
    is_channel_first
        A boolean that is true if the torch's model expect a channel dim and if this one come first
    """

    def __init__(self, torch_model: nn.Module, device: Union['torch.device', str],
                 is_channel_first: Optional[bool] = None):
        super().__init__()
        self.model = torch_model.to(device)
        self.device = device
        # with torch, the convention for CNN is (N, C, H, W)
        if is_channel_first is None:
            self.channel_first = self._check_conv_layers()
        else:
            self.channel_first = is_channel_first
        # deactivate all tf.function
        tf.config.run_functions_eagerly(True)

    # pylint: disable=arguments-differ
    @tf.custom_gradient
    def call(self, inputs: np.ndarray) -> Tuple[tf.Tensor, Callable]:
        """
        A method that allow to call the PyTorch model wrapped on inputs that are
        numpy arrays reshaped to match Xplique's explainers expectations (i.e. channel-last)
        , get the gradients from PyTorch framework and recast outputs and gradients
        as Tensorflow tensors.

        Parameters
        ----------
        inputs
            Processed inputs as numpy arrays
        
        Returns
        -------
        outputs
            Outputs of the PyTorch model converted to Tensorflow tensor
        grad
            The function that allow to compute the gradient of the PyTorch model and
            broadcast it for Tensorflow
        """
        # transform your numpy inputs to torch
        torch_inputs = self.transform_np_inputs(inputs).to(self.device)
        torch_inputs.requires_grad_(True)

        # make predictions
        self.model.zero_grad()
        outputs = self.model(torch_inputs)
        output_tensor = tf.constant(outputs.cpu().detach().numpy())

        def grad(upstream):
            upstream_tensor = tf.constant(upstream.numpy())
            torch.autograd.backward(
                outputs.cpu(),
                grad_tensors=from_numpy(upstream_tensor.numpy()),
                retain_graph=True
                )
            dx_torch = torch_inputs.grad

            dx_np = dx_torch.cpu().detach().numpy()
            if self.channel_first:
                # (N, C, H, W) -> (N, H, W, C) for explainer
                dx_np = np.moveaxis(dx_np, [1, 2, 3], [3, 1, 2])

            gradient = tf.constant(dx_np)
            return gradient

        return output_tensor, grad

    def transform_np_inputs(self, np_inputs: np.ndarray):
        """
        Methods that transform inputs as expected by the explainer to inputs expected
        by your PyTorch model.

        Notes
        -----
        If your model (or your data pipeline) has more specifities, change this function to make
        the bridge

        Parameters
        ----------
        np_inputs
            Inputs as numpy arrays (simpler for conversion between PyTorch
            and Tensorflow)

        Returns
        -------
        torch_inputs
            Torch Tensor with the format expected by the PyTorch model
        """
        if self.channel_first:
            # go from channel last to channel first for the torch model
            np_inputs = np.moveaxis(np_inputs, [3, 1, 2], [1, 2, 3])
        else:
            np_inputs = np.asarray(np_inputs)

        # convert numpy array to torch tensor
        torch_inputs = torch.Tensor(np_inputs)

        return torch_inputs

    def _check_conv_layers(self):
        """
        A method that checks if the PyTorch's model has 2D convolutional layer. 
        Indeed, convolution with PyTorch expects inputs in the shape (N, C, H, W)
        where TF expect (N, H, W, C).

        Returns
        -------
        has_conv_layers
            A boolean that says if the PyTorch model has Conv2d layers
        """
        has_conv_layers = False

        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                has_conv_layers = True
                break

        return has_conv_layers
