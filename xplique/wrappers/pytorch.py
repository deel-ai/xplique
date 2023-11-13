"""
Module for having a wrapper for PyTorch models
"""
import warnings

import tensorflow as tf
import numpy as np

from ..types import Union, Optional, Tuple, Callable

class TorchWrapper(tf.keras.Model):
    """
    A wrapper for PyTorch models so that they can be used in Xplique framework
    for most attribution methods

    Parameters
    ----------
    torch_model
        A nn.Module PyTorch model
    device
        If we are on GPU or CPU
    is_channel_first
        A boolean that is true if the torch's model expect a channel dim and if this one come first
    """

    def __init__(self, torch_model: "nn.Module", device: Union["torch.device", str],
                 is_channel_first: Optional[bool] = None
                 ): # pylint: disable=C0415,C0103,W0719

        try:
            super().__init__()
        except tf.errors.InternalError as error:
            raise Exception("If you have a tensorflow InternalError with cudaGetDevice() here, \
            it is possible that importing tensorflow before torch might resolve the issue."
            ) from error

        try:
            # use PyTorch functionality
            import torch
            from torch import nn
            from torch import from_numpy
            self.torch = torch
            self.nn = nn
            self.from_numpy = from_numpy
        except ImportError as exc:
            raise ImportError(
            "PyTorch is required to use this feature. \
             Please install PyTorch using 'pip install torch'."
            ) from exc

        assert not(torch_model.training), "Please provide a torch module in eval mode"
        self.model = torch_model.to(device)
        self.device = device
        # with torch, the convention for CNN is (N, C, H, W)
        if is_channel_first is None:
            self.channel_first = self._has_conv_layers()
        else:
            self.channel_first = is_channel_first
        # deactivate all tf.function
        tf.config.run_functions_eagerly(True)
        warnings.warn("TF is set to run eagerly to avoid conflict with PyTorch. Thus,\
                       TF functions might be slower")

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
        torch_inputs = self.np_img_to_torch(inputs).to(self.device)
        torch_inputs.requires_grad_(True)

        # make predictions
        self.model.zero_grad()
        outputs = self.model(torch_inputs)
        output_tensor = tf.constant(outputs.cpu().detach().numpy())

        def grad(upstream):
            self.torch.autograd.backward(
                outputs,
                grad_tensors=self.from_numpy(upstream.numpy()).to(self.device),
                retain_graph=False
            )
            dx_torch = torch_inputs.grad

            dx_np = dx_torch.cpu().detach().numpy()
            if self.channel_first:
                # (N, C, H, W) -> (N, H, W, C) for explainer
                dx_np = np.moveaxis(dx_np, [1, 2, 3], [3, 1, 2])

            gradient = tf.constant(dx_np)
            return gradient

        return output_tensor, grad

    def np_img_to_torch(self, np_inputs: np.ndarray):
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
        torch_inputs = self.torch.Tensor(np_inputs)

        return torch_inputs

    def _has_conv_layers(self):
        """
        A method that checks if the PyTorch models has 2D convolutional layer.
        Indeed, convolution with PyTorch expects inputs in the shape (N, C, H, W)
        where TF expect (N, H, W, C).

        Returns
        -------
        has_conv_layers
            A boolean that says if the PyTorch model has Conv2d layers
        """
        has_conv_layers = False

        for module in self.model.modules():
            if isinstance(module, self.nn.Conv2d):
                has_conv_layers = True
                break

        return has_conv_layers
