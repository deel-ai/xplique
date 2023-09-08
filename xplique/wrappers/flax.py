"""
Module for having a wrapper for Jax/Flax's model
"""
import warnings

import tensorflow as tf
import numpy as np

from ..types import Any, Tuple, Callable

class FlaxWrapper(tf.keras.Model):
    """
    A wrapper for Flax's modules so that they can be used in Xplique framework
    for most attribution methods

    The nn.Module is assumed to be in evaluation mode: model.apply will be used with `mutable=False`.

    Parameters
    ----------
    flax_model
        A nn.Module from flax.linen
    parameters
        A pytree of parameters. This can be any nested structure of dicts, listsm, tuples, namedtuples, etc.
          It must contains at least the key `params`.
          If the model has a batch norm, it may also contains the key `batch_stats`.
    """

    def __init__(self, flax_module: "nn.Module", parameters: Any
                 ): # pylint: disable=C0415,C0103

        super().__init__()

        try:
            # use Flax/Jax functionality
            import jax
            self.jax = jax
            self.jnp = jax.numpy
        except ImportError as exc:
            raise ImportError(
            "Flax is required to use this feature. \
             Please install Flax using 'pip install flax'."
            ) from exc

        self.model = flax_module
        self.parameters = parameters

        # compile the forward pass with fixed parameters
        self.closure = self._forward_closure_over_parameters()

        # Flax follows the convention of Tensorflow (N, H, W, C)
        self.channel_first = False

        # deactivate all tf.function
        tf.config.run_functions_eagerly(True)
        warnings.warn("TF is set to run eagerly to avoid conflict with Flax. Thus,\
                       TF functions might be slower")
    
    def _forward_closure_over_parameters(self):
        def _closure(jax_inputs):
          return self.model.apply(self.parameters, jax_inputs, mutable=False)
        return self.jax.jit(_closure)

    # pylint: disable=arguments-differ
    @tf.custom_gradient
    def call(self, inputs: np.ndarray) -> Tuple[tf.Tensor, Callable]:
        """
        A method that allow to call the Flax module wrapped on inputs that are
        numpy arrays, get the gradients from Flax framework and recast outputs and gradients
        as Tensorflow tensors.

        Parameters
        ----------
        inputs
            Processed inputs as numpy arrays, or list of numpy arrays.

        Returns
        -------
        outputs
            Outputs of the Flax module converted to Tensorflow tensor
        grad
            The function that allow to compute the gradient of the Flax module and
            broadcast it for Tensorflow
        """
        jax_inputs = self.jnp.array(inputs)
        # backward pass is a vector-Jacobian product (vjp)
        outputs, vjp_fun = self.jax.vjp(self.closure, jax_inputs)
        output_tensor = tf.constant(outputs)

        def grad(upstream):
            cotangent = self.jnp.array(upstream)
            # vjp_fun returns a tuple of length 1, so we extract the first element
            tangent = vjp_fun(cotangent)[0]
            dy_dx = tf.constant(tangent)
            return dy_dx

        return output_tensor, grad
