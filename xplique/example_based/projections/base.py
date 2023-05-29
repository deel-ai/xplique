"""
Base model for example-based 
"""

from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np

from ...attributions.base import sanitize_input_output
from ...types import Callable, Dict, Tuple, Union, Optional


class Projection(Callable):
    """
    ...
    """
    def __init__(self,
                 weights: Union[Callable, tf.Tensor, np.ndarray] = None,
                 space_projection: Callable = None):
        # assert weights is not None and space_projection is not None

        # Set weights or 
        if isinstance(weights, Callable):
            # weights is a function
            self.get_weights = weights
        else:
            if weights is None:
                # no weights
                self.get_weights = lambda inputs, targets=None: tf.ones(tf.shape(inputs))
            elif isinstance(weights, tf.Tensor) or isinstance(weights, np.ndarray):
                # weights is a tensor
                if isinstance(weights, np.ndarray):
                    weights = tf.convert_to_tensor(weights, dtype=tf.float32)
                # define a function that returns the weights
                def get_weights(inputs, targets=None):
                    nweights = tf.expand_dims(weights, axis=0)
                    return tf.repeat(nweights, tf.shape(inputs)[0], axis=0)
                self.get_weights = get_weights
            else:
                raise TypeError("`weights` should be a tensor or a `Callable`,"+\
                                f"not a {type(weights)}")
        
        # Set space_projection
        if space_projection is None:
            self.space_projection = lambda inputs: inputs
        elif isinstance(space_projection, Callable):
            self.space_projection = space_projection
        else:
            raise TypeError("`space_projection` should be a `Callable`,"+\
                            f"not a {type(space_projection)}")
        
        
    def get_input_weights(self,
                          inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                          targets: Optional[Union[tf.Tensor, np.ndarray]] = None):
        """
        if model is splited, input weights and weights are different 
        now limited to images
        ! if projected_inputs.shape==inputs.shape no resizing
        ...
        """
        projected_inputs = self.space_projection(inputs)
        weights = self.get_weights(projected_inputs, targets)

        # resizing
        resize_fn = tf.image.resize(weights, inputs.shape[1:-1], method="bicubic")
        weights = tf.cond(pred=projected_inputs.shape==inputs.shape,
                          true_fn=lambda: weights,
                          false_fn=lambda: resize_fn,)
        return weights

    @sanitize_input_output
    def project(self,
                inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                targets: Optional[Union[tf.Tensor, np.ndarray]] = None):
        """
        Project samples in a space meaningful for the model,
        either by weights the inputs, projecting in a latent space or both.
        This function should be called at the init and for each explanation.

        Parameters
        ----------
        inputs
            Dataset, Tensor or Array. Input samples to be explained.
            If Dataset, targets should not be provided (included in Dataset).
            Expected shape among (N, W), (N, T, W), (N, W, H, C).
            More information in the documentation.
        targets
            Additional parameter for some get_weights.

        Returns
        -------
        projected_samples
            The samples projected in the new space.
        """
        projected_inputs = self.space_projection(inputs)
        weights = self.get_weights(projected_inputs, targets)
        
        return tf.multiply(weights, projected_inputs)
    
    def __call__(self,
                 inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 targets: Optional[Union[tf.Tensor, np.ndarray]] = None):
        """project alias"""
        return self.project(inputs, targets)
