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
    Base class used by `NaturalExampleBasedExplainer` to projet samples to a meaningfull space
    for the model to explain.
    
    Projection have two parts a `space_projection` and `weights`, to apply a projection,
    the samples are first projected to a new space and then weighted.
    Either the `space_projection` or the `weights` could be `None` but,
    if both are, the projection is an identity function.
    
    At least one of the two part should include the model in the computation
    for distance between projected elements to make sense for the model.
    
    Note that the cost of this projection should be limited
    as it will be applied to all samples of the train dataset.
    
    Parameters
    ----------
    weights
        Either a Tensor or a Callable.
        - In the case of a Tensor, weights are applied in the projected space (after `space_projection`).
        Hence weights should have the same shape as a `projected_input`.
        - In the case of a Callable, the function should return the weights when called,
        as a way to get the weights (a Tensor)
        It is pertinent in the case on weights dependent on the inputs, i.e. local weighting.
        
        Example of Callable:
        ```
        def get_weights_example(projected_inputs: Union(tf.Tensor, np.ndarray), 
                                targets: Union(tf.Tensor, np.ndarray) = None):
            '''
            Example of function to get weights,
            projected_inputs are the elements for which weights are comlputed.
            targets are optionnal additionnal parameters for weights computation.
            '''
            weights = ...  # do some magic with inputs and targets, it should use the model.
            return weights
        ```
    space_projection
        Callable that take samples and return a Tensor in the projected sapce.
        An example of projected space is the latent space of a model.
        In this case, the model should be splitted and the 
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
        For visualization purpose (and only), we may be interested to project weights
        from the projected space to the input space.
        This is applied only if their is a difference in dimention.
        We assume here that we are treating images and an upsampling is applied.
        
        Parameters
        ----------
        inputs
            Dataset, Tensor or Array. Input samples to be explained.
            If Dataset, targets should not be provided (included in Dataset).
            Expected shape among (N, W), (N, T, W), (N, W, H, C).
            More information in the documentation.
        targets
            Additional parameter for `self.get_weights` function.
            
        Returns
        -------
        input_weights
            Tensor with the same dimension as `inputs` modulo the channels.
            They are an upsampled version of the actual weights used in the projection.
        """
        projected_inputs = self.space_projection(inputs)
        weights = self.get_weights(projected_inputs, targets)
        
        # take mean over channels for images
        channel_mean_fn = lambda: tf.reduce_mean(weights, axis=-1, keepdims=True)
        weights = tf.cond(pred=tf.shape(weights).shape[0] < 4,
                          true_fn=lambda: weights,
                          false_fn=channel_mean_fn)

        # resizing
        resize_fn = lambda: tf.image.resize(weights, inputs.shape[1:-1], method="bicubic")
        input_weights = tf.cond(pred=projected_inputs.shape==inputs.shape,
                                true_fn=lambda: weights,
                                false_fn=resize_fn,)
        return input_weights

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
            Additional parameter for `self.get_weights` function.

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
