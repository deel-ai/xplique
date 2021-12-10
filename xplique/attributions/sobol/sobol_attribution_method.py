from math import ceil

import cv2
import numpy as np
import tensorflow as tf

from ...types import Callable, Union, Optional
from ...commons import batch_tensor
from ..base import BlackBoxExplainer, sanitize_input_output
from .estimators import SobolEstimator, JansenEstimator
from .sampling import Sampler, TFSobolSequence
from .perturbations import inpainting


class SobolAttributionMethod(BlackBoxExplainer):
    """
    Sobol' Attribution Method.

    Once the explainer is initialized, you can call it with an array of inputs and labels (int)
    to get the STi.

    Parameters
    ----------
    model
        Model used for computing explanations.
    grid_size: int, optional
        Cut the image in a grid of grid_size*grid_size to estimate an indice per cell.
    nb_design: int, optional
        Must be a power of two. Number of design, the number of forward
        will be: nb_design * (grid_size**2 + 2).
    sampler : Sampler, optional
        Sampler used to generate the (quasi-)monte carlo samples.
    estimator: Estimator, optional
        Estimator used to compute the total order sobol' indices.
    perturbation_function: function, optional
        Function to call to apply the perturbation on the input.
    batch_size: int, optional,
        Batch size to use for the forwards.
    """

    def __init__(
        self,
        model,
        grid_size: int = 8,
        nb_design: int = 64,
        sampler: Sampler = TFSobolSequence(),
        estimator: SobolEstimator = JansenEstimator(),
        perturbation_function: Callable = inpainting,
        batch_size=256
    ):

        assert (nb_design & (nb_design-1) == 0) and nb_design != 0,\
            "The number of design must be a power of two."

        super().__init__(model, batch_size)

        self.grid_size = grid_size
        self.nb_design = nb_design
        self.perturbation_function = perturbation_function

        self.sampler = sampler
        self.estimator = estimator

        self.masks = sampler(grid_size**2, nb_design).reshape((-1, grid_size, grid_size, 1))

    @sanitize_input_output
    def explain(self,
                inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                targets: Optional[Union[tf.Tensor, np.ndarray]] = None) -> tf.Tensor:
        """
        Explain a particular prediction

        Parameters
        ----------
        inputs: ndarray or tf.Tensor [Nb_samples, Width, Height, Channels]
            Images to explain.
        labels: list of int,
            Label of the class to explain.
        """
        input_shape = (inputs.shape[1], inputs.shape[2])
        sobol_maps = None

        for input, target in zip(inputs, targets):

            perturbator = self.perturbation_function(input)
            outputs = None

            for batch_masks in batch_tensor(self.masks, self.batch_size):

                batch_y = SobolAttributionMethod._batch_forward(self.model, batch_masks,
                                                                perturbator, input_shape)
                outputs = batch_y if outputs is None else tf.concat([outputs, batch_y], axis=0)

            sobol_map = self.estimator(self.masks, outputs, self.nb_design)
            sobol_map = cv2.resize(sobol_map, input_shape, interpolation=cv2.INTER_CUBIC)

            sobol_maps = sobol_map if sobol_maps is None else tf.concat([sobol_maps, sobol_map],
                                                                         axis=0)

        return sobol_maps

    @staticmethod
    @tf.function
    def _batch_forward(model, masks, perturbator, input_shape):
        upsampled_masks = tf.image.resize(masks, input_shape)
        perturbated_inputs = perturbator(upsampled_masks)
        outputs = model(perturbated_inputs)
        return outputs
