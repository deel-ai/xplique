"""
Sobol Attribution Method explainer
"""

from enum import Enum

import cv2
import numpy as np
import tensorflow as tf

from ...types import Callable, Union, Optional, Tuple
from ...commons import batch_tensor, repeat_labels
from ..base import BlackBoxExplainer, sanitize_input_output
from .estimators import SobolEstimator, JansenEstimator
from .sampling import Sampler, TFSobolSequence
from .perturbations import amplitude, inpainting, blurring


class PerturbationFunction(Enum):
    """
    Sobol Perturbation function interface.
    """
    INPAINTING = inpainting
    BLURRING   = blurring
    AMPLITUDE  = amplitude

    @staticmethod
    def from_string(perturbation_function: str) -> 'PerturbationFunction':
        """
        Restore a perturbation function from a string.

        Parameters
        ----------
        perturbation_function
            String indicating the perturbation function to restore: must be one
            of 'inpainting', 'blurring' or 'amplitude'.

        Returns
        -------
        perturbation_function
            The PerturbationFunction object.
        """
        assert perturbation_function in ['inpainting', 'blurring', 'amplitude'], \
            "Only 'inpainting', 'blurring' and 'amplitude' are supported."

        if perturbation_function == 'amplitude':
            return PerturbationFunction.AMPLITUDE
        if perturbation_function == 'blurring':
            return PerturbationFunction.BLURRING
        return PerturbationFunction.INPAINTING


class SobolAttributionMethod(BlackBoxExplainer):
    """
    Sobol' Attribution Method.
    Compute the total order Sobol' indices using a perturbation function on a grid and an
    adapted sampling as described in the original paper.

    Ref. Fel, Cadène, Chalvidal & al., Look at the Variance! Efficient Black-box Explanations
    with Sobol-based Sensitivity Analysis, NeurIPS (2021), https://arxiv.org/abs/2111.04138

    Parameters
    ----------
    model
        Model used for computing explanations.
    grid_size
        Cut the image in a grid of (grid_size, grid_size) to estimate an indice per cell.
    nb_design
        Must be a power of two. Number of design, the number of forward
        will be: nb_design * (grid_size**2 + 2). Generally not above 32.
    sampler
        Sampler used to generate the (quasi-)monte carlo samples, QMC (sobol sequence
        recommended). For more option, see the sampler module.
    estimator
        Estimator used to compute the total order sobol' indices, Jansen recommended. For more
        option, see the estimator module.
    perturbation_function
        Function to call to apply the perturbation on the input. Can also be string in
        'inpainting', 'blur'.
    batch_size
        Batch size to use for the forwards.
    """

    def __init__(
        self,
        model,
        grid_size: int = 8,
        nb_design: int = 32,
        sampler: Optional[Sampler] = None,
        estimator: Optional[SobolEstimator] = None,
        perturbation_function: Optional[Union[Callable, str]] = "inpainting",
        batch_size=256
    ):

        assert (nb_design & (nb_design-1) == 0) and nb_design != 0,\
            "The number of design must be a power of two."

        super().__init__(model, batch_size)

        self.grid_size = grid_size
        self.nb_design = nb_design

        if isinstance(perturbation_function, str):
            self.perturbation_function = PerturbationFunction.from_string(perturbation_function)
        else:
            self.perturbation_function = perturbation_function

        self.sampler = sampler if sampler is not None else TFSobolSequence()
        self.estimator = estimator if estimator is not None else JansenEstimator()

        self.masks = self.sampler(grid_size**2, nb_design).reshape((-1, grid_size, grid_size, 1))

    @sanitize_input_output
    def explain(self,
                inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                targets: Optional[Union[tf.Tensor, np.ndarray]] = None) -> tf.Tensor:
        """
        Compute the total Sobol' indices according to the explainer parameter (perturbation
        function, grid size...). Accept Tensor, numpy array or tf.data.Dataset (in that case
        targets is None).

        Parameters
        ----------
        inputs
            Images to be explained, either tf.dataset, Tensor or numpy array.
            If Dataset, targets should not be provided (included in Dataset).
            Expected shape (N, W, H, C) or (N, W, H).
        targets
            One-hot encoding for classification or direction {-1, +1} for regression.
            Tensor or numpy array.
            Expected shape (N, C) or (N).

        Returns
        -------
        sobol_maps
            Sobol Attribution Method explanations, same shape as the inputs except for the channels.
        """
        input_shape = (inputs.shape[1], inputs.shape[2])
        sobol_maps = None

        for inp, target in zip(inputs, targets):

            perturbator = self.perturbation_function(inp)
            outputs = None

            for batch_masks in batch_tensor(self.masks, self.batch_size):

                batch_x, batch_y = SobolAttributionMethod._batch_perturbations(batch_masks,
                                                                               perturbator,
                                                                               target, input_shape)
                batch_outputs = self.inference_function(self.model, batch_x, batch_y)

                outputs = batch_outputs if outputs is None else tf.concat([outputs,
                                                                           batch_outputs], axis=0)

            sobol_map = self.estimator(self.masks, outputs, self.nb_design)
            sobol_map = cv2.resize(sobol_map, input_shape, interpolation=cv2.INTER_CUBIC)[None,:,:] # pylint: disable=E1101

            sobol_maps = sobol_map if sobol_maps is None else tf.concat([sobol_maps, sobol_map],
                                                                         axis=0)

        return sobol_maps

    @staticmethod
    @tf.function
    def _batch_perturbations(masks: tf.Tensor, perturbator: Callable, target: tf.Tensor,
                             input_shape: Tuple[int, int]) -> Union[tf.Tensor, tf.Tensor]:
        """
        Prepare perturbated input and replicated targets before a batch inference.

        Parameters
        ----------
        masks
            Perturbation masks in lower dimensions (grid_size, grid_size).
        perturbator
            Perturbation function to be called with the upsampled masks.
        target
            Label of a single prediction
        input_shape
            Shape of a single input

        Returns
        -------
        perturbated_inputs
            One inputs perturbated for each masks, according to the pertubation function
            modulated by the masks values.
        repeated_targets
            Replicated labels, one for each masks.
        """
        repeated_targets = repeat_labels(target[None, :], len(masks))

        upsampled_masks = tf.image.resize(masks, input_shape, method="nearest")
        perturbated_inputs = perturbator(upsampled_masks)

        return perturbated_inputs, repeated_targets
