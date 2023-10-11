"""
GSA Attribution Method base explainer
"""

from enum import Enum

import numpy as np
import tensorflow as tf

from ...types import Callable, Union, Optional, Tuple, OperatorSignature
from ...commons import batch_tensor, repeat_labels, Tasks
from ..base import BlackBoxExplainer, sanitize_input_output
from .perturbations import amplitude, inpainting, blurring


class PerturbationFunction(Enum):
    """
    GSA Perturbation function interface.
    """

    INPAINTING = inpainting
    BLURRING = blurring
    AMPLITUDE = amplitude

    @staticmethod
    def from_string(perturbation_function: str) -> "PerturbationFunction":
        """
        Restore a perturbation function from a string.

        Parameters
        ----------
        perturbation_function
            String indicating the perturbation function to restore: must be one
            of 'inpainting', 'blurring', or 'amplitude'.

        Returns
        -------
        perturbation_function
            The PerturbationFunction object.
        """
        assert perturbation_function in [
            "inpainting",
            "blurring",
            "amplitude",
        ], "Only 'inpainting', 'blurring' and 'amplitude' are supported."

        if perturbation_function == "amplitude":
            return PerturbationFunction.AMPLITUDE
        if perturbation_function == "blurring":
            return PerturbationFunction.BLURRING
        return PerturbationFunction.INPAINTING


class GSABaseAttributionMethod(BlackBoxExplainer):
    """
    GSA base Attribution Method.
    Base explainer for all the attribution method based on Global Sensitivity Analysis.

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
        Sampler function to call to generate masks.
    estimator
        Estimator used to compute the attribution score, e.g Sobol or HSIC estimator.
    perturbation_function
        Function to call to apply the perturbation on the input. Can also be string:
        'inpainting', 'blurring', or 'amplitude'.
    batch_size
        Batch size to use for the forwards.
    operator
        Function g to explain, g take 3 parameters (f, x, y) and should return a scalar,
        with f the model, x the inputs and y the targets. If None, use the standard
        operator g(f, x, y) = f(x)[y].
    """

    def __init__(
        self,
        model: tf.keras.Model,
        sampler: Callable,
        estimator: Callable,
        grid_size: int = 7,
        nb_design: int = 32,
        perturbation_function: Optional[Union[Callable, str]] = "inpainting",
        batch_size=256,
        operator: Optional[Union[Tasks, str, OperatorSignature]] = None,
    ):

        super().__init__(model, batch_size, operator)

        self.grid_size = grid_size
        self.nb_design = nb_design

        if isinstance(perturbation_function, str):
            self.perturbation_function = PerturbationFunction.from_string(
                perturbation_function
            )
        else:
            self.perturbation_function = perturbation_function

        self.sampler = sampler
        self.estimator = estimator

        self.masks = self.sampler(grid_size**2, nb_design).reshape(
            (-1, grid_size, grid_size, 1)
        )

    @sanitize_input_output
    def explain(
        self,
        inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
    ) -> tf.Tensor:
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
        attributions_maps
            GSA Attribution Method explanations, same shape as the inputs except for the channels.
        """
        # pylint: disable=E1101

        input_shape = (inputs.shape[1], inputs.shape[2])
        heatmaps = None

        for inp, target in zip(inputs, targets):

            perturbator = self.perturbation_function(inp)
            outputs = None

            for batch_masks in batch_tensor(self.masks, self.batch_size):

                batch_x, batch_y = self._batch_perturbations(
                    batch_masks, perturbator, target, input_shape
                )
                batch_outputs = self.inference_function(self.model, batch_x, batch_y)

                outputs = (
                    batch_outputs
                    if outputs is None
                    else tf.concat([outputs, batch_outputs], axis=0)
                )

            heatmap = self.estimator(self.masks, outputs, self.nb_design)
            heatmap = tf.image.resize(heatmap, input_shape, method=tf.image.ResizeMethod.BICUBIC)
            heatmap = heatmap[tf.newaxis]

            heatmaps = (
                heatmap if heatmaps is None else tf.concat([heatmaps, heatmap], axis=0)
            )

        return heatmaps

    @staticmethod
    @tf.function
    def _batch_perturbations(
        masks: tf.Tensor,
        perturbator: Callable,
        target: tf.Tensor,
        input_shape: Tuple[int, int],
    ) -> Union[tf.Tensor, tf.Tensor]:
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
