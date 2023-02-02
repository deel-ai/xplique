"""
Hsic Attribution Method explainer
"""

import tensorflow as tf

from ...types import Callable, Union, Optional
from .gsa_attribution_method import GSABaseAttributionMethod
from .samplers import Sampler, TFSobolSequence
from .hsic_estimators import (
    BinaryEstimator,
    HsicEstimator,
)


class HsicAttributionMethod(GSABaseAttributionMethod):
    """
    HSIC Attribution Method.
    Compute the dependance of each input dimension wrt the output using Hilbert-Schmidt Independance
    Criterion, a perturbation function on a grid and an adapted sampling as described in
    the original paper.

    Ref. Novello, Fel, Vigouroux, Making Sense of Dependance: Efficient Black-box Explanations
    Using Dependence Measure, https://arxiv.org/abs/2206.06219

    Parameters
    ----------
    model
        Model used for computing explanations.
    grid_size
        Cut the image in a grid of (grid_size, grid_size) to estimate an indice per cell.
    nb_design
        Number of design for the sampler.
    sampler
        Sampler used to generate the (quasi-)monte carlo samples, LHS or QMC.
        For more option, see the sampler module. Note that the original paper uses LHS but here
        the default sampler is TFSobolSequence as LHS requires scipy 1.7.0.
    estimator
        Estimator used to compute the HSIC score.
    perturbation_function
        Function to call to apply the perturbation on the input. Can also be string in
        'inpainting', 'blur'.
    batch_size
        Batch size to use for the forwards.
    operator
        Function g to explain, g take 3 parameters (f, x, y) and should return a scalar,
        with f the model, x the inputs and y the targets. If None, use the standard
        operator g(f, x, y) = f(x)[y].
    """

    def __init__(
        self,
        model,
        grid_size: int = 8,
        nb_design: int = 500,
        sampler: Optional[Sampler] = None,
        estimator: Optional[HsicEstimator] = None,
        perturbation_function: Optional[Union[Callable, str]] = "inpainting",
        batch_size=256,
        operator: Optional[Callable[[tf.keras.Model, tf.Tensor, tf.Tensor], float]] = None,
    ):

        sampler = sampler if sampler is not None else TFSobolSequence(binary=True)
        estimator = (
            estimator if estimator is not None else BinaryEstimator(output_kernel="rbf")
        )

        assert isinstance(sampler, Sampler), "The sampler must be a valid Sampler."
        assert isinstance(
            estimator, HsicEstimator
        ), "The estimator must be a valid HsicEstimator."

        if isinstance(estimator, BinaryEstimator):
            assert sampler.binary, "The sampler must be binary for BinaryEstimator."

        super().__init__(model = model, operator = operator,
                         sampler = sampler, estimator = estimator,
                         grid_size = grid_size, nb_design = nb_design,
                         perturbation_function = perturbation_function, batch_size = batch_size,
        )
