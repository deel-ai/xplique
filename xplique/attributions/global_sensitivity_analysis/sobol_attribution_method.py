"""
Sobol Attribution Method explainer
"""

from ...types import Callable, Union, Optional
from .gsa_attribution_method import GSABaseAttributionMethod
from .sobol_estimators import SobolEstimator, JansenEstimator
from .replicated_designs import ReplicatedSampler, TFSobolSequenceRS


class SobolAttributionMethod(GSABaseAttributionMethod):
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
        sampler: Optional[ReplicatedSampler] = None,
        estimator: Optional[SobolEstimator] = None,
        perturbation_function: Optional[Union[Callable, str]] = "inpainting",
        batch_size=256
    ):

        assert (nb_design & (nb_design-1) == 0) and nb_design != 0,\
            "The number of design must be a power of two."

        sampler = sampler if sampler is not None else TFSobolSequenceRS()
        estimator = estimator if estimator is not None else JansenEstimator()

        assert isinstance(sampler, ReplicatedSampler), "The sampler must be a"\
                                                       " valid Replicated Sampler."
        assert isinstance(estimator, SobolEstimator), "The estimator must be a"\
                                                      " valid Sobol estimator."

        super().__init__(model=model, sampler=sampler, estimator=estimator,
                         grid_size=grid_size, nb_design=nb_design,
                         perturbation_function=perturbation_function,
                         batch_size=batch_size)
