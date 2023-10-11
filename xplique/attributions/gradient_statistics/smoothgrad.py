"""
Module related to SmoothGrad method
"""

from .gradient_statistic import GradientStatistic
from ...commons.online_statistics import OnlineMean


class SmoothGrad(GradientStatistic):
    """
    Used to compute the SmoothGrad, by averaging Saliency maps of noisy samples centered on the
    original sample.

    Ref. Smilkov & al., SmoothGrad: removing noise by adding noise (2017).
    https://arxiv.org/abs/1706.03825

    Parameters
    ----------
    model
        The model from which we want to obtain explanations
    output_layer
        Layer to target for the outputs (e.g logits or after softmax).
        If an `int` is provided it will be interpreted as a layer index.
        If a `string` is provided it will look for the layer name.

        Default to the last layer.
        It is recommended to use the layer before Softmax.
    batch_size
        Number of inputs to explain at once, if None compute all at once.
    operator
        Function g to explain, g take 3 parameters (f, x, y) and should return a scalar,
        with f the model, x the inputs and y the targets. If None, use the standard
        operator g(f, x, y) = f(x)[y].
    nb_samples
        Number of noisy samples generated for the smoothing procedure.
    noise
        Scalar, noise used as standard deviation of a normal law centered on zero.
    """

    def _get_online_statistic_class(self) -> type:
        """
        Specify the online statistic (mean) for the parent class `__init__`.

        Returns
        -------
        online_statistic_class
            Class of the online statistic used to aggregated gradients on perturbed inputs.
        """
        return OnlineMean
