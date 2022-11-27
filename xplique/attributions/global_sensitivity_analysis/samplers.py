"""
Sampling methods
"""

from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
import scipy


class Sampler(ABC):
    """
    Base class for sampling.
    """

    def __init__(self, binary=False):
        self.binary = binary

    @abstractmethod
    def __call__(self, dimension, nb_design) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def make_binary(masks):
        """
        Transform [0, 1]^d Masks into binary {0, 1}^d masks.

        Parameters
        ----------
        masks
            Low resolution masks (before upsampling).

        Returns
        -------
        masks
            Binary masks.
        """
        return np.round(masks)


class ScipySampler(Sampler):
    """
    Base class based on Scipy qmc module for sampling.
    """

    def __init__(self, binary=False):
        super().__init__(binary)

        try:
            self.qmc = scipy.stats.qmc  # pylint: disable=E1101
        except AttributeError as err:
            raise ModuleNotFoundError(
                "Xplique need scipy>=1.7 to use this sampling."
            ) from err


class TFSobolSequence(Sampler):
    """
    Tensorflow Sobol LP tau sequence sampler.

    Ref. I. M. Sobol., The distribution of points in a cube and the accurate evaluation of
    integrals (1967).
    https://www.sciencedirect.com/science/article/abs/pii/0041555367901449
    """

    def __call__(self, dimension, nb_design):

        points = tf.math.sobol_sample(dimension, nb_design, dtype=tf.float32).numpy()
        if self.binary:
            points = self.make_binary(points)

        return points


class ScipySobolSequence(ScipySampler):
    """
    Scipy Sobol LP tau sequence sampler.

    Ref. I. M. Sobol., The distribution of points in a cube and the accurate evaluation of
    integrals (1967).
    https://www.sciencedirect.com/science/article/abs/pii/0041555367901449
    """

    def __call__(self, dimension, nb_design):
        sampler = self.qmc.Sobol(dimension, scramble=False)
        points = sampler.random(nb_design).astype(np.float32)
        if self.binary:
            points = self.make_binary(points)

        return points


class HaltonSequence(ScipySampler):
    """
    Halton sequence sampler.

    Ref. J.H. Halton., On the efficiency of certain quasi-random sequences of points in evaluating
    multi-dimensional integrals (1960).
    https://link.springer.com/article/10.1007/BF01386213
    """

    def __call__(self, dimension, nb_design):
        sampler = self.qmc.Halton(dimension, scramble=False)
        points = sampler.random(nb_design).astype(np.float32)
        if self.binary:
            points = self.make_binary(points)

        return points


class LatinHypercube(ScipySampler):
    """
    Latin hypercube replicated sampler.

    Ref. Mckay & al., A Comparison of Three Methods for Selecting Values of Input Variables in the
    Analysis of Output from a Computer Code (1979).
    https://www.jstor.org/stable/1268522
    """

    def __call__(self, dimension, nb_design):
        sampler = self.qmc.LatinHypercube(dimension)
        points = sampler.random(nb_design).astype(np.float32)
        if self.binary:
            points = self.make_binary(points)

        return points
