"""
Sampling methods for replicated designs
"""

from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf


class Sampler(ABC):
    """
    Base class for replicated design sampling.
    """

    def _build_replicated_design(self, a, b):
        """
        Build the replicated design matrix C using A & B

        Parameters
        ----------
        a: ndarray
          The masks values for the sampling matrix A.
        b: ndarray
          The masks values for the sampling matrix B.

        Returns
        -------
        c: ndarray
          The new replicated design matrix C generated from A & B.
        """
        c = np.array([a.copy() for _ in range(a.shape[-1])])
        for i in range(len(c)):
            c[i, :, i] = b[:, i]

        c = c.reshape((-1, a.shape[-1]))
        return c

    @abstractmethod
    def __call__(self, dimension, nb_design):
        raise NotImplementedError()


class TFSobolSequence(Sampler):
    """
    Tensorflow Sobol LP tau sequence sampler.

    Ref. I. M. Sobol., The distribution of points in a cube and the accurate evaluation of
    integrals (1967).
    https://www.sciencedirect.com/science/article/abs/pii/0041555367901449
    """

    def __call__(self, dimension, nb_design):
        ab = tf.math.sobol_sample(dimension*2, nb_design).numpy()
        a, b = ab[:, :dimension], ab[:, dimension:]
        c = self._build_replicated_design(a, b)

        return np.concatenate([a, b, c], 0)


class ScipySobolSequence(Sampler):
    """
    Scipy Sobol LP tau sequence sampler.

    Ref. I. M. Sobol., The distribution of points in a cube and the accurate evaluation of
    integrals (1967).
    https://www.sciencedirect.com/science/article/abs/pii/0041555367901449
    """

    def __call__(self, dimension, nb_design):
        sampler = qmc.Sobol(dimension*2, scramble=False)
        ab = sampler.random(nb_design)
        a, b = ab[:, :dimension], ab[:, dimension:]
        c = self._build_replicated_design(a, b)

        return np.concatenate([a, b, c], 0)


class HaltonSequence(Sampler):
    """
    Halton sequence sampler.

    Ref. J.H. Halton., On the efficiency of certain quasi-random sequences of points in evaluating
    multi-dimensional integrals (1960).
    https://link.springer.com/article/10.1007/BF01386213
    """

    def __call__(self, dimension, nb_design):
        sampler = qmc.Halton(dimension*2, scramble=False)
        ab = sampler.random(nb_design)
        a, b = ab[:, :dimension], ab[:, dimension:]
        c = self._build_replicated_design(a, b)

        return np.concatenate([a, b, c], 0)


class LHSampler(Sampler):
    """
    Latin hypercube sampler.

    Ref. Mckay & al., A Comparison of Three Methods for Selecting Values of Input Variables in the
    Analysis of Output from a Computer Code (1979).
    https://www.jstor.org/stable/1268522
    """

    def __call__(self, dimension, nb_design):
        sampler = qmc.LatinHypercube(dimension*2)
        ab = sampler.random(nb_design)
        a, b = ab[:, :dimension], ab[:, dimension:]
        c = self._build_replicated_design(a, b)

        return np.concatenate([a, b, c], 0)
