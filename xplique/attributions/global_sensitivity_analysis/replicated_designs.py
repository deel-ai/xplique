"""
Sampling methods for replicated designs
"""

from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
import scipy


class ReplicatedSampler(ABC):
    """
    Base class for replicated design sampling.
    """

    @staticmethod
    def build_replicated_design(sampling_a, sampling_b):
        """
        Build the replicated design matrix C using A & B

        Parameters
        ----------
        sampling_a
          The masks values for the sampling matrix A.
        sampling_b
          The masks values for the sampling matrix B.

        Returns
        -------
        replication_c
          The new replicated design matrix C generated from A & B.
        """
        replication_c = np.array([sampling_a.copy() for _ in range(sampling_a.shape[-1])])
        for i in range(len(replication_c)):
            replication_c[i, :, i] = sampling_b[:, i]

        replication_c = replication_c.reshape((-1, sampling_a.shape[-1]))

        return replication_c

    @abstractmethod
    def __call__(self, dimension, nb_design):
        raise NotImplementedError()


class ScipyReplicatedSampler(ReplicatedSampler):
    """
    Base class based on Scipy qmc module for replicated design sampling.
    """

    def __init__(self):
        try:
            self.qmc = scipy.stats.qmc # pylint: disable=E1101
        except AttributeError as err:
            raise ModuleNotFoundError("Xplique need scipy>=1.7 to use this sampling.") from err


class TFSobolSequenceRS(ReplicatedSampler):
    """
    Tensorflow Sobol LP tau sequence sampler.
    Ref. I. M. Sobol., The distribution of points in a cube and the accurate evaluation of
    integrals (1967).
    https://www.sciencedirect.com/science/article/abs/pii/0041555367901449
    """

    def __call__(self, dimension, nb_design):
        sampling_ab = tf.math.sobol_sample(dimension*2, nb_design, dtype=tf.float32).numpy()
        sampling_a, sampling_b = sampling_ab[:, :dimension], sampling_ab[:, dimension:]
        replicated_c = self.build_replicated_design(sampling_a, sampling_b)

        return np.concatenate([sampling_a, sampling_b, replicated_c], 0)


class ScipySobolSequenceRS(ScipyReplicatedSampler):
    """
    Scipy Sobol LP tau sequence sampler.

    Ref. I. M. Sobol., The distribution of points in a cube and the accurate evaluation of
    integrals (1967).
    https://www.sciencedirect.com/science/article/abs/pii/0041555367901449
    """

    def __call__(self, dimension, nb_design):
        sampler = self.qmc.Sobol(dimension*2, scramble=False)
        sampling_ab = sampler.random(nb_design).astype(np.float32)
        sampling_a, sampling_b = sampling_ab[:, :dimension], sampling_ab[:, dimension:]
        replicated_c = self.build_replicated_design(sampling_a, sampling_b)

        return np.concatenate([sampling_a, sampling_b, replicated_c], 0)


class HaltonSequenceRS(ScipyReplicatedSampler):
    """
    Halton sequence sampler.

    Ref. J.H. Halton., On the efficiency of certain quasi-random sequences of points in evaluating
    multi-dimensional integrals (1960).
    https://link.springer.com/article/10.1007/BF01386213
    """

    def __call__(self, dimension, nb_design):
        sampler = self.qmc.Halton(dimension*2, scramble=False)
        sampling_ab = sampler.random(nb_design).astype(np.float32)
        sampling_a, sampling_b = sampling_ab[:, :dimension], sampling_ab[:, dimension:]
        replicated_c = self.build_replicated_design(sampling_a, sampling_b)

        return np.concatenate([sampling_a, sampling_b, replicated_c], 0)


class LatinHypercubeRS(ScipyReplicatedSampler):
    """
    Latin hypercube replicated sampler.

    Ref. Mckay & al., A Comparison of Three Methods for Selecting Values of Input Variables in the
    Analysis of Output from a Computer Code (1979).
    https://www.jstor.org/stable/1268522
    """

    def __call__(self, dimension, nb_design):
        sampler = self.qmc.LatinHypercube(dimension*2)
        sampling_ab = sampler.random(nb_design).astype(np.float32)
        sampling_a, sampling_b = sampling_ab[:, :dimension], sampling_ab[:, dimension:]
        replicated_c = self.build_replicated_design(sampling_a, sampling_b)

        return np.concatenate([sampling_a, sampling_b, replicated_c], 0)
