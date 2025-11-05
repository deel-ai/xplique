"""
Sampling methods for replicated designs
"""

from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from einops import rearrange, repeat
import scipy


class ReplicatedSampler(ABC):
    """
    Base class for replicated design sampling.
    """

    @staticmethod
    @tf.function
    def build_replicated_design(sampling_a: tf.Tensor, sampling_b: tf.Tensor) -> tf.Tensor:
        """
        Build the replicated design matrix C using A & B via TF and einops.

        Parameters
        ----------
        sampling_a : tf.Tensor
            The masks values for the sampling matrix A, shape (nb_design, d).
        sampling_b : tf.Tensor
            The masks values for the sampling matrix B, shape (nb_design, d).

        Returns
        -------
        replication_c : tf.Tensor
            The replicated design matrix C of shape (nb_design * d, d).
        """
        # Get the number of dimensions (d). (Works in eager or graph mode.)
        d = tf.shape(sampling_a)[-1]
        # Create d copies of sampling_a with shape (d, nb_design, d)
        replication_c = repeat(sampling_a, 'b d -> i b d', i=d)
        # Similarly, replicate sampling_b to the same shape
        replication_b = repeat(sampling_b, 'b d -> i b d', i=d)
        # Create a diagonal mask: shape (d, 1, d) then broadcast to (d, nb_design, d)
        diag_mask = tf.eye(tf.cast(d, tf.int32), dtype=sampling_a.dtype)  # (d, d)
        diag_mask = tf.expand_dims(diag_mask, axis=1)  # (d, 1, d)
        diag_mask = tf.broadcast_to(diag_mask, tf.shape(replication_c))
        # For each "replication" i, replace the i-th column with sampling_b
        replication_c = replication_c * (1 - diag_mask) + replication_b * diag_mask
        # Flatten the first two dimensions so that replication_c has shape (nb_design * d, d)
        replication_c = rearrange(replication_c, 'i b d -> (i b) d')

        return replication_c

    @abstractmethod
    def __call__(self, dimension: int, nb_design: int) -> tf.Tensor:
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
    @tf.function
    def __call__(self, dimension: int, nb_design: int) -> tf.Tensor:
        # Generate 2*dimension numbers per design point.
        sampling_ab = tf.math.sobol_sample(dimension * 2, nb_design, dtype=tf.float32)
        sampling_a = sampling_ab[:, :dimension]  # (nb_design, dimension)
        sampling_b = sampling_ab[:, dimension:]  # (nb_design, dimension)
        replicated_c = ReplicatedSampler.build_replicated_design(sampling_a, sampling_b)
        return tf.concat([sampling_a, sampling_b, replicated_c], axis=0)



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
