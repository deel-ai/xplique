"""
HSIC estimator
"""
# pylint: disable=C0103

from abc import ABC, abstractmethod
from functools import partial

import tensorflow as tf
import numpy as np

from .kernels import Kernel

from ...commons import batch_tensor


class HsicEstimator(ABC):
    """
    Base class for HSIC estimator.
    """

    def __init__(self, output_kernel="rbf"):
        self.output_kernel = output_kernel
        assert output_kernel in [
            "rbf"
        ], "Only 'rbf' output kernel is supported for now."

        # set a batch_size higher than any `grid_size`² possible
        # updated if an `estimator_batch_size` is given to `HsicAttributionMethod`.
        self.batch_size = 100000

    @staticmethod
    def masks_dim(masks):
        """
        Deduce the number of dimensions using the sampling masks.

        Parameters
        ----------
        masks
          Low resolution masks (before upsampling) used, one for each output.

        Returns
        -------
        nb_dim
          The number of dimensions under study according to the masks.
        """
        nb_dim = np.prod(masks.shape[1:])
        return nb_dim

    @staticmethod
    def post_process(score, masks):
        """
        Post processing ops on the indices before sending them back. Makes sure the data
        format and shape is correct.

        Parameters
        ----------
        score
          Total order HSIC scores, one for each dimensions.
        masks
            Low resolution masks (before upsampling) used, one for each output.

        Returns
        -------
        score
          HSIC scores after post processing.
        """
        score = np.array(score, np.float32)
        return np.transpose(score.reshape(masks.shape[1:]), axes=(1, 0, 2))

    @abstractmethod
    def input_kernel_func(self, X, Y):
        """
        Kernel function for the input.

        Parameters
        ----------
        X
            Samples of input variable
        Y
            Samples of output variable

        Returns
        -------
        Kernel matrix
        """
        raise NotImplementedError()

    @abstractmethod
    def output_kernel_func(self, X, Y):
        """
        kernel function for the output

        Parameters
        ----------
        X
            Samples of input variable
        Y
            Samples of output variable

        Returns
        -------
        Kernel matrix
        """
        raise NotImplementedError()

    def set_batch_size(self, batch_size=None):
        """
        Set the batch size to use for the estimator.

        Parameters
        ----------
        batch_size
            Batch size to use for the estimator.
        """
        if batch_size is not None:
            self.batch_size = batch_size
        else:
            pass  # already set to 100000 in the init

    @tf.function
    def estimator(self, masks, L, nb_dim, nb_design):
        """
        tf operations related to the estimator for performances

        Parameters
        ----------
        masks
            binary masks, each dimension corresponding to an image patch
        L
            output samples kernel Gram matrix
        nb_dim
            number of input variables to consider
        nb_design
            number of points used to estimate HSIC

        Returns
        -------
        HSIC estimates
            Raw HSIC estimates in tensorflow
        """
        X = tf.transpose(masks)

        X1 = tf.reshape(X, (nb_dim, 1, nb_design, 1))
        X2 = tf.transpose(X1, [0, 1, 3, 2])

        # min(self.batch_size, nb_dim) is used to avoid OOM
        batch_size = tf.cond(nb_dim > self.batch_size,
                             lambda: tf.cast(tf.constant(self.batch_size), tf.int64),
                             lambda: tf.cast(nb_dim, tf.int64))

        # initialize array of scores
        scores = tf.zeros((0,))
        # batch over the mask dimensions (may be done only once)
        for x1, x2 in batch_tensor((X1, X2), tf.cast(batch_size, tf.int64)):

            K = self.input_kernel_func(x1, x2)
            K = tf.math.reduce_prod(1 + K, axis=1)

            H = tf.eye(nb_design) - tf.ones((nb_design, nb_design)) / nb_design
            HK = tf.einsum("jk,ikl->ijl", H, K)
            HL = tf.einsum("jk,kl->jl", H, L)

            Kc = tf.einsum("ijk,kl->ijl", HK, H)
            Lc = tf.einsum("jk,kl->jl", HL, H)

            score = tf.math.reduce_sum(Kc * tf.transpose(Lc), axis=[1, 2]) / nb_design

            # add score to array of scores
            scores = tf.concat([scores, score], axis=0)

        return scores

    def __call__(self, masks, outputs, nb_design):
        """
        Compute the test statistic using a self.output_kernel_func kernel for the output
        and self.input_kernel_func for the input, to be defined in child classes
        for the input.

        Parameters
        ----------
        masks
            binary masks, each dimension corresponding to an image patch
        outputs
            samples of the output variable
        nb_design
            number of points used to estimate HSIC

        Returns
        -------
        HSIC estimates
            Array with HSIC estimates for each patch
        """
        nb_dim = self.masks_dim(masks)

        Y = tf.cast(outputs, tf.float32)
        Y = tf.reshape(Y, (nb_design, 1))
        L = self.output_kernel_func(Y, tf.transpose(Y))

        score = self.estimator(masks, L, nb_dim, nb_design)

        return self.post_process(score, masks)


class BinaryEstimator(HsicEstimator):
    """
    Estimator based on the Dirac kernel for the input
    """

    def input_kernel_func(self, X, Y):
        return Kernel.from_string("binary")(X, Y)

    def output_kernel_func(self, X, Y):
        width_y = np.percentile(Y, 50.0).astype(np.float32)
        kernel_func = partial(Kernel.from_string(self.output_kernel), width=width_y)
        return kernel_func(X, Y)


class RbfEstimator(HsicEstimator):
    """
    Estimator based on the RBF kernel for the input
    """

    def input_kernel_func(self, X, Y):
        width_x = 0.5
        kernel_func = partial(Kernel.from_string("rbf"), width=width_x)
        return kernel_func(X, Y)

    def output_kernel_func(self, X, Y):
        width_y = np.percentile(Y, 50.0).astype(np.float32)
        kernel_func = partial(Kernel.from_string(self.output_kernel), width=width_y)
        return kernel_func(X, Y)


class SobolevEstimator(HsicEstimator):
    """
    Estimator based on the Sobolev kernel for the input
    """

    def input_kernel_func(self, X, Y):
        return Kernel.from_string("sobolev")(X, Y)

    def output_kernel_func(self, X, Y):
        width_y = np.percentile(Y, 50.0).astype(np.float32)
        kernel_func = partial(Kernel.from_string(self.output_kernel), width=width_y)
        return kernel_func(X, Y)
