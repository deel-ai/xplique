"""
HSIC estimator
"""
# pylint: disable=C0103

from abc import ABC, abstractmethod
from functools import partial

import tensorflow as tf
import tensorflow_probability as tfp
from einops import rearrange

from .kernels import Kernel

from ...commons import batch_tensor
from ...types import Union


class HsicEstimator(ABC):
    """
    Base class for HSIC estimator.
    """

    def __init__(self, output_kernel="rbf"):
        self.output_kernel = output_kernel
        assert output_kernel in ["rbf"], "Only 'rbf' output kernel is supported for now."
        # Set a high batch size (can be updated via set_batch_size).
        self.batch_size = 100000

    @staticmethod
    def masks_dim(masks: tf.Tensor) -> tf.Tensor:
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
        return tf.reduce_prod(tf.shape(masks)[1:])

    @staticmethod
    def post_process(score: tf.Tensor, masks: tf.Tensor) -> tf.Tensor:
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
        # Reshape to (H, W, 1) and then swap the first two axes.
        reshaped = tf.reshape(score, tf.shape(masks)[1:])
        return tf.transpose(reshaped, perm=[1, 0, 2])

    @abstractmethod
    def input_kernel_func(self, X: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:
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
    def output_kernel_func(self, X: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:
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

    # We do not decorate this with @tf.function to avoid retracing due to dynamic shapes.
    def estimator(self, masks: tf.Tensor, L: tf.Tensor,
                  nb_dim: Union[int, tf.Tensor], nb_design: Union[int, tf.Tensor]) -> tf.Tensor:
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
        # Rearrange to get (d, nb_design) where d = H*W*1.
        X = rearrange(masks, 'n h w c -> (c w h) n')
        # Add singleton dimensions: shape becomes (d, 1, nb_design, 1)
        X1 = rearrange(X, 'd n -> d 1 n 1')
        # Swap last two axes: shape becomes (d, 1, 1, nb_design)
        X2 = rearrange(X1, 'd a n b -> d a b n')

        # Use the minimum of self.batch_size and nb_dim to avoid OOM.
        batch_size = tf.cond(
            nb_dim > self.batch_size,
            lambda: tf.cast(self.batch_size, tf.int64),
            lambda: tf.cast(nb_dim, tf.int64)
        )

        scores = tf.zeros((0,), dtype=tf.float32)
        # Batch over the mask dimensions (using batch_tensor from xplique.commons).
        for x1, x2 in batch_tensor((X1, X2), batch_size):
            K = self.input_kernel_func(x1, x2)
            # Here we reduce over axis=1 (the kernel is computed per mask dimension)
            K = tf.math.reduce_prod(1 + K, axis=1)
            H = tf.eye(nb_design) - tf.ones((nb_design, nb_design), dtype=tf.float32) / tf.cast(nb_design, tf.float32)
            HK = tf.einsum("jk,ikl->ijl", H, K)
            HL = tf.einsum("jk,kl->jl", H, L)
            Kc = tf.einsum("ijk,kl->ijl", HK, H)
            Lc = tf.einsum("jk,kl->jl", HL, H)
            score = tf.math.reduce_sum(Kc * tf.transpose(Lc), axis=[1, 2]) / tf.cast(nb_design, tf.float32)
            scores = tf.concat([scores, score], axis=0)

        return scores

    def __call__(self, masks: tf.Tensor, outputs: tf.Tensor, nb_design: Union[int, tf.Tensor]) -> tf.Tensor:
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

        # Cast outputs to float and reshape to (nb_design, 1)
        Y = tf.cast(outputs, tf.float32)
        Y = tf.reshape(Y, (nb_design, 1))

        # Use tfp.stats.percentile to compute the median if needed in output_kernel_func.
        L = self.output_kernel_func(Y, tf.transpose(Y))
        score = self.estimator(masks, L, nb_dim, nb_design)
        return self.post_process(score, masks)


class BinaryEstimator(HsicEstimator):
    """
    HSIC estimator using the binary (Dirac) kernel for the input.
    """

    def input_kernel_func(self, X: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:
        # Use the binary kernel defined via Kernel.from_string.
        return Kernel.from_string("binary")(X, Y)

    def output_kernel_func(self, X: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:
        # Use tfp.stats.percentile to obtain the median.
        width_y = tfp.stats.percentile(Y, 50.0, interpolation='linear')
        width_y = tf.cast(width_y, tf.float32)
        kernel_func = partial(Kernel.from_string(self.output_kernel), width=width_y)
        return kernel_func(X, Y)


class RbfEstimator(HsicEstimator):
    """
    HSIC estimator using the RBF kernel for the input.
    """

    def input_kernel_func(self, X: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:
        width_x = 0.5
        kernel_func = partial(Kernel.from_string("rbf"), width=width_x)
        return kernel_func(X, Y)

    def output_kernel_func(self, X: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:
        width_y = tfp.stats.percentile(Y, 50.0, interpolation='linear')
        width_y = tf.cast(width_y, tf.float32)
        kernel_func = partial(Kernel.from_string(self.output_kernel), width=width_y)
        return kernel_func(X, Y)


class SobolevEstimator(HsicEstimator):
    """
    HSIC estimator using the Sobolev kernel for the input.
    """

    def input_kernel_func(self, X: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:
        return Kernel.from_string("sobolev")(X, Y)

    def output_kernel_func(self, X: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:
        width_y = tfp.stats.percentile(Y, 50.0, interpolation='linear')
        width_y = tf.cast(width_y, tf.float32)
        kernel_func = partial(Kernel.from_string(self.output_kernel), width=width_y)
        return kernel_func(X, Y)
