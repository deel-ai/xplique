"""
Sobol' total order estimators module
"""

from abc import ABC, abstractmethod

import tensorflow as tf
from einops import rearrange

from ...types import Tuple, Union


EPS = 1e-12


def _sample_var_1d(x: tf.Tensor) -> tf.Tensor:
    """
    Unbiased sample variance for a 1D tensor (ddof=1).
    Returns scalar (same dtype as x).
    """
    x = tf.cast(x, dtype=tf.float32)
    n = tf.cast(tf.size(x), x.dtype)
    mean = tf.reduce_mean(x)
    # Sum of squared deviations
    ssd = tf.reduce_sum(tf.square(x - mean))
    denom = tf.maximum(n - 1.0, 1.0)  # guard for n=1
    return ssd / denom


def _sample_var_along_last(x: tf.Tensor) -> tf.Tensor:
    """
    Unbiased sample variance along the last axis (ddof=1).
    For input (..., N) -> output (...,)
    """
    x = tf.cast(x, dtype=tf.float32)
    n = tf.cast(tf.shape(x)[-1], x.dtype)
    mean = tf.reduce_mean(x, axis=-1, keepdims=True)
    ssd = tf.reduce_sum(tf.square(x - mean), axis=-1)
    denom = tf.maximum(n - 1.0, 1.0)
    return ssd / denom


class SobolEstimator(ABC):
    """
    Base class for Sobol' total order estimators.
    """

    @staticmethod
    @tf.function(jit_compile=True)
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
        shape1p = tf.shape(masks)[1:]  # (H, W[, C])
        nb_dim = tf.reduce_prod(shape1p)  # H*W*(C?)
        return nb_dim

    @staticmethod
    @tf.function(jit_compile=True)
    def split_abc(outputs: tf.Tensor,
                  nb_design: Union[tf.Tensor, int],
                  nb_dim: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Split the outputs values into the 3 sampling matrices A, B and C.

        Parameters
        ----------
        outputs
          Model outputs for each sample point of matrices A, B and C (in order).
        nb_design
          Number of points for matrices A (the same as B).
        nb_dim
          Number of dimensions to estimate.

        Returns
        -------
        a
          The results for the sample points in matrix A.
        b
          The results for the sample points in matrix A.
        c
          The results for the sample points in matrix C.
        """
        outputs = tf.convert_to_tensor(outputs)
        nb_design = tf.cast(nb_design, tf.int32)
        nb_dim = tf.cast(nb_dim, tf.int32)

        n_total_expected = nb_design * (2 + nb_dim)
        n_total = tf.shape(outputs)[0]

        # Checks done in-graph (no Python branching):
        tf.debugging.assert_equal(
            n_total, n_total_expected,
            message="outputs length must be nb_design * (2 + nb_dim)"
        )

        a = outputs[:nb_design]  # (N,)
        b = outputs[nb_design:nb_design * 2]  # (N,)
        c_flat = outputs[nb_design * 2:]  # (D*N,)

        # Reshape C to (D, N)
        c = rearrange(c_flat, '(d n) -> d n', d=nb_dim)

        return a, b, c

    @staticmethod
    @tf.function(jit_compile=True)
    def post_process(stis: tf.Tensor, masks: tf.Tensor) -> tf.Tensor:
        """
        Post processing ops on the indices before sending them back. Makes sure the data
        format and shape is correct.

        Parameters
        ----------
        stis
          Total order Sobol' indices, one for each dimensions.
        masks
            Low resolution masks (before upsampling) used, one for each output.

        Returns
        -------
        stis
          Total order Sobol' indices after post processing.
        """
        stis = tf.convert_to_tensor(stis)
        target_shape = tf.shape(masks)[1:]  # (H, W[, C])
        return tf.reshape(stis, target_shape)

    @abstractmethod
    def __call__(self, masks, outputs, nb_design):
        """
        Compute the Sobol' total order indices according to the Jansen algorithm.

        Ref. Jansen, M., Analysis of variance designs for model output (1999)
        https://www.sciencedirect.com/science/article/abs/pii/S0010465598001544

        Parameters
        ----------
        masks
          Low resolution masks (before upsampling) used, one for each output.
        outputs
          Model outputs associated to each masks. One for each sample point of
          matrices A, B and C (in order).
        nb_design
          Number of points for matrices A (the same as B).

        Returns
        -------
        sti: ndarray
          Total order Sobol' indices, one for each dimensions.
        """
        raise NotImplementedError()


class JansenEstimator(SobolEstimator):
    """
    Jansen estimator for total order Sobol' indices.

    Ref. Jansen, M., Analysis of variance designs for model output (1999)
    https://www.sciencedirect.com/science/article/abs/pii/S0010465598001544
    """

    @tf.function(jit_compile=True)
    def __call__(self, masks: tf.Tensor, outputs: tf.Tensor, nb_design: int) -> tf.Tensor:
        """
        Compute the Sobol' total order indices according to the Jansen algorithm.

        Parameters
        ----------
        masks
          Low resolution masks (before upsampling) used, one for each output.
        outputs
          Model outputs associated to each masks. One for each sample point of
          matrices A, B and C (in order).
        nb_design
          Number of points for matrices A (the same as B).

        Returns
        -------
        sti
          Total order Sobol' indices, one for each dimensions.
        """
        nb_dim = self.masks_dim(masks)
        a, _, c = self.split_abc(outputs, nb_design, nb_dim)  # a:(N,), c:(D,N)

        a = tf.cast(a, dtype=tf.float32)
        c = tf.cast(c, dtype=tf.float32)

        var_a = _sample_var_1d(a)
        var_a = tf.maximum(var_a, tf.constant(EPS, dtype=a.dtype))

        # (D, N) broadcast: a -> (1, N)
        diff = a[None, :] - c
        numerator = tf.reduce_sum(tf.square(diff), axis=-1)  # (D,)

        n = tf.cast(tf.shape(a)[0], a.dtype)
        st = numerator / (2.0 * n * var_a)  # (D,)

        return self.post_process(st, masks)


class HommaEstimator(SobolEstimator):
    """
    Homma estimator for total order Sobol' indices.

    Ref. Homma & al., Importance measures in global sensitivity analysis of nonlinear models (1996)
    https://www.sciencedirect.com/science/article/abs/pii/0951832096000026
    """

    @tf.function(jit_compile=True)
    def __call__(self, masks: tf.Tensor, outputs: tf.Tensor, nb_design: int) -> tf.Tensor:
        """
        Compute the Sobol' total order indices according to the Homma-Saltelli algorithm.

        Parameters
        ----------
        masks
          Low resolution masks (before upsampling) used, one for each output.
        outputs
          Model outputs associated to each masks. One for each sample point of
          matrices A, B and C (in order).
        nb_design
          Number of points for matrices A (the same as B).

        Returns
        -------
        sti
          Total order Sobol' indices, one for each dimensions.
        """
        nb_dim = self.masks_dim(masks)
        a, _, c = self.split_abc(outputs, nb_design, nb_dim)  # a:(N,), c:(D,N)

        a = tf.cast(a, dtype=tf.float32)
        c = tf.cast(c, dtype=tf.float32)

        mu_a = tf.reduce_mean(a)
        var_a = _sample_var_1d(a)
        var_a = tf.maximum(var_a, tf.constant(EPS, dtype=a.dtype))

        # E[A*C_i] = mean over N of elementwise product
        e_ac = tf.reduce_mean(c * a[None, :], axis=-1)  # (D,)

        st = (var_a - e_ac + tf.square(mu_a)) / var_a  # (D,)
        return self.post_process(st, masks)


class JanonEstimator(SobolEstimator):
    """
    Janon estimator for total order Sobol' indices.

    Ref. Janon & al., Asymptotic normality and efficiency of two Sobol index estimators (2014)
    https://hal.inria.fr/hal-00665048v2/document
    """

    @tf.function(jit_compile=True)
    def __call__(self, masks: tf.Tensor, outputs: tf.Tensor, nb_design: int) -> tf.Tensor:
        """
        Compute the Sobol' total order indices according to the Janon algorithm.

        Parameters
        ----------
        masks
          Low resolution masks (before upsampling) used, one for each output.
        outputs
          Model outputs associated to each masks. One for each sample point of
          matrices A, B and C (in order).
        nb_design
          Number of points for matrices A (the same as B).

        Returns
        -------
        sti
          Total order Sobol' indices, one for each dimensions.
        """
        nb_dim = self.masks_dim(masks)
        a, _, c = self.split_abc(outputs, nb_design, nb_dim)  # a:(N,), c:(D,N)

        a = tf.cast(a, dtype=tf.float32)
        c = tf.cast(c, dtype=tf.float32)

        n = tf.cast(tf.shape(a)[0], a.dtype)
        denom_cov = tf.maximum(n - 1.0, 1.0)

        mu_a = tf.reduce_mean(a)  # scalar
        mu_c = tf.reduce_mean(c, axis=-1)  # (D,)
        mu_ac = 0.5 * (mu_a + mu_c)  # (D,)

        sum_a2 = tf.reduce_sum(tf.square(a))  # scalar
        sum_c2 = tf.reduce_sum(tf.square(c), axis=-1)  # (D,)

        var = (sum_a2 + sum_c2) / (2.0 * denom_cov) - tf.square(mu_ac)  # (D,)
        var = tf.maximum(var, tf.constant(EPS, dtype=a.dtype))

        e_ac = tf.reduce_mean(c * a[None, :], axis=-1)  # (D,)
        st = 1.0 - (e_ac - tf.square(mu_ac)) / var  # (D,)

        return self.post_process(st, masks)


class GlenEstimator(SobolEstimator):
    """
    Glen-Isaacs estimator for total order Sobol' indices.

    Ref. Glen & al., Estimating Sobol sensitivity indices using correlations (2012)
    https://dl.acm.org/doi/abs/10.1016/j.envsoft.2012.03.014
    """

    @tf.function(jit_compile=True)
    def __call__(self, masks: tf.Tensor, outputs: tf.Tensor, nb_design: int) -> tf.Tensor:
        """
        Compute the Sobol' total order indices according to the Glen-Isaacs algorithm.

        Parameters
        ----------
        masks
          Low resolution masks (before upsampling) used, one for each output.
        outputs
          Model outputs associated to each masks. One for each sample point of
          matrices A, B and C (in order).
        nb_design
          Number of points for matrices A (the same as B).

        Returns
        -------
        sti
          Total order Sobol' indices, one for each dimensions.
        """
        nb_dim = self.masks_dim(masks)
        a, _, c = self.split_abc(outputs, nb_design, nb_dim)  # a:(N,), c:(D,N)

        a = tf.cast(a, dtype=tf.float32)
        c = tf.cast(c, dtype=tf.float32)

        n = tf.cast(tf.shape(a)[0], a.dtype)
        denom = tf.maximum(n - 1.0, 1.0)

        mu_a = tf.reduce_mean(a)  # scalar
        mu_c = tf.reduce_mean(c, axis=-1)  # (D,)

        a_c = a - mu_a  # (N,)
        c_c = c - mu_c[:, None]  # (D, N)

        cov = tf.reduce_sum(a_c[None, :] * c_c, axis=-1) / denom  # (D,)

        var_a = _sample_var_1d(a)  # scalar
        var_c = _sample_var_along_last(c)  # (D,)

        var_a = tf.maximum(var_a, tf.constant(EPS, dtype=a.dtype))
        var_c = tf.maximum(var_c, tf.constant(EPS, dtype=a.dtype))

        corr = cov / tf.sqrt(var_a * var_c)  # (D,)
        st = 1.0 - corr  # (D,)

        return self.post_process(st, masks)


class SaltelliEstimator(SobolEstimator):
    """
    Saltelli estimator for total order Sobol' indices.

    Ref. Satelli & al., Global Sensitivity Analysis. The Primer.
    https://onlinelibrary.wiley.com/doi/book/10.1002/9780470725184
    """

    @tf.function(jit_compile=True)
    def __call__(self, masks: tf.Tensor, outputs: tf.Tensor, nb_design: int) -> tf.Tensor:
        """
        Compute the Sobol' total order indices according to the Saltelli algorithm.

        Parameters
        ----------
        masks
          Low resolution masks (before upsampling) used, one for each output.
        outputs
          Model outputs associated to each masks. One for each sample point of
          matrices A, B and C (in order).
        nb_design
          Number of points for matrices A (the same as B).

        Returns
        -------
        sti
          Total order Sobol' indices, one for each dimensions.
        """
        nb_dim = self.masks_dim(masks)
        a, _, c = self.split_abc(outputs, nb_design, nb_dim)  # a:(N,), c:(D,N)

        a = tf.cast(a, dtype=tf.float32)
        c = tf.cast(c, dtype=tf.float32)

        mu_a = tf.reduce_mean(a)
        var_a = _sample_var_1d(a)
        var_a = tf.maximum(var_a, tf.constant(EPS, dtype=a.dtype))

        e_ac = tf.reduce_mean(c * a[None, :], axis=-1)  # (D,)
        st = 1.0 - (e_ac - tf.square(mu_a)) / var_a  # (D,)

        return self.post_process(st, masks)
