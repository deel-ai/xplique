"""
Kernel functions for HSIC estimators.
"""
# pylint: disable=C0103

from enum import Enum
import tensorflow as tf


@tf.function
def rbf(X, Y, width):
    """
    Radial Basis Function kernel.

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
    XY = (X - Y) ** 2
    K = tf.exp(-XY / (2 * width**2))
    return K


@tf.function
def binary(X, Y):
    """
    Dirac kernel

    Parameters
    ----------
    X
        Samples of input variable (must be binary)
    Y
        Samples of output variable

    Returns
    -------
    Kernel matrix
    """
    K = 0.5 - (X - Y) ** 2
    return K


@tf.function
def sobolev(X, Y):
    """
    Sobolev kernel of degree 2.

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
    XX = tf.math.abs(X - Y)
    B2XX = XX**2 - XX + 1 / 6

    Xr1 = tf.math.abs(X) - 0.5
    Xr2 = tf.math.abs(Y) - 0.5
    B1XX = tf.einsum("ijkl, ijlm-> ijkm", Xr1, Xr2)

    K = B2XX / 2 + B1XX
    return K


class Kernel(Enum):
    """
    GSA Perturbation function interface.
    """

    RBF = rbf
    BINARY = binary
    SOBOLEV = sobolev

    @staticmethod
    def from_string(kernel_function: str) -> "Kernel":
        """
        Restore a perturbation function from a string.

        Parameters
        ----------
        perturbation_function
            String indicating the perturbation function to restore: must be one
            of 'inpainting', 'blurring' or 'amplitude'.

        Returns
        -------
        perturbation_function
            The PerturbationFunction object.
        """
        assert kernel_function in [
            "rbf",
            "binary",
            "sobolev",
        ], "Only 'rbf', 'binary' and 'sobolev' are supported."

        if kernel_function == "rbf":
            return Kernel.RBF
        if kernel_function == "binary":
            return Kernel.BINARY
        return Kernel.SOBOLEV
