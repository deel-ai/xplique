"""
Sobol' total order estimators module
"""

from abc import ABC, abstractmethod

import numpy as np


class SobolEstimator(ABC):
    """
    Base class for Sobol' total order estimators.
    """

    def _masks_dim(self, masks):
        """
        Deduce the number of dimensions using the sampling masks.

        Parameters
        ----------
        masks: ndarray
          Low resolution masks (before upsampling) used, one for each output.

        Returns
        -------
        nb_dim: int
          The number of dimensions under study according to the masks.
        """
        nb_dim = np.prod(masks.shape[1:])
        return nb_dim

    def _split_abc(self, outputs, nb_design, nb_dim):
        """
        Split the outputs values into the 3 sampling matrices A, B and C.

        Parameters
        ----------
        outputs: ndarray
          Model outputs for each sample point of matrices A, B and C (in order).
        nb_design: int
          Number of points for matrices A (the same as B).
        nb_dim: int
          Number of dimensions to estimate.

        Returns
        -------
        a: ndarray
          The results for the sample points in matrix A.
        b: ndarray
          The results for the sample points in matrix A.
        c: ndarray
          The results for the sample points in matrix C.
        """
        a = outputs[:nb_design]
        b = outputs[nb_design:nb_design*2]
        c = np.array([outputs[nb_design*2 + nb_design*i:nb_design*2 + nb_design*(i+1)]
                      for i in range(nb_dim)])
        return a, b, c

    def _post_process(self, stis, masks):
        """
        Post processing ops on the indices before sending them back. Makes sure the data
        format and shape is correct.

        Parameters
        ----------
        stis: ndarray
          Total order Sobol' indices, one for each dimensions.
        masks: ndarray
            Low resolution masks (before upsampling) used, one for each output.

        Returns
        -------
        stis: ndarray
          Total order Sobol' indices after post processing.
        """
        stis = np.array(stis, np.float32)
        return stis.reshape(masks.shape[1:])

    @abstractmethod
    def __call__(self, masks, outputs, nb_design):
        """
        Compute the Sobol' total order indices according to the Jansen algorithm.

        Ref. Jansen, M., Analysis of variance designs for model output (1999)
        https://www.sciencedirect.com/science/article/abs/pii/S0010465598001544

        Parameters
        ----------
        masks: ndarray
          Low resolution masks (before upsampling) used, one for each output.
        outputs: ndarray
          Model outputs associated to each masks. One for each sample point of
          matrices A, B and C (in order).
        nb_design: int
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

    def __call__(self, masks, outputs, nb_design):
        """
        Compute the Sobol' total order indices according to the Jansen algorithm.

        Parameters
        ----------
        masks: ndarray
          Low resolution masks (before upsampling) used, one for each output.
        outputs: ndarray
          Model outputs associated to each masks. One for each sample point of
          matrices A, B and C (in order).
        nb_design: int
          Number of points for matrices A (the same as B).

        Returns
        -------
        sti: ndarray
          Total order Sobol' indices, one for each dimensions.
        """
        nb_dim = self._masks_dim(masks)
        a, b, c = self._split_abc(outputs, nb_design, nb_dim)

        f0 = np.mean(a)
        var = np.sum([(v - f0)**2 for v in a]) / (len(a) - 1)

        stis = [
            np.sum((a - c[i])**2.0) / (2 * nb_design * var)
            for i in range(nb_dim)
        ]

        return self._post_process(stis, masks)


class HommaEstimator(SobolEstimator):
    """
    Homma estimator for total order Sobol' indices.

    Ref. Homma & al., Importance measures in global sensitivity analysis of nonlinear models (1996)
    https://www.sciencedirect.com/science/article/abs/pii/0951832096000026
    """

    def __call__(self, masks, outputs, nb_design):
        """
        Compute the Sobol' total order indices according to the Homma-Saltelli algorithm.

        Parameters
        ----------
        masks: ndarray
          Low resolution masks (before upsampling) used, one for each output.
        outputs: ndarray
          Model outputs associated to each masks. One for each sample point of
          matrices A, B and C (in order).
        nb_design: int
          Number of points for matrices A (the same as B).

        Returns
        -------
        sti: ndarray
          Total order Sobol' indices, one for each dimensions.
        """
        nb_dim = self._masks_dim(masks)
        a, b, c = self._split_abc(outputs, nb_design, nb_dim)

        f0 = np.mean(a)
        var = np.sum([(v - f0)**2 for v in a]) / (len(a) - 1)

        stis = [
            (var - (1. / nb_design) * np.sum(a * c[i]) + f0**2.0) / var
            for i in range(nb_dim)
        ]

        return self._post_process(stis, masks)


class JanonEstimator(SobolEstimator):
    """
    Janon estimator for total order Sobol' indices.

    Ref. Janon & al., Asymptotic normality and efficiency of two Sobol index estimators (2014)
    https://hal.inria.fr/hal-00665048v2/document
    """

    def __call__(self, masks, outputs, nb_design):
        """
        Compute the Sobol' total order indices according to the Janon algorithm.

        Parameters
        ----------
        masks: ndarray
          Low resolution masks (before upsampling) used, one for each output.
        outputs: ndarray
          Model outputs associated to each masks. One for each sample point of
          matrices A, B and C (in order).
        nb_design: int
          Number of points for matrices A (the same as B).

        Returns
        -------
        sti: ndarray
          Total order Sobol' indices, one for each dimensions.
        """
        nb_dim = self._masks_dim(masks)
        a, b, c = self._split_abc(outputs, nb_design, nb_dim)

        f0 = [(1. / nb_design) * np.sum(a + c[i]) / 2. for i in range(nb_dim)]
        var = [(1. / (nb_design - 1.)) * np.sum(a**2. + c[i]**2.) /
               2. - f0[i]**2. for i in range(nb_dim)]

        stis = [
            1. - ((1. / nb_design) * np.sum(a * c[i]) - f0[i]**2.) / var[i]
            for i in range(nb_dim)
        ]

        return self._post_process(stis, masks)


class GlenEstimator(SobolEstimator):
    """
    Glen-Isaacs estimator for total order Sobol' indices.

    Ref. Glen & al., Estimating Sobol sensitivity indices using correlations (2012)
    https://dl.acm.org/doi/abs/10.1016/j.envsoft.2012.03.014
    """

    def __call__(self, masks, outputs, nb_design):
        """
        Compute the Sobol' total order indices according to the Glen-Isaacs algorithm.

        Parameters
        ----------
        masks: ndarray
          Low resolution masks (before upsampling) used, one for each output.
        outputs: ndarray
          Model outputs associated to each masks. One for each sample point of
          matrices A, B and C (in order).
        nb_design: int
          Number of points for matrices A (the same as B).

        Returns
        -------
        sti: ndarray
          Total order Sobol' indices, one for each dimensions.
        """
        nb_dim = self._masks_dim(masks)
        a, b, c = self._split_abc(outputs, nb_design, nb_dim)

        mean_a = np.mean(a)
        mean_c = [np.mean(ci) for ci in c]
        var_a = np.var(a)
        var_c = [np.var(ci) for ci in c]

        stis = [
            1.0 - (1. / (nb_design - 1.) * np.sum((a - mean_a) *
                                                  (c[i] - mean_c[i])) / (var_a * var_c[i])**0.5)
            for i in range(nb_dim)
        ]

        return self._post_process(stis, masks)


class SaltelliEstimator(SobolEstimator):
    """
    Saltelli estimator for total order Sobol' indices.

    Ref. Satelli & al., Global Sensitivity Analysis. The Primer.
    https://onlinelibrary.wiley.com/doi/book/10.1002/9780470725184
    """

    def __call__(self, masks, outputs, nb_design):
        """
        Compute the Sobol' total order indices according to the Saltelli algorithm.

        Parameters
        ----------
        masks: ndarray
          Low resolution masks (before upsampling) used, one for each output.
        outputs: ndarray
          Model outputs associated to each masks. One for each sample point of
          matrices A, B and C (in order).
        nb_design: int
          Number of points for matrices A (the same as B).

        Returns
        -------
        sti: ndarray
          Total order Sobol' indices, one for each dimensions.
        """
        nb_dim = self._masks_dim(masks)
        a, b, c = self._split_abc(outputs, nb_design, nb_dim)

        f0 = np.mean(a)
        var = np.sum([(v - f0)**2 for v in a]) / (len(a) - 1)

        stis = [
            1. - ((1. / nb_design) * np.sum(a * c[i]) - f0**2.) / var
            for i in range(nb_dim)
        ]

        return self._post_process(stis, masks)
