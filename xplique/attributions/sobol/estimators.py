"""
Sobol' total order estimators module
"""

from abc import ABC, abstractmethod

import numpy as np


class SobolEstimator(ABC):
    """
    Base class for Sobol' total order estimators.
    """

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
    def split_abc(outputs, nb_design, nb_dim):
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
        sampling_a = outputs[:nb_design]
        sampling_b = outputs[nb_design:nb_design*2]
        replication_c = np.array([outputs[nb_design*2 + nb_design*i:nb_design*2 + nb_design*(i+1)]
                      for i in range(nb_dim)])
        return sampling_a, sampling_b, replication_c

    @staticmethod
    def post_process(stis, masks):
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

    def __call__(self, masks, outputs, nb_design):
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
        sampling_a, _, replication_c = self.split_abc(outputs, nb_design, nb_dim)

        mu_a = np.mean(sampling_a)
        var = np.sum([(v - mu_a)**2 for v in sampling_a]) / (len(sampling_a) - 1)

        stis = [
            np.sum((sampling_a - replication_c[i])**2.0) / (2 * nb_design * var)
            for i in range(nb_dim)
        ]

        return self.post_process(stis, masks)


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
        sampling_a, _, replication_c = self.split_abc(outputs, nb_design, nb_dim)

        mu_a = np.mean(sampling_a)
        var = np.sum([(v - mu_a)**2 for v in sampling_a]) / (len(sampling_a) - 1)

        stis = [
            (var - (1. / nb_design) * np.sum(sampling_a * replication_c[i]) + mu_a**2.0) / var
            for i in range(nb_dim)
        ]

        return self.post_process(stis, masks)


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
        sampling_a, _, replication_c = self.split_abc(outputs, nb_design, nb_dim)

        mu_ac = [(1. / nb_design) * np.sum(sampling_a + replication_c[i]) / 2.
                 for i in range(nb_dim)]
        var   = [(1. / (nb_design - 1.)) * np.sum(sampling_a**2. + replication_c[i]**2.) /
                  2. - mu_ac[i]**2. for i in range(nb_dim)]

        stis = [
            1. - ((1. / nb_design) * np.sum(sampling_a * replication_c[i]) - mu_ac[i]**2.) / var[i]
            for i in range(nb_dim)
        ]

        return self.post_process(stis, masks)


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
        sampling_a, _, replication_c = self.split_abc(outputs, nb_design, nb_dim)

        mu_a = np.mean(sampling_a)
        mu_c = [np.mean(ci) for ci in replication_c]

        var_a = np.var(sampling_a)
        var_c = [np.var(ci) for ci in replication_c]

        stis = [
            1.0 - (1. / (nb_design - 1.) * np.sum((sampling_a - mu_a) *
                  (replication_c[i] - mu_c[i])) / (var_a * var_c[i])**0.5)
            for i in range(nb_dim)
        ]

        return self.post_process(stis, masks)


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
        sampling_a, _, replication_c = self.split_abc(outputs, nb_design, nb_dim)

        mu_a = np.mean(sampling_a)
        var = np.sum([(v - mu_a)**2 for v in sampling_a]) / (len(sampling_a) - 1)

        stis = [
            1. - ((1. / nb_design) * np.sum(sampling_a * replication_c[i]) - mu_a**2.) / var
            for i in range(nb_dim)
        ]

        return self.post_process(stis, masks)
