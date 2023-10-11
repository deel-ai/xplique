"""
Possible online reducers
"""
from abc import ABC, abstractmethod

import tensorflow as tf


class OnlineStatistic(ABC):
    """
    Abstract class to compute statistics online. This is useful to save memory costs.
    """
    @abstractmethod
    def update(self, elements):
        """
        Update the running statistic by taking new elements into account.
        It reduces elements to a statistic and only saves the statistic.

        Parameters
        ----------
        elements
            Batch of batch of elements.
            Part of all the elements the statistic should be computed on.
            Shape: (inputs_batch_size, perturbation_batch_size, ...)
        """
        raise NotImplementedError

    @abstractmethod
    def get_statistic(self):
        """
        Return the final value of the statistic.

        Returns
        -------
        statistic
            The statistic computed online.
        """
        raise NotImplementedError


class OnlineMean(OnlineStatistic):
    """
    Update the running statistic by taking new elements into account.
    It reduces elements to a mean and only saves the mean.

    Attributes
    ----------
    elements
        Batch of batch of elements.
        Part of all the elements the statistic should be computed on.
        Shape: (inputs_batch_size, perturbation_batch_size, ...)
    """
    def __init__(self):
        self.elements_counter = 0
        self.actual_sum = 0

    def update(self, elements):
        """
        Update the running mean by taking new elements into account.

        Parameters
        ----------
        elements
            Batch of batch of elements.
            Part of all the elements the mean should be computed on.
            Shape: (inputs_batch_size, perturbation_batch_size, ...)
        """
        new_elements_sum = tf.reduce_sum(elements, axis=1)
        new_elements_count = elements.shape[1]

        # actualize mean
        self.actual_sum += new_elements_sum

        # actualize count
        self.elements_counter += new_elements_count

    def get_statistic(self):
        """
        Return the final value of the mean.

        Returns
        -------
        mean
            The mean computed online.
        """
        return self.actual_sum / self.elements_counter


class OnlineSquareMean(OnlineStatistic):
    """
    Update the running statistic by taking new elements into account.
    It reduces elements to a square mean and only saves the square mean.

    Attributes
    ----------
    elements
        Batch of batch of elements.
        Part of all the elements the statistic should be computed on.
        Shape: (inputs_batch_size, perturbation_batch_size, ...)
    """
    def __init__(self):
        self.elements_counter = 0
        self.actual_square_sum = 0

    def update(self, elements):
        """
        Update the running square mean by taking new elements into account.

        Parameters
        ----------
        elements
            Batch of batch of elements.
            Part of all the elements the square mean should be computed on.
            Shape: (inputs_batch_size, perturbation_batch_size, ...)
        """
        new_elements_square_sum = tf.reduce_sum(elements**2, axis=1)
        new_elements_count = elements.shape[1]

        # actualize mean
        self.actual_square_sum += new_elements_square_sum

        # actualize count
        self.elements_counter += new_elements_count

    def get_statistic(self):
        """
        Return the final value of the square mean.

        Returns
        -------
        square_mean
            The square mean computed online.
        """
        return self.actual_square_sum / self.elements_counter


class OnlineVariance(OnlineStatistic):
    """
    Update the running statistic by taking new elements into account.
    It reduces elements to a variance and only saves the variance.

    Attributes
    ----------
    elements
        Batch of batch of elements.
        Part of all the elements the statistic should be computed on.
        Shape: (inputs_batch_size, perturbation_batch_size, ...)
    """
    def __init__(self):
        self.online_mean = OnlineMean()
        self.online_square_mean = OnlineSquareMean()

    def update(self, elements):
        """
        Update the running variance by taking new elements into account.

        Parameters
        ----------
        elements
            Batch of batch of elements.
            Part of all the elements the variance should be computed on.
            Shape: (inputs_batch_size, perturbation_batch_size, ...)
        """
        self.online_mean.update(elements)
        self.online_square_mean.update(elements)

    def get_statistic(self):
        """
        Return the final value of the variance.

        Returns
        -------
        variance
            The variance computed online.
        """
        # compute coefficient for an unbiased variance
        elements_counter = self.online_mean.elements_counter
        assert elements_counter >= 2, "Variance cannot be computed with only one element." +\
            "In the case of `VarGrad`, increase `nb_samples`."
        unbiased_coefficient = elements_counter / (elements_counter - 1)

        # compute the online mean and square mean
        mean = self.online_mean.get_statistic()
        square_mean = self.online_square_mean.get_statistic()

        # compute variance
        return unbiased_coefficient * (square_mean - mean**2)
