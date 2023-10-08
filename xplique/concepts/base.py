"""
Module related to abstract concept explainer
"""

from abc import ABC, abstractmethod
import numpy as np

class BaseConceptExtractor(ABC):

    """
    Base class for concept extraction models.

    Parameters
    ----------
    number_of_concepts
        The number of concepts to extract. Default is 20.
    batch_size
        The batch size to use during training and prediction. Default is 64.

    """
    @abstractmethod
    def __init__(self, number_of_concepts: int = 20,
                       batch_size: int = 64):
        self.number_of_concepts = number_of_concepts
        self.batch_size = batch_size

        # sanity checks
        assert(number_of_concepts > 0), "number_of_concepts must be greater than 0"
        assert(batch_size > 0), "batch_size must be greater than 0"

    @abstractmethod
    def fit(self, inputs):
        """
        Fit the CAVs to the input data.

        Parameters
        ----------
        inputs
            The input data to fit the model on.

        Returns
        -------
        tuple
            A tuple containing the input data and the matrices (U, W) that factorize the data.

        """
        raise NotImplementedError

    @abstractmethod
    def check_if_fitted(self):
        """Checks if the factorization model has been fitted to input data.

        Raises
        ------
        NotFittedError
            If the factorization model has not been fitted to input data.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, inputs) -> np.ndarray:
        """
        Transform the input data into a concepts embedding.

        Parameters
        ----------
        inputs
            The input data to transform.

        Returns
        -------
        array-like
            The transformed embedding of the input data.

        """
        raise NotImplementedError
