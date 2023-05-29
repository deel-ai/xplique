"""
Base model for example-based 
"""

from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np

from ...types import Callable, Dict, Tuple, Union, Optional, List

from ...attributions.base import sanitize_input_output


class BaseSearchMethod(ABC):
    """
    Base class for natural example-base methods, they operate on a training set.
    It should manage examples searching algorithm

    Parameters
    ----------
    search_set
        The dataset used to train the model,
        also use by the function to calcul the closest examples.
    """

    def __init__(self,
                 search_set: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 k: int = 1,
                 returns: Optional[Union[List[str], str]] = None):
        self.search_set = search_set
        # ATTENTION, si le dataset est gros, 
        # on risque d'en avoir plein de stocké en parallèle
        # Exemple de KNN dans influenciae

        self.set_k(k)
        self.set_returns(returns)

    def set_k(self, k: int):
        assert isinstance(k, int) and k >= 1, f"k should be an int >= 1 and not {k}"
        self.k = k

    def set_returns(self, returns: Optional[Union[List[str], str]] = None):
        if returns is None:
            self.returns = ["examples"]
        elif isinstance(returns, str):
            possibilities = ["examples", "indices", "distances", "include_inputs"]
            if returns == "all":
                self.returns = possibilities
            elif returns in possibilities:
                self.returns = [returns]
            else:
                raise ValueError(f"{returns} should belong to {possibilities}")
        elif isinstance(returns, list):
            self.returns = returns
        else:
            raise ValueError(f"{returns} should either be `str` or `List[str]`")


    @abstractmethod
    @sanitize_input_output
    def find_samples(self, inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray]):
        """
        Search the samples to return as examples. Called by the explain methods.
        It may also return the indices corresponding to the samples,
        based on `return_indices` value.

        Parameters
        ----------
        inputs
            Dataset, Tensor or Array. Input samples to be explained.
            If Dataset, targets should not be provided (included in Dataset).
            Expected shape among (N, W), (N, T, W), (N, W, H, C).
            More information in the documentation.

        Returns
        -------
        selected_samples
            The desired samples among the search_set.
        samples_indices
            The indices corresponding to the samples.
        """
        raise NotImplementedError()

    def __call__(self, inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray]):
        """find_samples alias"""
        return self.find_samples(inputs)