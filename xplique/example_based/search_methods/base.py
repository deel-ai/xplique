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
    Base class used by `NaturalExampleBasedExplainer` search examples in
    a meaningful space for the model. It can also be used alone but will not provided
    model explanations.

    Parameters
    ----------
    search_set
        The dataset from which examples should be extracted.
        For natural example-based methods it is the train dataset.
    k
        The number of examples to retrieve.
    returns
        String or list of string with the elements to return in `self.find_examples()`.
        See `self.set_returns()` for detail.
    """
    def __init__(self,
                 search_set: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 k: int = 1,
                 returns: Optional[Union[List[str], str]] = None):
        self.search_set = search_set
        self.set_k(k)
        self.set_returns(returns)

    def set_k(self, k: int):
        """
        Change value of k with constructing a new `BaseSearchMethod`.
        It is useful because the constructor can be computionnaly expensive.
        
        Parameters
        ----------
        k
            The number of examples to retrieve.
        """
        assert isinstance(k, int) and k >= 1, f"k should be an int >= 1 and not {k}"
        self.k = k

    def set_returns(self, returns: Optional[Union[List[str], str]] = None):
        """
        Set `self.returns` used to define returned elements in `self.find_examples()`.
        
        Parameters
        ----------
        returns
            Most elements are useful in `xplique.plots.plot_examples()`.
            `returns` can be set to 'all' for all possible elements to be returned.
                - 'examples' correspond to the expected examples,
                the inputs may be included in first position. (n, k(+1), ...)
                - 'indices' the indices of the examples in the `search_set`.
                Used to retrieve the original example and labels. (n, k, ...)
                - 'distances' the distances between the inputs and the corresponding examples.
                They are associated to the examples. (n, k, ...)
                - 'include_inputs' specify if inputs should be included in the returned elements.
                Note that it changes the number of returned elements from k to k+1.
        """
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
    def find_examples(self, inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray]):
        """
        Search the samples to return as examples. Called by the explain methods.
        It may also return the indices corresponding to the samples,
        based on `self.returns` value.

        Parameters
        ----------
        inputs
            Dataset, Tensor or Array. Input samples to be explained.
            Expected shape among (N, W), (N, T, W), (N, W, H, C).
        """
        raise NotImplementedError()

    def __call__(self, inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray]):
        """find_samples alias"""
        return self.find_examples(inputs)