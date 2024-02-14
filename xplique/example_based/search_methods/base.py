"""
Base search method for example-based module
"""

from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np

from ...types import Callable, Union, Optional, List

from ...commons import sanitize_dataset


def _sanitize_returns(returns: Optional[Union[List[str], str]] = None,
                      possibilities: List[str] = None,
                      default: Union[List[str], str] = None):
    """
    Factorization of `set_returns` for `BaseSearchMethod` and `SimilarExamples`.
    It cleans the `returns` parameter.
    Results is either a sublist of possibilities or a value among possibilities.

    Parameters
    ----------
    returns
        The value to verify and put to the `instance.returns` attribute.
    possibilities
        List of possible unit values for `instance.returns`.
    default
        Value in case `returns` is None.

    Returns
    -------
    returns
        The cleaned `returns` value.
    """
    if possibilities is None:
        possibilities = ["examples"]
    if default is None:
        default = ["examples"]

    if returns is None:
        returns = default
    elif isinstance(returns, str):
        if returns == "all":
            returns = possibilities
        elif returns in possibilities:
            returns = [returns]
        else:
            raise ValueError(f"{returns} should belong to {possibilities}")
    elif isinstance(returns, list):
        pass  # already in the right format.
    else:
        raise ValueError(f"{returns} should either be `str` or `List[str]`")

    return returns


class BaseSearchMethod(ABC):
    """
    Base class used by `NaturalExampleBasedExplainer` search examples in
    a meaningful space for the model. It can also be used alone but will not provided
    model explanations.

    Parameters
    ----------
    cases_dataset
        The dataset used to train the model, examples are extracted from the dataset.
        For natural example-based methods it is the train dataset.
    k
        The number of examples to retrieve.
    search_returns
        String or list of string with the elements to return in `self.find_examples()`.
        See `self.set_returns()` for detail.
    batch_size
        Number of sample treated simultaneously.
        It should match the batch size of the `search_set` in the case of a `tf.data.Dataset`.
    """

    def __init__(
        self,
        cases_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        k: int = 1,
        search_returns: Optional[Union[List[str], str]] = None,
        batch_size: Optional[int] = 32,
    ): # pylint: disable=R0801
        
        # set batch size
        if hasattr(cases_dataset, "_batch_size"):
            self.batch_size = cases_dataset._batch_size
        else:
            self.batch_size = batch_size

        self.cases_dataset = sanitize_dataset(cases_dataset, self.batch_size)

        self.set_k(k)
        self.set_returns(search_returns)

    def set_k(self, k: int):
        """
        Change value of k with constructing a new `BaseSearchMethod`.
        It is useful because the constructor can be computationally expensive.

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
        possibilities = ["examples", "indices", "distances", "include_inputs"]
        default = "examples"
        self.returns = _sanitize_returns(returns, possibilities, default)


    @abstractmethod
    def find_examples(self, inputs: Union[tf.Tensor, np.ndarray]):
        """
        Search the samples to return as examples. Called by the explain methods.
        It may also return the indices corresponding to the samples,
        based on `self.returns` value.

        Parameters
        ----------
        inputs
            Tensor or Array. Input samples to be explained.
            Expected shape among (N, W), (N, T, W), (N, W, H, C).
        """
        raise NotImplementedError()

    def __call__(self, inputs: Union[tf.Tensor, np.ndarray]):
        """find_samples alias"""
        return self.find_examples(inputs)
