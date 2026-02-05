"""
Base search method for example-based module
"""

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import tensorflow as tf

from ...types import List, Optional, Union
from ..datasets_operations.tf_dataset_operations import sanitize_dataset


class ORDER(Enum):
    """
    Enumeration for the two types of ordering for the sorting function.
    ASCENDING puts the elements with the smallest value first.
    DESCENDING puts the elements with the largest value first.
    """

    ASCENDING = 1
    DESCENDING = 2


def _sanitize_returns(
    returns: Optional[Union[List[str], str]] = None,
    possibilities: List[str] = None,
    default: Union[List[str], str] = None,
) -> List[str]:
    """
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
    Base class for the example-based search methods. This class is abstract.
    It should be inherited by the search methods that are used to find examples in a dataset.
    It also defines the interface for the search methods.

    Parameters
    ----------
    cases_dataset
        The dataset containing the examples to search in.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Be careful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    k
        The number of examples to retrieve at each call.
    search_returns
        String or list of string with the elements to return in `self.find_examples()`.
        It should be a subset of `self._returns_possibilities` or `"all"`.
        See self.returns setter for more detail.
    batch_size
        Number of samples treated simultaneously.
        It should match the batch size of the cases_dataset in the case of a `tf.data.Dataset`.
    """

    _returns_possibilities = ["examples", "indices", "distances", "include_inputs"]

    def __init__(
        self,
        cases_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        k: int = 1,
        search_returns: Optional[Union[List[str], str]] = None,
        batch_size: Optional[int] = 32,
    ):
        # set batch size
        if isinstance(cases_dataset, tf.data.Dataset):
            self.batch_size = tf.shape(next(iter(cases_dataset)))[0].numpy()
        else:
            self.batch_size = batch_size

        self.cases_dataset = sanitize_dataset(cases_dataset, self.batch_size)

        self.k = k
        self.returns = search_returns

    @property
    def k(self) -> int:
        """Getter for the k parameter."""
        return self._k

    @k.setter
    def k(self, k: int):
        """Setter for the k parameter."""
        assert isinstance(k, int) and k >= 1, f"k should be an int >= 1 and not {k}"
        self._k = k

    @property
    def returns(self) -> Union[List[str], str]:
        """Getter for the returns parameter."""
        return self._returns

    @returns.setter
    def returns(self, returns: Union[List[str], str]):
        """
        Setter for the returns parameter used to define returned elements in `self.explain()`.

        Parameters
        ----------
        returns
            Most elements are useful in `xplique.plots.plot_examples()`.
            `returns` can be set to 'all' for all possible elements to be returned.
                - 'examples' correspond to the expected examples,
                the inputs may be included in first position. (n, k(+1), ...)
                - 'distances' the distances between the inputs and the corresponding examples.
                They are associated to the examples. (n, k, ...)
                - 'labels' if provided through `dataset_labels`,
                they are the labels associated with the examples. (n, k, ...)
                - 'include_inputs' specify if inputs should be included in the returned elements.
                Note that it changes the number of returned elements from k to k+1.
        """
        default = "examples"
        self._returns = _sanitize_returns(returns, self._returns_possibilities, default)

    @abstractmethod
    def find_examples(
        self,
        inputs: Union[tf.Tensor, np.ndarray],
        targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
    ) -> dict:
        """
        Search the samples to return as examples. Called by the explain methods.
        It may also return the indices corresponding to the samples,
        based on `self.returns` value.

        Parameters
        ----------
        inputs
            Tensor or Array. Input samples to be explained.
            Expected shape among (N, W), (N, T, W), (N, W, H, C).
        targets
            Tensor or Array. Target of the samples to be explained.

        Returns
        -------
        return_dict
            Dictionary containing the elements to return which are specified in `self.returns`.
        """
        raise NotImplementedError()

    def __call__(
        self,
        inputs: Union[tf.Tensor, np.ndarray],
        targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
    ) -> dict:
        """find_samples() alias"""
        return self.find_examples(inputs, targets)
