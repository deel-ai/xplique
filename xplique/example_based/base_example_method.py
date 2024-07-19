"""
Base model for example-based
"""

from abc import ABC, abstractmethod

import math

import tensorflow as tf
import numpy as np

from ..types import Callable, Dict, List, Optional, Type, Union

from ..commons import sanitize_inputs_targets
from ..commons import sanitize_dataset, dataset_gather
from .search_methods import BaseSearchMethod
from .projections import Projection

from .search_methods.base import _sanitize_returns


class BaseExampleMethod(ABC):
    """
    Base class for natural example-based methods explaining classification models.
    An example-based method is a method that explains a model's predictions by providing examples from the cases_dataset
    (usually the training dataset). The examples are selected with the help of a search method that performs a search in
    the search space. The search space is defined with the help of a projection function that projects the cases_dataset
    and the (inputs, targets) to explain into a space where the search method is relevant.

    Parameters
    ----------
    cases_dataset
        The dataset used to train the model, examples are extracted from this dataset.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Be careful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    labels_dataset
        Labels associated to the examples in the dataset. Indices should match with cases_dataset.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Batch size and cardinality of other datasets should match `cases_dataset`.
        Be careful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    targets_dataset
        Targets associated to the cases_dataset for dataset projection, oftentimes the one-hot encoding of a model's
        predictions. See `projection` for detail.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Batch size and cardinality of other datasets should match `cases_dataset`.
        Be careful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    k
        The number of examples to retrieve per input.
    projection
        Projection or Callable that project samples from the input space to the search space.
        The search space should be a space where distances are relevant for the model.
        It should not be `None`, otherwise, the model is not involved thus not explained. If you are interested in
        searching the input space, you should use a `BaseSearchMethod` instead. 

        Example of Callable:
        ```
        def custom_projection(inputs: tf.Tensor, np.ndarray, targets: tf.Tensor, np.ndarray = None):
            '''
            Example of projection,
            inputs are the elements to project.
            targets are optional parameters to orientated the projection.
            '''
            projected_inputs = # do some magic on inputs, it should use the model.
            return projected_inputs
        ```
    case_returns
        String or list of string with the elements to return in `self.explain()`.
        See the returns property for details.
    batch_size
        Number of sample treated simultaneously for projection and search.
        Ignored if `tf.data.Dataset` are provided (those are supposed to be batched).
    """
    _returns_possibilities = ["examples", "distances", "labels", "include_inputs"]

    def __init__(
        self,
        cases_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        labels_dataset: Optional[Union[tf.data.Dataset, tf.Tensor, np.ndarray]] = None,
        targets_dataset: Optional[Union[tf.data.Dataset, tf.Tensor, np.ndarray]] = None,
        k: int = 1,
        projection: Union[Projection, Callable] = None,
        case_returns: Union[List[str], str] = "examples",
        batch_size: Optional[int] = 32,
    ):
        assert (
            projection is not None
        ), "`BaseExampleMethod` without Projection method should be a `BaseSearchMethod`."

        # set attributes
        self.batch_size = self._initialize_cases_dataset(
            cases_dataset, labels_dataset, targets_dataset, batch_size
        )

        self._search_returns = ["indices", "distances"]

        # check projection
        assert hasattr(projection, "__call__"), "projection should be a callable."
        if isinstance(projection, Projection):
            self.projection = projection
        elif hasattr(projection, "__call__"):
            self.projection = Projection(get_weights=None, space_projection=projection)
        else:
            raise AttributeError(
                "projection should be a `Projection` or a `Callable`, not a"
                + f"{type(projection)}"
            )

        # project dataset
        self.projected_cases_dataset = self.projection.project_dataset(self.cases_dataset,
                                                                       self.targets_dataset)

        # set properties
        self.k = k
        self.returns = case_returns
    
    @property
    @abstractmethod
    def search_method_class(self) -> Type[BaseSearchMethod]:
        """
        When inheriting from `BaseExampleMethod`, one should define the search method class to use.
        """
        raise NotImplementedError

    @property
    def k(self) -> int:
        """Getter for the k parameter."""
        return self._k

    @k.setter
    def k(self, k: int):
        """Setter for the k parameter."""
        assert isinstance(k, int) and k >= 1, f"k should be an int >= 1 and not {k}"
        self._k = k

        try:
            self.search_method.k = k
        except AttributeError:
            pass

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

    def _initialize_cases_dataset(
        self,
        cases_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        labels_dataset: Optional[Union[tf.data.Dataset, tf.Tensor, np.ndarray]],
        targets_dataset: Optional[Union[tf.data.Dataset, tf.Tensor, np.ndarray]],
        batch_size: Optional[int],
    ) -> int:
        """
        Factorization of `__init__()` method for dataset related attributes.

        Parameters
        ----------
        cases_dataset
            The dataset used to train the model, examples are extracted from this dataset.
        labels_dataset
            Labels associated to the examples in the cases_dataset.
            Indices should match with cases_dataset.
        targets_dataset
            Targets associated to the cases_dataset for dataset projection.
            See `projection` for details.
        batch_size
            Number of sample treated simultaneously when using the datasets.
            Ignored if `tf.data.Dataset` are provided (those are supposed to be batched).

        Returns
        -------
        batch_size
            Number of sample treated simultaneously when using the datasets.
            Extracted from the datasets in case they are `tf.data.Dataset`.
            Otherwise, the input value.
        """
        # at least one dataset provided
        if isinstance(cases_dataset, tf.data.Dataset):
            # set batch size (ignore provided argument) and cardinality
            if isinstance(cases_dataset.element_spec, tuple):
                batch_size = tf.shape(next(iter(cases_dataset))[0])[0].numpy()
            else:
                batch_size = tf.shape(next(iter(cases_dataset)))[0].numpy()

            cardinality = cases_dataset.cardinality().numpy()
        else:
            # if cases_dataset is not a `tf.data.Dataset`, then neither should the other.
            assert not isinstance(labels_dataset, tf.data.Dataset), (
                "if the cases_dataset is not a `tf.data.Dataset`, "
                + "then neither should the labels_dataset."
            )
            assert not isinstance(targets_dataset, tf.data.Dataset), (
                "if the cases_dataset is not a `tf.data.Dataset`, "
                + "then neither should the targets_dataset."
            )
            # set batch size and cardinality
            batch_size = min(batch_size, len(cases_dataset))
            cardinality = math.ceil(len(cases_dataset) / batch_size)

        # verify cardinality and create datasets from the tensors
        self.cases_dataset = sanitize_dataset(
            cases_dataset, batch_size, cardinality
        )
        self.labels_dataset = sanitize_dataset(
            labels_dataset, batch_size, cardinality
        )
        self.targets_dataset = sanitize_dataset(
            targets_dataset, batch_size, cardinality
        )

        # if the provided `cases_dataset` has several columns
        if isinstance(self.cases_dataset.element_spec, tuple):
            # switch case on the number of columns of `cases_dataset`
            if len(self.cases_dataset.element_spec) == 2:
                assert self.labels_dataset is None, (
                    "The second column of `cases_dataset` is assumed to be the labels. "
                    + "Hence, `labels_dataset` should be empty."
                )
                self.labels_dataset = self.cases_dataset.map(lambda x, y: y)
                self.cases_dataset = self.cases_dataset.map(lambda x, y: x)

            elif len(self.cases_dataset.element_spec) == 3:
                assert self.labels_dataset is None, (
                    "The second column of `cases_dataset` is assumed to be the labels. "
                    + "Hence, `labels_dataset` should be empty."
                )
                assert self.targets_dataset is None, (
                    "The second column of `cases_dataset` is assumed to be the labels. "
                    + "Hence, `labels_dataset` should be empty."
                )
                self.targets_dataset = self.cases_dataset.map(lambda x, y, t: t)
                self.labels_dataset = self.cases_dataset.map(lambda x, y, t: y)
                self.cases_dataset = self.cases_dataset.map(lambda x, y, t: x)
            else:
                raise AttributeError(
                    "`cases_dataset` cannot possess more than 3 columns, "
                    + f"{len(self.cases_dataset.element_spec)} were detected."
                )

        # prefetch datasets
        self.cases_dataset = self.cases_dataset.prefetch(tf.data.AUTOTUNE)
        if self.labels_dataset is not None:
            self.labels_dataset = self.labels_dataset.prefetch(tf.data.AUTOTUNE)
        if self.targets_dataset is not None:
            self.targets_dataset = self.targets_dataset.prefetch(tf.data.AUTOTUNE)

        return batch_size

    @sanitize_inputs_targets
    def explain(
        self,
        inputs: Union[tf.Tensor, np.ndarray],
        targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
    ):
        """
        Return the relevant examples to explain the (inputs, targets).
        It projects inputs with `self.projection` in the search space
        and find examples with the `self.search_method`.

        Parameters
        ----------
        inputs
            Tensor or Array. Input samples to be explained.
            Expected shape among (N, W), (N, T, W), (N, W, H, C).
            More information in the documentation.
        targets
            Targets associated to the cases_dataset for dataset projection.
            See `projection` for details.

        Returns
        -------
        return_dict
            Dictionary with listed elements in `self.returns`.
            The elements that can be returned are defined with _returns_possibilities static attribute of the class.
        """
        # project inputs into the search space
        projected_inputs = self.projection(inputs, targets)

        # look for relevant elements in the search space
        search_output = self.search_method(projected_inputs, targets)

        # manage returned elements
        return self.format_search_output(search_output, inputs)

    def __call__(
        self,
        inputs: Union[tf.Tensor, np.ndarray],
        targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
    ):
        """explain() alias"""
        return self.explain(inputs, targets)

    def format_search_output(
        self,
        search_output: Dict[str, tf.Tensor],
        inputs: Union[tf.Tensor, np.ndarray],
    ):
        """
        Format the output of the `search_method` to match the expected returns in `self.returns`.

        Parameters
        ----------
        search_output
            Dictionary with the required outputs from the `search_method`.
        inputs
            Tensor or Array. Input samples to be explained.
            Expected shape among (N, W), (N, T, W), (N, W, H, C).
        # targets
        #     Targets associated to the cases_dataset for dataset projection.
        #     See `projection` for details.

        Returns
        -------
        return_dict
            Dictionary with listed elements in `self.returns`.
            The elements that can be returned are defined with _returns_possibilities static attribute of the class.
        """
        # initialize return dictionary
        return_dict = {}

        # gather examples, labels, and targets from the example's indices of the search output
        examples = dataset_gather(self.cases_dataset, search_output["indices"])
        examples_labels = dataset_gather(self.labels_dataset, search_output["indices"])

        # add examples and weights
        if "examples" in self.returns:  #  or "weights" in self.returns:
            if "include_inputs" in self.returns:
                # include inputs
                inputs = tf.expand_dims(inputs, axis=1)
                examples = tf.concat([inputs, examples], axis=1)
            if "examples" in self.returns:
                return_dict["examples"] = examples

        # add indices, distances, and labels
        if "indices" in self.returns:
            return_dict["indices"] = search_output["indices"]
        if "distances" in self.returns:
            return_dict["distances"] = search_output["distances"]
        if "labels" in self.returns:
            assert (
                examples_labels is not None
            ), "The method cannot return labels without a label dataset."
            return_dict["labels"] = examples_labels

        return return_dict
