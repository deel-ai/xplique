"""
Base model for example-based
"""

from abc import ABC, abstractmethod
import warnings

import tensorflow as tf
import numpy as np

from ..types import Callable, Dict, List, Optional, Type, Union

from ..commons import sanitize_inputs_targets
from .datasets_operations.harmonize import harmonize_datasets
from .datasets_operations.tf_dataset_operations import dataset_gather
from .search_methods import BaseSearchMethod
from .projections import Projection

from .search_methods.base import _sanitize_returns


class BaseExampleMethod(ABC):
    """
    Base class for natural example-based methods explaining classification models.
    An example-based method is a method that explains a model's predictions by providing
    examples from the cases_dataset (usually the training dataset). The examples are selected with
    the help of a search method that performs a search in the search space. The search space is
    defined with the help of a projection function that projects the cases_dataset
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
        Targets associated to the cases_dataset for dataset projection,
        oftentimes the one-hot encoding of a model's predictions. See `projection` for detail.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Batch size and cardinality of other datasets should match `cases_dataset`.
        Be careful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    k
        The number of examples to retrieve per input.
    projection
        Projection or Callable that project samples from the input space to the search space.
        The search space should be a space where distances are relevant for the model.
        It should not be `None`, otherwise, the model is not involved thus not explained.

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
    # pylint: disable=too-many-instance-attributes
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
        if projection is None:
            warnings.warn(
                "Example-based methods without projection will not explain the model."\
                + "To explain the model, consider using projections like the LatentSpaceProjection."
            )

        # set attributes
        self.cases_dataset, self.labels_dataset, self.targets_dataset, self.batch_size =\
            harmonize_datasets(cases_dataset, labels_dataset, targets_dataset, batch_size)

        self._search_returns = ["indices", "distances"]

        # check projection
        if isinstance(projection, Projection):
            self.projection = projection
        elif hasattr(projection, "__call__"):
            self.projection = Projection(get_weights=None, space_projection=projection)
        elif projection is None:
            self.projection = Projection(get_weights=None, space_projection=None)
        else:
            raise AttributeError(
                f"projection should be a `Projection` or a `Callable`, not a {type(projection)}"
            )

        # project dataset
        self.projected_cases_dataset = self.projection.project_dataset(self.cases_dataset,
                                                                       self.targets_dataset)

        # set properties
        self.k = k
        if self.labels_dataset is None\
                and ("labels" in case_returns or case_returns in ["all", "labels"]):
            raise AttributeError(
                "The method cannot return labels without a label dataset."
            )
        self.returns = case_returns

        # temporary value for the search method
        self.search_method = None

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
            The elements that can be returned are defined with the `_returns_possibilities`
            static attribute of the class.
        """
        # project inputs into the search space
        projected_inputs = self.projection(inputs, targets)

        # look for relevant elements in the search space
        search_output = self.search_method.find_examples(projected_inputs, targets)

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
            The elements that can be returned are defined with the `_returns_possibilities`
            static attribute of the class.
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
