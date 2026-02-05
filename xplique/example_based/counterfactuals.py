"""
Implementation of both counterfactuals and semi factuals methods for classification tasks.
"""

import numpy as np
import tensorflow as tf

from ..commons import sanitize_inputs_targets
from ..types import Callable, DatasetOrTensor, List, Optional, Union
from .base_example_method import BaseExampleMethod
from .projections import Projection
from .search_methods import ORDER, FilterKNN


class NaiveCounterFactuals(BaseExampleMethod):
    """
    This class allows to search for counterfactuals by searching for the closest sample to
    a query in a projection space that do not have the same model's prediction.
    It is a naive approach as it follows a greedy approach.

    Parameters
    ----------
    cases_dataset
        The dataset used to train the model, examples are extracted from this dataset.
        All datasets (cases, labels, and targets) should be of the same type.
        Supported types are: `tf.data.Dataset`, `torch.utils.data.DataLoader`,
        `tf.Tensor`, `np.ndarray`, `torch.Tensor`.
        For datasets with multiple columns, the first column is assumed to be the cases.
        While the second column is assumed to be the labels, and the third the targets.
        Warning: datasets tend to reshuffle at each iteration, ensure the datasets are
        not reshuffle as we use index in the dataset.
    targets_dataset
        Targets associated with the `cases_dataset` for dataset projection,
        oftentimes the one-hot encoding of a model's predictions. See `projection` for detail.
        They are also used to know the prediction of the model on the dataset.
        It should have the same type as `cases_dataset`.
    labels_dataset
        Labels associated with the examples in the `cases_dataset`.
        It should have the same type as `cases_dataset`.
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
        See the base class returns property for more details.
    batch_size
        Number of samples treated simultaneously for projection and search.
        Ignored if `cases_dataset` is a batched `tf.data.Dataset` or
        a batched `torch.utils.data.DataLoader` is provided.
    distance
        Distance function for examples search. It can be an integer, a string in
        {"manhattan", "euclidean", "cosine", "chebyshev", "inf"}, or a Callable,
        by default "euclidean".
    """

    # pylint: disable=duplicate-code

    def __init__(
        self,
        cases_dataset: DatasetOrTensor,
        targets_dataset: DatasetOrTensor,
        labels_dataset: Optional[DatasetOrTensor] = None,
        k: int = 1,
        projection: Union[Projection, Callable] = None,
        case_returns: Union[List[str], str] = "examples",
        batch_size: Optional[int] = None,
        distance: Union[int, str, Callable] = "euclidean",
    ):
        super().__init__(
            cases_dataset=cases_dataset,
            labels_dataset=labels_dataset,
            targets_dataset=targets_dataset,
            k=k,
            projection=projection,
            case_returns=case_returns,
            batch_size=batch_size,
        )

        # initiate search_method
        self.search_method = self.search_method_class(
            cases_dataset=self.projected_cases_dataset,
            targets_dataset=self.targets_dataset,
            k=self.k,
            search_returns=self._search_returns,
            batch_size=self.batch_size,
            distance=distance,
            filter_fn=self.filter_fn,
            order=ORDER.ASCENDING,
        )

    @property
    def search_method_class(self):
        """
        This property defines the search method class to use for the search.
        In this case, it is the FilterKNN that is an efficient KNN search method
        ignoring non-acceptable cases, thus not considering them in the search.
        """
        return FilterKNN

    def filter_fn(self, _, __, targets, cases_targets) -> tf.Tensor:
        """
        Filter function to mask the cases for which the model's prediction
        is different from the model's prediction on the inputs.
        """
        # get the labels predicted by the model
        # (n, )
        predicted_labels = tf.argmax(targets, axis=-1)

        # for each input, if the target label is the same as the predicted label
        # the mask as a True value and False otherwise
        label_targets = tf.argmax(cases_targets, axis=-1)  # (bs,)
        mask = tf.not_equal(tf.expand_dims(predicted_labels, axis=1), label_targets)  # (n, bs)
        return mask


class LabelAwareCounterFactuals(BaseExampleMethod):
    """
    This method will search the counterfactuals of a query within an expected class.
    This class should be provided with the query when calling the explain method.

    Parameters
    ----------
    cases_dataset
        The dataset used to train the model, examples are extracted from this dataset.
        All datasets (cases, labels, and targets) should be of the same type.
        Supported types are: `tf.data.Dataset`, `torch.utils.data.DataLoader`,
        `tf.Tensor`, `np.ndarray`, `torch.Tensor`.
        For datasets with multiple columns, the first column is assumed to be the cases.
        While the second column is assumed to be the labels, and the third the targets.
        Warning: datasets tend to reshuffle at each iteration, ensure the datasets are
        not reshuffle as we use index in the dataset.
    targets_dataset
        Targets associated with the `cases_dataset` for dataset projection,
        oftentimes the one-hot encoding of a model's predictions. See `projection` for detail.
        They are also used to know the prediction of the model on the dataset.
        It should have the same type as `cases_dataset`.
    labels_dataset
        Labels associated with the examples in the `cases_dataset`.
        It should have the same type as `cases_dataset`.
    k
        The number of examples to retrieve per input.
    projection
        Projection or Callable that project samples from the input space to the search space.
        The search space should be a space where distances are relevant for the model.
        It should not be `None`, otherwise, the model is not involved thus not explained.

        Example of Callable:
        ```
        def custom_projection(inputs: tf.Tensor, np.ndarray):
            '''
            Example of projection,
            inputs are the elements to project.
            '''
            projected_inputs = # do some magic on inputs, it should use the model.
            return projected_inputs
        ```
    case_returns
        String or list of string with the elements to return in `self.explain()`.
        See the base class returns property for more details.
    batch_size
        Number of samples treated simultaneously for projection and search.
        Ignored if `cases_dataset` is a batched `tf.data.Dataset` or
        a batched `torch.utils.data.DataLoader` is provided.
    distance
        Distance for the FilterKNN search method.
        Distance function for examples search. It can be an integer, a string in
        {"manhattan", "euclidean", "cosine", "chebyshev", "inf"}, or a Callable,
        by default "euclidean".
    """

    # pylint: disable=duplicate-code

    def __init__(
        self,
        cases_dataset: DatasetOrTensor,
        targets_dataset: DatasetOrTensor,
        labels_dataset: Optional[DatasetOrTensor] = None,
        k: int = 1,
        projection: Union[Projection, Callable] = None,
        case_returns: Union[List[str], str] = "examples",
        batch_size: Optional[int] = None,
        distance: Union[int, str, Callable] = "euclidean",
    ):
        super().__init__(
            cases_dataset=cases_dataset,
            labels_dataset=labels_dataset,
            targets_dataset=targets_dataset,
            k=k,
            projection=projection,
            case_returns=case_returns,
            batch_size=batch_size,
        )

        # initiate search_method
        self.search_method = self.search_method_class(
            cases_dataset=self.projected_cases_dataset,
            targets_dataset=self.targets_dataset,
            k=self.k,
            search_returns=self._search_returns,
            batch_size=self.batch_size,
            distance=distance,
            filter_fn=self.filter_fn,
            order=ORDER.ASCENDING,
        )

    @property
    def search_method_class(self):
        """
        This property defines the search method class to use for the search.
        In this case, it is the FilterKNN that is an efficient KNN search method ignoring
        non-acceptable cases, thus not considering them in the search.
        """
        return FilterKNN

    def filter_fn(self, _, __, cf_expected_classes, cases_targets) -> tf.Tensor:
        """
        Filter function to mask the cases for which the target is different from
        the target(s) expected for the counterfactuals.

        Parameters
        ----------
        cf_expected_classes
            The one-hot encoding of the target class for the counterfactuals.
        cases_targets
            The one-hot encoding of the target class for the cases.
        """
        cases_predicted_labels = tf.argmax(cases_targets, axis=-1)
        cf_label_targets = tf.argmax(cf_expected_classes, axis=-1)
        mask = tf.equal(tf.expand_dims(cf_label_targets, axis=1), cases_predicted_labels)
        return mask

    @sanitize_inputs_targets
    def explain(
        self,
        inputs: Union[tf.Tensor, np.ndarray],
        targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
        cf_expected_classes: Union[tf.Tensor, np.ndarray] = None,
    ):
        """
        Return the relevant CF examples to explain the inputs.
        The CF examples are searched within cases
        for which the target is the one provided in `cf_targets`.
        It projects inputs with `self.projection` in the search space and
        find examples with the `self.search_method`.

        Parameters
        ----------
        inputs
            Tensor or Array. Input samples to be explained.
            Expected shape among (N, W), (N, T, W), (N, W, H, C).
            More information in the documentation.
        targets
            Tensor or Array. One-hot encoded labels or regression target (e.g {+1, -1}),
            one for each sample. If not provided, the model's predictions are used.
            Targets associated to the `inputs` for projection.
            Shape: (n, nb_classes) where n is the number of samples and
            nb_classes is the number of classes.
            It is used in the `projection`. But `projection` can compute it internally.
        cf_expected_classes
            Tensor or Array. One-hot encoding of the target class for the counterfactuals.

        Returns
        -------
        return_dict
            Dictionary with listed elements in `self.returns`.
            The elements that can be returned are defined with the `_returns_possibilities`
            static attribute of the class.
        """
        assert cf_expected_classes is not None, "cf_expected_classes should be provided."

        # project inputs into the search space
        projected_inputs = self.projection(inputs, targets)

        # look for relevant elements in the search space
        search_output = self.search_method(projected_inputs, cf_expected_classes)

        # manage returned elements
        return self.format_search_output(search_output, inputs)

    def __call__(
        self,
        inputs: Union[tf.Tensor, np.ndarray],
        targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
        cf_expected_classes: Union[tf.Tensor, np.ndarray] = None,
    ):
        """explain() alias"""
        return self.explain(inputs, targets, cf_expected_classes)
