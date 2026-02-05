"""
Base model for example-based
"""

import tensorflow as tf

from ..attributions.base import BlackBoxExplainer
from ..types import Callable, DatasetOrTensor, List, Optional, Type, Union
from .base_example_method import BaseExampleMethod
from .projections import AttributionProjection, HadamardProjection, Projection
from .search_methods import KNN, ORDER, BaseSearchMethod


class SimilarExamples(BaseExampleMethod):
    """
    Class for similar example-based method. This class allows to search the k Nearest Neighbor
    of an input in the projected space (defined by the projection method)
    using the distance defined by the distance method provided.

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
    labels_dataset
        Labels associated with the examples in the `cases_dataset`.
        It should have the same type as `cases_dataset`.
    targets_dataset
        Targets associated with the `cases_dataset` for dataset projection,
        oftentimes the one-hot encoding of a model's predictions. See `projection` for detail.
        It should have the same type as `cases_dataset`.
        It is not be necessary for all projections.
        Furthermore, projections which requires it compute it internally by default.
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
        Distance for the knn search method. It can be an integer, a string in
        {"manhattan", "euclidean", "cosine", "chebyshev", "inf"}, or a Callable,
        by default "euclidean".
    """

    def __init__(
        self,
        cases_dataset: DatasetOrTensor,
        labels_dataset: Optional[DatasetOrTensor] = None,
        targets_dataset: Optional[DatasetOrTensor] = None,
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
            search_returns=self._search_returns,
            k=self.k,
            batch_size=self.batch_size,
            distance=distance,
            order=ORDER.ASCENDING,
        )

    @property
    def search_method_class(self) -> Type[BaseSearchMethod]:
        return KNN


class Cole(SimilarExamples):
    """
    Cole is a similar examples method that gives the most similar examples
    to a query in some specific projection space.
    Cole uses the model to build a search space so that distances are meaningful for the model.
    It uses attribution methods to weight inputs.
    Those attributions may be computed in the latent space for high-dimensional data like images.

    It is an implementation of a method proposed by Kenny et Keane in 2019,
    Twin-Systems to Explain Artificial Neural Networks using Case-Based Reasoning:
    https://researchrepository.ucd.ie/handle/10197/11064

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
    labels_dataset
        Labels associated with the examples in the `cases_dataset`.
        It should have the same type as `cases_dataset`.
    targets_dataset
        Targets associated with the `cases_dataset` for dataset projection,
        oftentimes the one-hot encoding of a model's predictions. See `projection` for detail.
        It should have the same type as `cases_dataset`.
        It is not be necessary for all projections.
        Furthermore, projections which requires it compute it internally by default.
    k
        The number of examples to retrieve per input.
    distance
        Distance function for examples search. It can be an integer, a string in
        {"manhattan", "euclidean", "cosine", "chebyshev", "inf"}, or a Callable,
        by default "euclidean".
    case_returns
        String or list of string with the elements to return in `self.explain()`.
        See the base class returns property for details.
    batch_size
        Number of samples treated simultaneously for projection and search.
        Ignored if `cases_dataset` is a batched `tf.data.Dataset` or
        a batched `torch.utils.data.DataLoader` is provided.
    latent_layer
        Layer used to split the model, the first part will be used for projection and
        the second to compute the attributions. By default, the model is not split.
        For such split, the `model` should be a `tf.keras.Model`.

        If an `int` is provided it will be interpreted as a layer index.
        If a `string` is provided it will look for the layer name.

        The method as described in the paper apply the separation on the last convolutional layer.
        To do so, the `"last_conv"` parameter will extract it.
        Otherwise, `-1` could be used for the last layer before softmax.
    attribution_method
        Class of the attribution method to use for projection.
        It should inherit from `xplique.attributions.base.BlackBoxExplainer`.
        It can also be `"gradient"` to make the hadamard product between with the gradient.
        It was deemed the best method in the original paper, and we optimized it for speed.
        By default, it is set to `"gradient"`.
    attribution_kwargs
        Parameters to be passed for the construction of the `attribution_method`.
    """

    def __init__(
        self,
        cases_dataset: DatasetOrTensor,
        model: Union[tf.keras.Model, "torch.nn.Module"],
        labels_dataset: Optional[DatasetOrTensor] = None,
        targets_dataset: Optional[DatasetOrTensor] = None,
        k: int = 1,
        distance: Union[str, Callable] = "euclidean",
        case_returns: Optional[Union[List[str], str]] = "examples",
        batch_size: Optional[int] = None,
        latent_layer: Optional[Union[str, int]] = None,
        attribution_method: Union[str, Type[BlackBoxExplainer]] = "gradient",
        **attribution_kwargs,
    ):
        assert targets_dataset is not None

        # build the corresponding projection
        if isinstance(attribution_method, str) and attribution_method.lower() == "gradient":
            operator = attribution_kwargs.get("operator", None)

            projection = HadamardProjection(
                model=model,
                latent_layer=latent_layer,
                operator=operator,
            )
        elif issubclass(attribution_method, BlackBoxExplainer):
            # build attribution projection
            projection = AttributionProjection(
                model=model,
                attribution_method=attribution_method,
                latent_layer=latent_layer,
                **attribution_kwargs,
            )
        else:
            raise ValueError(
                "`attribution_method` should be 'gradient' or a subclass of BlackBoxExplainer, "
                + f"not {attribution_method}"
            )

        super().__init__(
            cases_dataset=cases_dataset,
            targets_dataset=targets_dataset,
            labels_dataset=labels_dataset,
            projection=projection,
            k=k,
            case_returns=case_returns,
            batch_size=batch_size,
            distance=distance,
        )
