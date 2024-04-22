"""
Base model for example-based
"""

import math

import tensorflow as tf
import numpy as np

from ..types import Callable, Dict, List, Optional, Type, Union

from ..commons import sanitize_inputs_targets
from ..commons import sanitize_dataset, dataset_gather
from .search_methods import KNN, BaseSearchMethod
from .projections import Projection

from .search_methods.base import _sanitize_returns


class BaseExampleMethod:
    """
    Base class for natural example-based methods explaining models,
    they project the cases_dataset into a pertinent space for the with a `Projection`,
    then they call the `BaseSearchMethod` on it.

    Parameters
    ----------
    cases_dataset
        The dataset used to train the model, examples are extracted from the dataset.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Be careful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    labels_dataset
        Labels associated to the examples in the dataset. Indices should match with cases_dataset.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Batch size and cardinality of other dataset should match `cases_dataset`.
        Be careful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    targets_dataset
        Targets associated to the cases_dataset for dataset projection. See `projection` for detail.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Batch size and cardinality of other dataset should match `cases_dataset`.
        Be careful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    search_method
        An algorithm to search the examples in the projected space.
    k
        The number of examples to retrieve.
    projection
        Projection or Callable that project samples from the input space to the search space.
        The search space should be a space where distance make sense for the model.
        It should not be `None`, otherwise,
        all examples could be computed only with the `search_method`.

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
        See `self.set_returns()` for detail.
    batch_size
        Number of sample treated simultaneously for projection and search.
        Ignored if `tf.data.Dataset` are provided (those are supposed to be batched).
    search_method_kwargs
        Parameters to be passed at the construction of the `search_method`.
    """

    def __init__(
        self,
        cases_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        labels_dataset: Optional[Union[tf.data.Dataset, tf.Tensor, np.ndarray]] = None,
        targets_dataset: Optional[Union[tf.data.Dataset, tf.Tensor, np.ndarray]] = None,
        search_method: Type[BaseSearchMethod] = KNN,
        k: int = 1,
        projection: Union[Projection, Callable] = None,
        case_returns: Union[List[str], str] = "examples",
        batch_size: Optional[int] = 32,
        **search_method_kwargs,
    ):
        assert (
            projection is not None
        ), "`BaseExampleMethod` without `projection` is a `BaseSearchMethod`."

        # set attributes
        batch_size = self.__initialize_cases_dataset(
            cases_dataset, labels_dataset, targets_dataset, batch_size
        )

        self.k = k
        self.set_returns(case_returns)

        assert hasattr(projection, "__call__"), "projection should be a callable."

        # check projection type
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
        projected_cases_dataset = self.projection.project_dataset(self.cases_dataset,
                                                                  self.targets_dataset)

        # set `search_returns` if not provided and overwrite it otherwise
        search_method_kwargs["search_returns"] = ["indices", "distances"]

        # initiate search_method
        self.search_method = search_method(
            cases_dataset=projected_cases_dataset,
            k=k,
            batch_size=batch_size,
            targets_dataset=self.targets_dataset,
            **search_method_kwargs,
        )

    def __initialize_cases_dataset(
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
            The dataset used to train the model, examples are extracted from the dataset.
        labels_dataset
            Labels associated to the examples in the dataset.
            Indices should match with cases_dataset.
        targets_dataset
            Targets associated to the cases_dataset for dataset projection.
            See `projection` for detail.
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
                input_batch = next(iter(cases_dataset))[0]
                if isinstance(input_batch, dict): # for the case where input is a dict (HF)
                    assert ("input_ids" in input_batch.keys()), f"As the input batch is a dictionnary we expect it to \
                          be a dictionnary as expected by Hugging Face model thus containing 'input_ids'. The dict \
                            keys are {input_batch.keys()}."
                    inp = input_batch['input_ids']
                    batch_size = tf.shape(inp)[0].numpy()
                else:
                    batch_size = tf.shape(input_batch)[0].numpy()
            else:
                batch_size = tf.shape(next(iter(cases_dataset)))[0].numpy()

            cardinality = cases_dataset.cardinality().numpy()
        else:
            # if case_dataset is not a `tf.data.Dataset`, then neither should the other.
            assert not isinstance(labels_dataset, tf.data.Dataset)
            assert not isinstance(targets_dataset, tf.data.Dataset)
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
                    "The second column of `cases_dataset` is assumed to be the labels."
                    + "Hence, `labels_dataset` should be empty."
                )
                self.labels_dataset = self.cases_dataset.map(lambda x, y: y)
                self.cases_dataset = self.cases_dataset.map(lambda x, y: x)

            elif len(self.cases_dataset.element_spec) == 3:
                assert self.labels_dataset is None, (
                    "The second column of `cases_dataset` is assumed to be the labels."
                    + "Hence, `labels_dataset` should be empty."
                )
                assert self.targets_dataset is None, (
                    "The second column of `cases_dataset` is assumed to be the labels."
                    + "Hence, `labels_dataset` should be empty."
                )
                self.targets_dataset = self.cases_dataset.map(lambda x, y, t: t)
                self.labels_dataset = self.cases_dataset.map(lambda x, y, t: y)
                self.cases_dataset = self.cases_dataset.map(lambda x, y, t: x)
            else:
                raise AttributeError(
                    "`cases_dataset` cannot possess more than 3 columns,"
                    + f"{len(self.cases_dataset.element_spec)} were detected."
                )

        self.cases_dataset = self.cases_dataset.prefetch(tf.data.AUTOTUNE)
        if self.labels_dataset is not None:
            self.labels_dataset = self.labels_dataset.prefetch(tf.data.AUTOTUNE)
        if self.targets_dataset is not None:
            self.targets_dataset = self.targets_dataset.prefetch(tf.data.AUTOTUNE)

        return batch_size

    def set_k(self, k: int):
        """
        Setter for the k parameter.

        Parameters
        ----------
        k
            Number of examples to return, it should be a positive integer.
        """
        assert isinstance(k, int) and k >= 1, f"k should be an int >= 1 and not {k}"
        self.k = k
        self.search_method.set_k(k)

    def set_returns(self, returns: Union[List[str], str]):
        """
        Set `self.returns` used to define returned elements in `self.explain()`.

        Parameters
        ----------
        returns
            Most elements are useful in `xplique.plots.plot_examples()`.
            `returns` can be set to 'all' for all possible elements to be returned.
                - 'examples' correspond to the expected examples,
                the inputs may be included in first position. (n, k(+1), ...)
                - 'weights' the weights in the input space used in the projection.
                They are associated to the input and the examples. (n, k(+1), ...)
                - 'distances' the distances between the inputs and the corresponding examples.
                They are associated to the examples. (n, k, ...)
                - 'labels' if provided through `dataset_labels`,
                they are the labels associated with the examples. (n, k, ...)
                - 'include_inputs' specify if inputs should be included in the returned elements.
                Note that it changes the number of returned elements from k to k+1.
        """
        possibilities = ["examples", "weights", "distances", "labels", "include_inputs"]
        default = "examples"
        self.returns = _sanitize_returns(returns, possibilities, default)

    @sanitize_inputs_targets
    def explain(
        self,
        inputs: Union[tf.Tensor, np.ndarray],
        targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
    ):
        """
        Compute examples to explain the inputs.
        It project inputs with `self.projection` in the search space
        and find examples with `self.search_method`.

        Parameters
        ----------
        inputs
            Tensor or Array. Input samples to be explained.
            Expected shape among (N, W), (N, T, W), (N, W, H, C).
            More information in the documentation.
        targets
            Tensor or Array passed to the projection function.

        Returns
        -------
        return_dict
            Dictionary with listed elements in `self.returns`.
            If only one element is present it returns the element.
            The elements that can be returned are:
            examples, weights, distances, indices, and labels.
        """
        # project inputs
        projected_inputs = self.projection(inputs, targets)

        # look for closest elements to projected inputs
        search_output = self.search_method(projected_inputs, targets)

        # manage returned elements
        return self.format_search_output(search_output, inputs, targets)

    def __call__(
        self,
        inputs: Union[tf.Tensor, np.ndarray],
        targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
    ):
        """explain alias"""
        return self.explain(inputs, targets)

    def format_search_output(
        self,
        search_output: Dict[str, tf.Tensor],
        inputs: Union[tf.Tensor, np.ndarray],
        targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
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
            More information in the documentation.
        targets
            Tensor or Array passed to the projection function.
            Here it is used by the explain function of attribution methods.
            Refer to the corresponding method documentation for more detail.
            Note that the default method is `Saliency`.

        Returns
        -------
        return_dict
            Dictionary with listed elements in `self.returns`.
            If only one element is present it returns the element.
            The elements that can be returned are:
            examples, weights, distances, indices, and labels.
        """
        return_dict = {}

        examples = dataset_gather(self.cases_dataset, search_output["indices"])
        examples_labels = dataset_gather(self.labels_dataset, search_output["indices"])
        examples_targets = dataset_gather(
            self.targets_dataset, search_output["indices"]
        )

        # add examples and weights
        if "examples" in self.returns or "weights" in self.returns:
            if "include_inputs" in self.returns:
                # include inputs
                if isinstance(inputs, dict):
                    assert "input_ids" in inputs.keys(), f"Expected inputs to be a dictionnary with 'input_ids' as key. \
                        The keys are {inputs.keys()}."
                    input_ids = inputs["input_ids"]
                    attention_mask = inputs["attention_mask"]
                    input_ids = tf.expand_dims(input_ids, axis=1)
                    attention_mask = tf.expand_dims(attention_mask, axis=1)
                    examples["input_ids"] = tf.concat([tf.cast(input_ids, dtype=examples["input_ids"].dtype), examples["input_ids"]], axis=1)
                    examples["attention_mask"] = tf.concat([tf.cast(attention_mask, dtype= examples["attention_mask"].dtype), examples["attention_mask"]], axis=1)
                else:
                    inputs = tf.expand_dims(inputs, axis=1)
                    examples = tf.concat([inputs, examples], axis=1)
                if targets is not None:
                    targets = tf.expand_dims(targets, axis=1)
                    examples_targets = tf.concat([targets, examples_targets], axis=1)
                else:
                    examples_targets = [None] * len(examples)
            if "examples" in self.returns:
                return_dict["examples"] = examples
            if "weights" in self.returns:
                # get weights of examples (n, k, ...)
                # we iterate on the inputs dimension through maps
                # and ask weights for batch of examples
                weights = []
                for ex, ex_targ in zip(examples, examples_targets):
                    if isinstance(self.projection, Projection):
                        # get weights in the input space
                        weights.append(self.projection.get_input_weights(ex, ex_targ))
                    else:
                        raise AttributeError(
                            "Cannot extract weights from the provided projection function"
                            + "Either remove 'weights' from the `case_returns` or"
                            + "inherit from `Projection` and overwrite `get_input_weights`."
                        )

                return_dict["weights"] = tf.stack(weights, axis=0)

                # optimization test TODO
                # return_dict["weights"] = tf.vectorized_map(
                #     fn=lambda x: self.projection.get_input_weights(x[0], x[1]),
                #     elems=(examples, examples_targets),
                #     # fn_output_signature=tf.float32,
                # )

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

        # return a dict only different variables are returned
        if len(return_dict) == 1:
            return list(return_dict.values())[0]
        return return_dict
