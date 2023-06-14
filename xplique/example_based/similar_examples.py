"""
Base model for example-based 
"""

from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np

from ..types import Callable, Dict, List, Optional, Tuple, Type, Union

from ..commons import sanitize_inputs_targets
from .search_methods import SklearnKNN
from .search_methods.base import BaseSearchMethod
from .projections.base import Projection


class SimilarExamples():
    """
    Base class for natural example-base methods explaining models,
    they project the case_dataset into a pertinent space for the with a `Projection`,
    then they call the `BaseSearchMethod` on it.

    Parameters
    ----------
    case_dataset
        The dataset used to train the model, examples are extracted from the dataset.
    dataset_labels
        Labels associated to the examples in the dataset. Indices should match with case_dataset.
    dataset_targets
        Targets associated to the case_dataset for dataset projection. See `projection` for detail.
    search_method
        An algorithm to search the examples in the projected space.
    k
        The number of examples to retrieve.
    projection
        Projection or Callable that project samples from the input space to the search space.
        The search space sould be a space where distance make sense for the model.
        It should not be `None`, otherwise,
        all examples could be computed only with the `search_method`.

        Example of Callable:
        ```
        def custom_projection(inputs: tf.Tensor, np.ndarray, targets: tf.Tensor, np.ndarray = None):
            '''
            Example of projection,
            inputs are the elements to project.
            targets are optionnal parameters to orientated the projection.
            '''
            projected_inputs = # do some magic on inputs, it should use the model.
            return projected_inputs
        ```
    case_returns
        String or list of string with the elements to return in `self.explain()`.
        See `self.set_returns()` for detail.
    search_method_kwargs
        Parameters to be passed at the construction of the `search_method`.
    """

    def __init__(self,
                 case_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 labels_dataset: Union[tf.Tensor, np.ndarray] = None,
                 dataset_targets: Union[tf.Tensor, np.ndarray] = None,
                 search_method: Type[BaseSearchMethod] = SklearnKNN,
                 k: int = 1,
                 projection: Union[Projection, Callable] = None,
                 case_returns: Union[List[str], str] = "examples",
                 **search_method_kwargs):
        assert projection is not None,\
            "`SimilarExamples` without `projection` is a `BaseSearchMethod`."

        # set attributes
        if isinstance(case_dataset, tuple) and labels_dataset is None:
            # assuming (x_train, y_train)
            self.case_dataset = case_dataset[0]
            self.labels_dataset = case_dataset[1]
        else:
            self.case_dataset = case_dataset
            self.labels_dataset = labels_dataset
        self.dataset_targets = dataset_targets
        self.k = k
        self.set_returns(case_returns)
        self.projection = projection

        # project dataset
        projected_dataset = self.projection(self.case_dataset, self.dataset_targets)

        # set `search_returns` if not provided and overwrite it otherwise
        search_method_kwargs["search_returns"] = ["indices", "distances"]
                

        # initiate search_method
        self.search_method = search_method(search_set=projected_dataset, 
                                           k=k, **search_method_kwargs)

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
        if returns is None:
            self.returns = ["examples"]
        elif isinstance(returns, str):
            possibilities = ["examples", "weights", "distances", "labels", "include_inputs"]
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
        
    @sanitize_inputs_targets
    def explain(self,
                inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                targets: Optional[Union[tf.Tensor, np.ndarray]] = None):
        """
        Compute examples to explain the inputs.
        It project inputs with `self.projection` in the search space
        and find examples with `self.search_method`.
        
        inputs
            Dataset, Tensor or Array. Input samples to be explained.
            If Dataset, targets should not be provided (included in Dataset).
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
            Dictionnary with listed elements in `self.returns`.
            If only one element is present it returns the element.
            The elements that can be returned are:
            examples, weights, distances, indices, and labels.
        """
        # project inputs
        projected_inputs = self.projection(inputs, targets)

        # look for closest elements to projected inputs
        search_output = self.search_method(projected_inputs)

        # manage returned elements
        return self.format_search_output(search_output, inputs, targets)

    def __call__(self,
                 inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 targets: Optional[Union[tf.Tensor, np.ndarray]] = None):
        """explain alias"""
        return self.explain(inputs, targets)
    
    def format_search_output(self,
                             search_output: Dict[str, tf.Tensor],
                             inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                             targets: Optional[Union[tf.Tensor, np.ndarray]] = None):
        """
        Format the output of the `search_method` to match the expected returns in `self.returns`.
        
        Parameters
        ----------
        search_output
            Dictionnary with the required outputs from the `search_method`. 
        inputs
            Dataset, Tensor or Array. Input samples to be explained.
            If Dataset, targets should not be provided (included in Dataset).
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
            Dictionnary with listed elements in `self.returns`.
            If only one element is present it returns the element.
            The elements that can be returned are:
            examples, weights, distances, indices, and labels.
        """
        return_dict = {}

        # add examples and weights
        if "examples" or "weights" in self.returns:
            # get examples from indices
            examples = tf.gather(self.case_dataset, search_output["indices"])
            if targets is not None:
                examples_targets = tf.gather(self.dataset_targets, search_output["indices"])
            if "include_inputs" in self.returns:
                # include inputs
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
                            "Cannot extract weights from the provided projection function"+\
                            "Either remove 'weights' from the `case_returns` or"+\
                            "inherit from `Projection` and overwrite `get_input_weights`.")
                    
                return_dict["weights"] = tf.stack(weights, axis=0)

                # optimization test
                # return_dict["weights"] = tf.vectorized_map(
                #     fn=lambda x: self.projection.get_input_weights(x[0], x[1]),
                #     elems=(examples, examples_targets),
                #     # fn_output_signature=tf.float32,
                # )
        
        # add indices, distances, and labels
        if "distances" in self.returns:
            return_dict["distances"] = search_output["distances"]
        if "labels" in self.returns:
            assert self.labels_dataset is not None, "Cole method cannot return labels without a label dataset."
            return_dict["labels"] = tf.gather(self.labels_dataset, search_output["indices"])
        
        # return a dict only different variables are returned
        if len(return_dict) == 1:
            return list(return_dict.values())[0]
        else:
            return return_dict