"""
Module related to Case Base Explainer
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from ..attributions.base import BlackBoxExplainer, sanitize_input_output
from ..attributions import Saliency
from ..types import Callable, List, Optional, Union, Type

from .base import NaturalExampleBasedExplainer
from .projections import Projection, AttributionProjection
from .search_methods import BaseSearchMethod, SklearnKNN


class Cole(NaturalExampleBasedExplainer):
    """
    ...
    """
    def __init__(self,
                 case_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 dataset_targets: Union[tf.Tensor, np.ndarray] = None,
                 search_method: Type[BaseSearchMethod] = SklearnKNN,
                 k: int = 1,
                 distance: Union[str, Callable] = "euclidean",
                 projection: Optional[Union[Projection, Callable]] = None,
                 returns: Optional[Union[List[str], str]] = "examples",
                 model: Optional[Callable] = None,
                 latent_layer: Optional[Union[str, int]] = None,
                 attribution_method: Type[BlackBoxExplainer] = Saliency,
                 **attribution_kwargs,
                 ):
        # set attributes
        if isinstance(case_dataset, tuple):
            # assuming (x_train, y_train)
            self.case_dataset = case_dataset[0]
            self.dataset_targets = case_dataset[1]
        else:
            self.case_dataset = case_dataset
            self.dataset_targets = dataset_targets
        self.k = k
        self.set_returns(returns)

        # set projection
        if projection is not None:
            self.projection = projection
        else:
            if model is None:
                raise ValueError("The Cole method use attribution projection,"+\
                                 "either provide a projection or a model and an attribution method")
            else:
                self.projection = AttributionProjection(model=model, method=attribution_method,
                                                        latent_layer=latent_layer,
                                                        **attribution_kwargs)
        
        # project dataset
        projected_dataset = self.projection(self.case_dataset, self.dataset_targets)

        # initiate search_method
        self.search_method = search_method(search_set=projected_dataset, k=k, distance=distance,
                                           algorithm='auto', returns=["indices", "distances"])

    def explain(self,
                inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                targets: Optional[Union[tf.Tensor, np.ndarray]] = None):
        """
        ...
        """
        # project inputs
        projected_inputs = self.projection(inputs, targets)

        # look for closest elements to projected inputs
        search_output = self.search_method(projected_inputs)

        # manage returned elements
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
                    weights.append(self.projection.get_input_weights(ex, ex_targ))
                    print(weights[-1])
                return_dict["weights"] = tf.stack(weights, axis=0)

                # optimization test
                # return_dict["weights"] = tf.vectorized_map(
                #     fn=lambda x: self.projection.get_input_weights(x[0], x[1]),
                #     elems=(examples, examples_targets),
                #     # fn_output_signature=tf.float32,
                # )
        
        # add indices and distances
        if "distances" in self.returns:
            return_dict["distances"] = search_output["distances"]
        if "indices" in self.returns:
            return_dict["indices"] = search_output["indices"]
        
        # Return a dict only different variables are returned
        if len(return_dict) == 1:
            return list(return_dict.values())[0]
        else:
            return return_dict