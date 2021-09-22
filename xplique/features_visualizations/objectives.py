"""
Objective wrapper and utils to build a function to be optimized
"""

import itertools

import tensorflow as tf
import numpy as np

from ..commons import find_layer
from ..types import Union, List, Callable, Tuple, Optional
from .losses import dot_cossim


class Objective:
    """
    Use to combine several sub-objectives into one.

    Each sub-objective act on a layer, possibly on a neuron or a channel (in
    that case we apply a mask on the layer values), or even multiple neurons (in
    that case we have multiples masks). When two sub-objectives are added, we
    optimize all their combinations.

    e.g Objective 1 target the neurons 1 to 10 of the logits l1,...,l10
        Objective 2 target a direction on the first layer d1
        Objective 3 target each of the 5 channels on another layer c1,...,c5

        The resulting Objective will have 10*5*1 combinations. The first input
        will optimize l1+d1+c1 and the last one l10+d1+c5.

    Parameters
    ----------
    model
        Model used for optimization.
    layers
        A list of the layers output for each sub-objectives.
    masks
        A list of masks that will be applied on the targeted layer for each
        sub-objectives.
    funcs
        A list of loss functions for each sub-objectives.
    multipliers
        A list of multiplication factor for each sub-objectives
    names
        A list of name for each sub-objectives
    """

    def __init__(self,
                 model: tf.keras.Model,
                 layers: List[tf.keras.layers.Layer],
                 masks: List[tf.Tensor],
                 funcs: List[Callable],
                 multipliers: List[float],
                 names: List[str]):
        self.model = model
        self.layers = layers
        self.masks = masks
        self.funcs = funcs
        self.multipliers = multipliers
        self.names = names

    def __add__(self, term):
        if not isinstance(term, Objective):
            raise ValueError(f"{term} is not an objective.")
        return Objective(
                self.model,
                layers=self.layers + term.layers,
                masks=self.masks + term.masks,
                funcs=self.funcs + term.funcs,
                multipliers=self.multipliers + term.multipliers,
                names=self.names + term.names
        )

    def __sub__(self, term):
        if not isinstance(term, Objective):
            raise ValueError(f"{term} is not an objective.")
        term.multipliers = [-1.0 * m for m in term.multipliers]
        return self + term

    def __mul__(self, factor: float):
        if not isinstance(factor, (int, float)):
            raise ValueError(f"{factor} is not a number.")
        self.multipliers = [m * factor for m in self.multipliers]
        return self

    def __rmul__(self, factor: float):
        return self * factor

    def compile(self) -> Tuple[tf.keras.Model, Callable, List[str], Tuple]:
        """
        Compile all the sub-objectives into one and return the objects
        for the optimisation process.

        Returns
        -------
        model_reconfigured
            Model with the outputs needed for the optimization.
        objective_function
            Function to call that compute the loss for the objectives.
        names
            Names of each objectives.
        input_shape
            Shape of the input, one sample for each optimization.
        """
        # the number of inputs will be the number of combinations possible
        # of the objectives, the mask are used to take into account
        # these combinations
        nb_sub_objectives = len(self.multipliers)

        # re-arrange to match the different objectives with the model outputs
        masks = np.array([np.array(m) for m in itertools.product(*self.masks)])
        masks = [tf.cast(tf.stack(masks[:, i]), tf.float32) for i in
                 range(nb_sub_objectives)]

        # the name of each combination is the concatenation of each objectives
        names = np.array([' & '.join(names) for names in
                           itertools.product(*self.names)])
        # one multiplier by sub-objective
        multipliers = tf.constant(self.multipliers)

        def objective_function(model_outputs):
            loss = 0.0
            for output_index in range(0, nb_sub_objectives):
                loss += self.funcs[output_index](model_outputs[output_index],
                                                 masks[output_index]) * \
                                                 multipliers[output_index]
            return loss

        # the model outputs will be composed of the layers needed
        model_reconfigured = tf.keras.Model(self.model.input, [*self.layers])

        nb_combinations = masks[0].shape[0]
        input_shape = (nb_combinations, *model_reconfigured.input.shape[1:])

        return model_reconfigured, objective_function, names, input_shape

    @staticmethod
    def layer(model: tf.keras.Model,
              layer: Union[str, int],
              reducer: str = "magnitude",
              multiplier: float = 1.0,
              name: Optional[str] = None):
        """
        Util to build an objective to maximise a layer.

        Parameters
        ----------
        model
            Model used for optimization.
        layer
            Index or name of the targeted layer.
        reducer
            Type of reduction to apply, 'mean' will optimize the mean value of the
            layer, 'magnitude' will optimize the mean of the absolute values.
        multiplier
            Multiplication factor of the objective.
        name
            A name for the objective.

        Returns
        -------
        objective
            An objective ready to be compiled
        """
        layer = find_layer(model, layer)
        layer_shape = layer.output.shape
        mask = np.ones((1, *layer_shape[1:]))

        if name is None:
            name = [f"Layer#{layer.name}"]

        power = 2.0 if reducer == "magnitude" else 1.0

        def optim_func(model_output, mask):
            return tf.reduce_mean((model_output * mask) ** power)

        return Objective(model, [layer.output], [mask], [optim_func], [multiplier], [name])

    @staticmethod
    def direction(model: tf.keras.Model,
                  layer: Union[str, int],
                  vectors: Union[tf.Tensor, List[tf.Tensor]],
                  multiplier: float = 1.0,
                  cossim_pow: float = 2.0,
                  names: Optional[Union[str, List[str]]] = None):
        """
        Util to build an objective to maximise a direction of a layer.

        Parameters
        ----------
        model
            Model used for optimization.
        layer
            Index or name of the targeted layer.
        vectors
            Direction(s) to optimize.
        multiplier
            Multiplication factor of the objective.
        cossim_pow
            Power of the cosine similarity, higher value encourage the objective to care more about
            the angle of the activations.
        names
            A name for each objectives.

        Returns
        -------
        objective
            An objective ready to be compiled
        """
        layer = find_layer(model, layer)
        masks = vectors if isinstance(vectors, list) else [vectors]

        if names is None:
            names = [f"Direction#{layer.name}_{i}" for i in range(len(masks))]

        def optim_func(model_output, mask):
            return dot_cossim(model_output, mask, cossim_pow)

        return Objective(model, [layer.output], [masks], [optim_func], [multiplier], [names])

    @staticmethod
    def channel(model: tf.keras.Model,
                layer: Union[str, int],
                channel_ids: Union[int, List[int]],
                multiplier: float = 1.0,
                names: Optional[Union[str, List[str]]] = None):
        """
        Util to build an objective to maximise a channel.

        Parameters
        ----------
        model
            Model used for optimization.
        layer
            Index or name of the targeted layer.
        channel_ids
            Indexes of the channels to maximise.
        multiplier
            Multiplication factor of the objectives.
        names
            Names for each objectives.

        Returns
        -------
        objective
            An objective containing a sub-objective for each channels.
        """
        layer = find_layer(model, layer)
        layer_shape = layer.output.shape
        channel_ids = channel_ids if isinstance(channel_ids, list) else [channel_ids]

        # for each targeted channel, create a boolean mask on the layer to target
        # the channel
        masks = np.zeros((len(channel_ids), *layer_shape[1:]))
        for i, c_id in enumerate(channel_ids):
            masks[i, ..., c_id] = 1.0

        if names is None:
            names = [f"Channel#{layer.name}_{ch_id}" for ch_id in channel_ids]

        axis_to_reduce = list(range(1, len(layer_shape)))

        def optim_func(output, target):
            return tf.reduce_mean(output * target, axis=axis_to_reduce)

        return Objective(model, [layer.output], [masks], [optim_func], [multiplier], [names])

    @staticmethod
    def neuron(model: tf.keras.Model,
               layer: Union[str, int],
               neurons_ids: Union[int, List[int]],
               multiplier: float = 1.0,
               names: Optional[Union[str, List[str]]] = None):
        """
        Util to build an objective to maximise a neuron.

        Parameters
        ----------
        model
            Model used for optimization.
        layer
            Index or name of the targeted layer.
        neurons_ids
            Indexes of the neurons to maximise.
        multiplier
            Multiplication factor of the objectives.
        names
            Names for each objectives.

        Returns
        -------
        objective
            An objective containing a sub-objective for each neurons.
        """
        layer = find_layer(model, layer)

        neurons_ids = neurons_ids if isinstance(neurons_ids, list) else [neurons_ids]
        nb_objectives = len(neurons_ids)
        layer_shape = layer.output.shape[1:]

        # for each targeted neurons, create a boolean mask on the layer to target it
        masks = np.zeros((nb_objectives, *layer_shape))
        masks = masks.reshape((nb_objectives, -1))
        for i, neuron_id in enumerate(neurons_ids):
            masks[i, neuron_id] = 1.0
        masks = masks.reshape((nb_objectives, *layer_shape))

        if names is None:
            names = [f"Neuron#{layer.name}_{neuron_id}" for neuron_id in neurons_ids]

        axis_to_reduce = list(range(1, len(layer_shape)+1))

        def optim_func(output, target):
            return tf.reduce_mean(output * target, axis=axis_to_reduce)

        return Objective(model, [layer.output], [masks], [optim_func], [multiplier], [names])
