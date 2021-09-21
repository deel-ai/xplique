import tensorflow as tf
import numpy as np

from xplique.features_visualizations import Objective
from ..utils import almost_equal


def dummy_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input((28, 28, 3)),
        tf.keras.layers.Conv2D(16, (3, 3)),
        tf.keras.layers.Conv2D(16, (3, 3), name="early"),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Conv2D(16, (3, 3)),
        tf.keras.layers.Conv2D(16, (3, 3), name="features"),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Flatten(name="pre-logits"),
        tf.keras.layers.Dense(10, name="logits")
    ])
    model.compile()
    return model


def test_layer():
    """ Ensure we can target a layer """
    model = dummy_model()

    objectif = Objective.layer(model, -1)

    mask = np.ones(model.get_layer("logits").output.shape[1:])
    assert almost_equal(objectif.masks[0], mask)

    model_reconf, objective_func, names, input_shape = objectif.compile()

    assert model_reconf.outputs[0].name == model.layers[-1].output.name
    assert names[0] == "Layer#logits"
    assert input_shape == (1, 28, 28, 3)


def test_direction():
    """ Ensure we can target a direction """
    model = dummy_model()

    direction_vec = np.random.random(model.get_layer("early").output.shape[1:])
    objectif = Objective.direction(model, "early", direction_vec)

    assert almost_equal(objectif.masks[0], direction_vec)

    model_reconf, objective_func, names, input_shape = objectif.compile()

    assert model_reconf.outputs[0].name == model.layers[1].output.name
    assert names[0] == "Direction#early_0"
    assert input_shape == (1, 28, 28, 3)


def test_channel():
    """ Ensure we can target a channel """
    model = dummy_model()
    nb_channels = 10 # number of channel to optim (1...n)

    objectif = Objective.channel(model, "early", list(range(nb_channels)))

    # the first mask should target the first channel
    mask_1 = np.zeros(model.layers[1].output.shape[1:])
    mask_1[:, :, 0] = 1.0
    assert almost_equal(objectif.masks[0][0], mask_1)

    model_reconf, objective_func, names, input_shape = objectif.compile()

    # ensure we are targeting the good layer (target by id should be equal to target by name)
    assert model_reconf.outputs[0].name == model.layers[1].output.name
    assert all([names[i] == f"Channel#early_{i}" for i in range(nb_channels)])
    # we should have one input for each optim (one for each of the channels
    assert input_shape == (10, 28, 28, 3)


def test_neurons():
    """ Ensure we can target neurons """
    model = dummy_model()

    objectif = Objective.neuron(model, "early", list(range(20)))

    masks = np.zeros((20) + model.get_layer("early").output.shape[1:])
    for i in range(20):
        masks[i, 0, i // masks.shape[3], i % masks.shape[3]] = 1.0

    assert almost_equal(objectif.masks[0], masks)

    model_reconf, objective_func, names, input_shape = objectif.compile()

    assert model_reconf.outputs[0].name == model.layers[1].output.name
    assert all([names[i] == f"Neuron#early_{i}" for i in range(20)])
    assert input_shape == (20, 28, 28, 3)


def test_combinations():
    """ Ensure the combinations of the objective are correct """
    model = dummy_model()

    layer_obj = Objective.layer(model, -1)

    direction_vec = np.random.random(model.get_layer("pre-logits").output.shape[1:])
    direction_obj = Objective.direction(model, "pre-logits", direction_vec)

    nb_channels = 5
    channels_obj = Objective.channel(model, "early", list(range(nb_channels)))
    mask_channel_0 = np.zeros(model.get_layer("early").output.shape[1:])
    mask_channel_0[:, :, 0] = 1.0

    nb_neurons = 3
    neurons_obj = Objective.neuron(model, "features", list(range(nb_neurons)))
    mask_neurons = np.zeros((nb_neurons) + model.get_layer("features").output.shape[1:])
    for i in range(nb_neurons):
        mask_neurons[i, 0, i // mask_neurons.shape[3], i % mask_neurons.shape[3]] = 1.0

    combined_obj = layer_obj + direction_obj + channels_obj + neurons_obj

    # the generation of masks must remain similar to the operation on singular objectives
    assert len(combined_obj.masks) == 4
    assert almost_equal(combined_obj.masks[0][0], np.ones(model.layers[-1].output.shape[1:]))
    assert almost_equal(combined_obj.masks[1][0], direction_vec)
    assert almost_equal(combined_obj.masks[2][0], mask_channel_0)
    assert almost_equal(combined_obj.masks[3], mask_neurons)

    model_reconf, objective_func, names, input_shape = combined_obj.compile()

    assert [o.name for o in model_reconf.outputs] == [model.layers[-1].output.name, # layer obj
                                                      model.layers[-2].output.name, # direction obj
                                                      model.layers[1].output.name,  # channel obj
                                                      model.layers[-4].output.name] # neuron obj
    assert_names = [
        f"Layer#logits & Direction#pre-logits_0 & Channel#early_{c_id} & Neuron#features_{n_id}" for
        c_id in range(nb_channels) for n_id in range(nb_neurons)]
    assert all([names[i] == assert_names[i] for i in range(nb_neurons*nb_channels)])
    assert input_shape == (nb_neurons * nb_channels, 28, 28, 3) # number of combinations
