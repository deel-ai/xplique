"""
Tests that BlackBox attribution model can indeed handle
several model type with a native User eXperience
"""

import numpy as np
import tensorflow as tf

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from xplique.attributions import (Occlusion, Rise, Lime, KernelShap)
from xplique.commons.tf_operations import predictions_one_hot, batch_predictions_one_hot
from xplique.commons.callable_operations import predictions_one_hot_callable, \
    batch_predictions_one_hot_callable

from ..utils import generate_data, generate_model, generate_regression_model

def _default_methods_tabular(model):
    return [
        Occlusion(model, patch_size = 1, patch_stride = 1),
        Lime(model),
        KernelShap(model)
    ]

def _default_methods_images(model):
    return [
        Rise(model),
        Occlusion(model),
        Lime(model, map_to_interpret_space=_map_to_interpret_space),
        KernelShap(model, map_to_interpret_space=_map_to_interpret_space)
    ]

def _map_to_interpret_space(inp):
    width = inp.shape[0]
    height = inp.shape[1]

    mapping = tf.range(width*height)
    mapping = tf.reshape(mapping, (width, height))
    mapping = tf.cast(mapping, tf.int32)

    return mapping

class DenseModule(tf.Module):
    """
    A generic tf.Module Dense layer
    """
    def __init__(self, input_size, nb_labels):
        super(DenseModule, self).__init__()
        self.weights = tf.Variable(
            tf.random.normal([input_size, nb_labels]), name='w')
        self.bias = tf.Variable(tf.zeros([nb_labels]), name='b')
    def __call__(self, x):
        output = tf.matmul(x, self.weights) + self.bias
        return tf.nn.relu(output)

class ImagesModule(tf.Module):
    """
    A generic tf.Module with flatten and two dense layers
    """
    def __init__(self, input_shape, nb_labels):
        super(ImagesModule, self).__init__()
        self.input_size = input_shape[0]*input_shape[1]*input_shape[2]
        self.dense1 = DenseModule(self.input_size, 64)
        self.dense2 = DenseModule(64, nb_labels)
    def __call__(self, x):
        x = tf.reshape(x, [x.shape[0],-1])
        x = self.dense1(x)
        output = self.dense2(x)
        return output

def _default_tf_model_tabular(input_size, nb_labels):
    tf_keras_model = generate_regression_model((input_size,), nb_labels)
    tf_keras_layer = tf.keras.layers.Dense(nb_labels)
    tf_keras_layer(tf.random.normal([32, input_size]))
    tf_module = DenseModule(input_size, nb_labels)

    return [
        tf_keras_model,
        tf_keras_layer,
        tf_module
    ]

def _default_tf_model_images(input_shape, nb_labels):

    tf_keras_model = generate_model(input_shape, nb_labels)

    # create a mode through tf functionnal api
    functional_inputs = tf.keras.Input(shape = input_shape)
    functional_flatten = tf.keras.layers.Flatten()
    functional_x = functional_flatten(functional_inputs)
    functional_x = tf.keras.layers.Dense(64, activation="relu")(functional_x)
    functional_outputs = tf.keras.layers.Dense(nb_labels)(functional_x)
    tf_functional_model = tf.keras.Model(inputs=functional_inputs, outputs=functional_outputs)

    tf_module = ImagesModule(input_shape, nb_labels)

    return [
        tf_keras_model,
        tf_functional_model,
        tf_module
    ]

def _default_callable_tabular(input_size, nb_labels):
    sk_svc = SVC(probability=True)
    sk_rf = RandomForestClassifier()
    sk_api = RandomSklearn(input_size, nb_labels)
    np_callable = RandomNpCallable(input_size, nb_labels)

    return [
        sk_svc,
        sk_rf,
        sk_api,
        np_callable
    ]

def _default_callable_images(input_shape, nb_labels):
    sk_api = RandomSklearn(input_shape, nb_labels)
    np_callable = RandomNpCallable(input_shape, nb_labels)

    return [
        sk_api,
        np_callable
    ]

class RandomSklearn():
    """
    A class with a fit and predict_proba attributes such as the one
    in sklearn api, which returns random probabilities
    """
    def __init__(self, input_shape, nb_labels):
        self.input_shape = input_shape
        self.nb_labels = nb_labels
    def fit(self, inputs, targets):
        pass
    def predict_proba(self, inputs):
        return np.random.random((inputs.shape[0], self.nb_labels))

class RandomNpCallable():
    """
    A class with a call defines on numpy array
    """
    def __init__(self, input_shape, nb_labels):
        self.input_shape = input_shape
        self.nb_labels = nb_labels
    def fit(self, inputs, targets):
        pass
    def __call__(self, inputs):
        return np.random.random((inputs.shape[0], self.nb_labels))  

def test_tf_models_tabular():
    """
    Test if on tabular data, BlackBox methods work as expected with tf family of callable,
    at least the most commons.
    """
    input_size = 32
    nb_labels = 10
    samples = 10

    inputs, targets = generate_data((input_size,), nb_labels, samples)
    models = _default_tf_model_tabular(input_size, nb_labels)

    for model in models:
        explainers = _default_methods_tabular(model)

        for explainer in explainers:
            assert explainer.inference_function is predictions_one_hot
            assert explainer.batch_inference_function is batch_predictions_one_hot

            explanations = explainer(inputs, targets)

            assert explanations.shape == (samples, input_size)

def test_tf_models_images():
    """
    Test if on images data, BlackBox methods work as expected with tf family of callable,
    at least the most commons.
    """
    input_shapes = [(28, 28, 1), (32, 32, 3)]
    nb_labels = 10
    samples = 10

    for input_shape in input_shapes:
        inputs, targets = generate_data(input_shape, nb_labels, samples)
        models = _default_tf_model_images(input_shape, nb_labels)
        for model in models:

            explainers = _default_methods_images(model)

            for explainer in explainers:
                assert explainer.inference_function is predictions_one_hot
                assert explainer.batch_inference_function is batch_predictions_one_hot

                explanations = explainer(inputs, targets)

                assert explanations.shape == (samples, *input_shape[:2])

def test_callable_models_tabular():
    """
    Test if on tabular data we can handle sklearn models, sklearn like models (with
    predict_proba attributes) and callable which requires a np.ndarray input
    """
    input_size = 32
    nb_labels = 10
    samples = 10

    inputs, targets = generate_data((input_size,), nb_labels, samples)
    models = _default_callable_tabular(input_size, nb_labels)

    for model in models:
        no_one_hot_targets = np.arange(0,10)

        model.fit(inputs, no_one_hot_targets)
        explainers = _default_methods_tabular(model)

        for explainer in explainers:
            assert explainer.inference_function is predictions_one_hot_callable
            assert explainer.batch_inference_function is batch_predictions_one_hot_callable

            explanations = explainer(inputs, targets)

            assert explanations.shape == (samples, input_size)

def test_callable_models_images():
    """
    Test if on images data we can handle sklearn like models (with predict_proba attributes)
    and callable which requires a np.ndarray input
    """
    input_shapes = [(28, 28, 1), (32, 32, 3)]
    nb_labels = 10
    samples = 10

    for input_shape in input_shapes:
        inputs, targets = generate_data(input_shape, nb_labels, samples)
        models = _default_callable_images(input_shape, nb_labels)
        for model in models:

            explainers = _default_methods_images(model)

            for explainer in explainers:
                assert explainer.inference_function is predictions_one_hot_callable
                assert explainer.batch_inference_function is batch_predictions_one_hot_callable

                explanations = explainer(inputs, targets)

                assert explanations.shape == (samples, *input_shape[:2])  
