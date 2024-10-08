import signal, time

import numpy as np
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Conv1D, Conv2D, Activation, GlobalAveragePooling1D,
                                     Dropout, Flatten, MaxPooling2D, Input, Reshape)
from tensorflow.keras.utils import to_categorical
import requests

def generate_data(x_shape=(32, 32, 3), num_labels=10, samples=100):
    x = np.random.rand(samples, *x_shape).astype(np.float32)
    y = to_categorical(np.random.randint(0, num_labels, samples), num_labels)

    return x, y

def generate_model(input_shape=(32, 32, 3), output_shape=10):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(4, kernel_size=(2, 2),
                     activation='relu', name='conv2d'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(output_shape))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd')

    return model

def generate_agnostic_model(input_shape=(3,), nb_labels=3):
    model = Sequential()
    model.add(Input(input_shape))
    model.add(Flatten())
    model.add(Dense(nb_labels))

    return model

def generate_timeseries_model(input_shape=(20, 10), output_shape=10):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(4, kernel_size=3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(output_shape))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd',
                  metrics=['accuracy'])

    return model

def generate_regression_model(features_shape, output_shape=1):
    model = Sequential()
    model.add(Input(shape=features_shape))
    model.add(Flatten())
    model.add(Dense(4, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(output_shape))
    model.compile(loss='mean_absolute_error', optimizer='sgd',
                  metrics=['accuracy'])

    return model

def almost_equal(arr1, arr2, epsilon=1e-6):
    """Ensure two array are almost equal at an epsilon"""
    return np.sum(np.abs(arr1 - arr2)) < epsilon


def generate_dataset(nb_samples: int = 100):
    np.random.seed(0)

    # dataset parameters
    features_coef = np.array([1, -2, -3, 4, 0, 0, 0, 0])
    nb_features = len(features_coef)

    # create dataset
    dataset = np.random.normal(4, 1, (nb_samples, nb_features))
    noise = np.random.normal(0, 1, nb_samples)
    target = dataset.dot(features_coef) + noise

    return dataset, target, features_coef


def generate_linear_model(
        features_coef,
        library: str="keras"
):
    if library == "sklearn":
        sk_linear = LinearRegression()

        # fit is necessary for the model to use predict
        data = np.ones((len(features_coef),)).reshape(1, -1)
        target = np.array([0])
        sk_linear.fit(data, target)
        # but it does not impact the coefficients
        sk_linear.coef_ = features_coef

        # Create the wrapper class
        class Wrapper:
            # The init method is necessary for every class
            def __init__(self, model):
                self.model = model

            # The call method calls the predict method
            def __call__(self, inputs):
                pred = self.model.predict(inputs)
                return pred

        sk_model = Wrapper(sk_linear)
        return sk_model

    elif library == "keras":

        inputs = Input(shape=(len(features_coef),))

        # create one dense layer with the weights
        weights = np.expand_dims(np.array(features_coef), axis=1)
        bias = np.array([0])
        output = tf.keras.layers.Dense(
            1, weights=[weights, bias]
        )(inputs)

        # create and compile model
        tf_model = Model(inputs, output)
        tf_model.compile(optimizer="adam", loss="mae")

        return tf_model

def generate_object_detection_model(input_shape=(32, 32, 3), max_nb_boxes=10, nb_labels=5, with_nmf=False):
    # create a model that generates max_nb_boxes and select some randomly
    output_shape = (max_nb_boxes, 5 + nb_labels)
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(4, kernel_size=(2, 2),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(np.prod(output_shape)))
    model.add(Reshape(output_shape))
    model.add(Activation('sigmoid'))
    model.compile(loss='mae', optimizer='sgd')

    # ensure iou computation will work
    def make_plausible_boxes(model_output):
        coordinates = tf.sort(model_output[:, :, :4], axis=-1) * 200
        probabilities = model_output[:, :, 4][:, :, tf.newaxis]
        classifications = tf.nn.softmax(model_output[:, :, 5:], axis=-1)
        new_output = tf.concat([coordinates, probabilities, classifications], axis=-1)
        return new_output
    
    valid_model = lambda inputs: make_plausible_boxes(model(inputs))

    # equivalent of nms
    def randomly_select_boxes(boxes):
        boxes_ids = tf.range(tf.shape(boxes)[0])
        nb_boxes = tf.experimental.numpy.random.randint(1, max_nb_boxes)
        boxes_ids = tf.random.shuffle(boxes_ids)[:nb_boxes]
        return tf.gather(boxes, boxes_ids)

    # model with nms
    def model_with_random_nb_boxes(input):
        all_boxes = valid_model(input)
        some_boxes = [randomly_select_boxes(boxes) for boxes in all_boxes]
        return some_boxes
    
    if with_nmf:
        return model_with_random_nb_boxes
    return valid_model

def download_file(identifier: str,
                  destination: str):
    """
    Helper to download a binary file from Google Drive.

    Parameters
    ----------
    identifier
        The file id of the document to download. It follows this
        naming: https://drive.google.com/file/d/<file_id>/
    destination
        The path to save the file locally.
    """
    googledoc_url = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(googledoc_url, params = {'id' : identifier}, stream = True)

    # Save the response contents locally
    with open(destination, "wb") as file:
        chunk_size = 32768
        for chunk in response.iter_content(chunk_size):
            if chunk:
                file.write(chunk)

def get_gaussian_data(nb_classes=3, nb_samples_class=20, n_dims=1):
    tf.random.set_seed(42)

    sigma = 1
    mu = [10 * (id + 1) for id in range(nb_classes)]

    X = tf.concat([
        tf.random.normal(shape=(nb_samples_class, n_dims), mean=mu[i], stddev=sigma, dtype=tf.float32)
        for i in range(nb_classes)
    ], axis=0)

    y = tf.concat([
        tf.ones(shape=(nb_samples_class), dtype=tf.int32) * i
        for i in range(nb_classes)
    ], axis=0)

    return(X, y)
