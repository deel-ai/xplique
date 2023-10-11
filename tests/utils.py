import signal, time

import numpy as np
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Conv1D, Conv2D, Activation, GlobalAveragePooling1D,
                                     Dropout, Flatten, MaxPooling2D, Input, Reshape)
from tensorflow.keras.utils import to_categorical
from PIL import Image, ImageDraw, ImageFont
import urllib.request
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

def generate_txt_images_data(x_shape=(32, 32, 3), num_labels=10, samples=100):
    """
    Generate an image dataset composed of white texts over black background.
    The texts are words of 3 successive letters, the number of classes is set by the
    parameter num_labels. The location of the text in the image is cycling over the
    image dimensions.
    Ex: with num_labels=3, the 3 classes will be 'ABC', 'BCD' and 'CDE'.

    """
    all_labels_str = "".join([chr(lab_idx) for lab_idx in range(65, 65+num_labels+2)])  # ABCDEF
    labels_str = [all_labels_str[i:i+3] for i in range(len(all_labels_str) - 2)]        # ['ABC', 'BCD', 'CDE', 'DEF']

    def create_image_from_txt(image_shape, txt, offset_x, offset_y):
        # Get a Pillow font (OS independant)
        try:
            fnt = ImageFont.truetype("FreeMono.ttf", 16)
        except OSError:
            # dl the font it is it not in the system
            url = "https://github.com/python-pillow/Pillow/raw/main/Tests/fonts/FreeMono.ttf"
            urllib.request.urlretrieve(url, "tests/FreeMono.ttf")
            fnt = ImageFont.truetype("tests/FreeMono.ttf", 16)

        # Make a black image and draw the input text in white at the location offset_x, offset_y
        rgb = (len(image_shape) == 3 and image_shape[2] > 1)
        if rgb:
            image = Image.new("RGB", (image_shape[0], image_shape[1]), (0, 0, 0))
        else:
            # grayscale
            image = Image.new("L", (image_shape[0], image_shape[1]), 0)
        d = ImageDraw.Draw(image)
        d.text((offset_x, offset_y), txt, font=fnt, fill='white')
        return image

    x = np.empty((samples, *x_shape)).astype(np.float32)
    y = np.empty(samples)

    # Iterate over the samples and generate images of labels shifted by increasing offsets
    offset_x_max = x_shape[0] - 25
    offset_y_max = x_shape[1] - 10

    current_label_id = 0
    offset_x = offset_y = 0
    for i in range(samples):
        image = create_image_from_txt(x_shape, txt=labels_str[current_label_id], offset_x=offset_x, offset_y=offset_y)
        image = np.reshape(image, x_shape)
        x[i] = np.array(image).astype(np.float32)/255.0
        y[i] = current_label_id

        # cycle labels
        current_label_id = (current_label_id + 1) % num_labels
        offset_x = (offset_x + 1) % offset_x_max
        offset_y = ((i+2) % offset_y_max)
        if offset_y > offset_y_max:
            break
    x = x[0:i]
    y = y[0:i]
    return x, to_categorical(y, num_labels), i, labels_str

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
