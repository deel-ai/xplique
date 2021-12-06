# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import numpy as np
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


def generate_data(x_shape=(32, 32, 3), num_labels=10, samples=100):
    x = np.random.rand(samples, *x_shape).astype(np.float32)
    y = to_categorical(np.random.randint(0, num_labels, samples), num_labels)

    return x, y


def generate_model(input_shape=(32, 32, 3), output_shape=10):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(4, kernel_size=(2, 2), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(output_shape))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="sgd")

    return model


def generate_timeseries_model(input_shape=(20, 10), output_shape=10):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(4, kernel_size=3, activation="relu"))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(output_shape))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="sgd")

    return model


def generate_regression_model(features_shape, output_shape=1):
    model = Sequential()
    model.add(Input(shape=features_shape))
    model.add(Flatten())
    model.add(Dense(4, activation="relu"))
    model.add(Dense(4, activation="relu"))
    model.add(Dense(output_shape))
    model.compile(loss="mean_absolute_error", optimizer="sgd")

    return model


def almost_equal(arr1, arr2, epsilon=1e-6):
    """Ensure two array are almost equal at an epsilon"""
    return np.sum(np.abs(arr1 - arr2)) < epsilon
