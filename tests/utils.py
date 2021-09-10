import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Activation, Dropout, Flatten, MaxPooling2D, Input
from tensorflow.keras.utils import to_categorical

def generate_data(x_shape=(32, 32, 3), num_labels=10, samples=100):
    x = np.random.rand(samples, *x_shape).astype(np.float32)
    y = to_categorical(np.random.randint(0, num_labels, samples), num_labels)

    return x, y

def generate_model(input_shape=(32, 32, 3), output_shape=10):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(4, kernel_size=(2, 2),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(output_shape))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd')

    return model

def generate_regression_model(features_shape, output_shape=1):
    model = Sequential()
    model.add(Input(shape=features_shape))
    model.add(Flatten())
    model.add(Dense(4, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(output_shape))
    model.compile(loss='mean_absolute_error',
                  optimizer='sgd')

    return model

def almost_equal(arr1, arr2, epsilon=1e-6):
    """Ensure two array are almost equal at an epsilon"""
    return np.sum(np.abs(arr1 - arr2)) < epsilon
