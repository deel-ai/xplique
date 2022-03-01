import numpy as np
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, Activation, GlobalAveragePooling1D, Dropout, Flatten, MaxPooling2D, Input
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

