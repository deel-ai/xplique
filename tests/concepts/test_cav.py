import numpy as np
import tensorflow as tf
from sklearn.linear_model import SGDClassifier

from xplique.concepts import Cav
from ..utils import generate_data, generate_model


def test_shape():
    """Ensure the output shape is correct"""

    input_shapes = [(8, 8, 3), (8, 8, 1)]
    nb_labels = 2
    nb_samples = 100

    for input_shape in input_shapes:
        x, y = generate_data(input_shape, nb_labels, nb_samples)
        model = generate_model(input_shape, nb_labels)

        for layer in model.layers[1:-1]:
            output_shape = layer.output.shape
            cav = Cav(model, layer.name)(x, x)

            # cav should have the same shape as output, except for the batch size
            assert cav.shape == output_shape[1:]


def test_classifier():
    """For a given simple example, ensure that the differents classifiers works"""
    x = np.random.random((5000, 2))
    y = np.array(x[:, 0]**2 + x[:, 1]**2 > 1.0, np.float32) # y = (x1^2 + x2^2) > 1 (not linearly separable)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(2),
        tf.keras.layers.Lambda(lambda x: x ** 2, name="square_layer"),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.1))
    model.fit(x, y, batch_size=32, epochs=1)

    # the second layer (x -> x^2) make the problem linearly separable
    # the cav vector should be [1, 1]
    positive = []
    negative = []
    for i in range(len(y)):
        if y[i]:
            positive.append(x[i])
        else:
            negative.append(x[i])
    positive = np.array(positive)
    negative = np.array(negative)

    cavs = [
        Cav(model, "square_layer", classifier="SGD")(positive, negative),
        Cav(model, "square_layer", classifier="SVC")(positive, negative),
        Cav(model, "square_layer", classifier=SGDClassifier(alpha=0.1))(positive, negative)
    ]

    # the perfect cav vector should be [0.5, 0.5]
    for cav in cavs:
        assert np.sum(np.abs(cav - [0.5, 0.5])) < 0.1
