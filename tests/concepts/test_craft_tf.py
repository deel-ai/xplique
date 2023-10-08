import numpy as np
import tensorflow as tf
import random
import pytest
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Activation, Flatten, Input
from tensorflow.keras.optimizers import Adam

from xplique.concepts import CraftTf as Craft
from ..utils import generate_data, generate_model, generate_txt_images_data
from ..utils import download_file


def test_shape():
    """Ensure the output shape is correct"""

    input_shapes = [(32, 32, 3), (32, 32, 1), (64, 64, 3), (64, 32, 3)]
    nb_labels = 3
    nb_samples = 100

    for input_shape in input_shapes:
        # Generate a fake dataset
        x, y = generate_data(input_shape, nb_labels, nb_samples)
        model = generate_model(input_shape, nb_labels)
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(0.1))

        # Cut the model in two parts (as explained in the paper)
        # First part is g(.) our 'input_to_latent' model,
        # second part is h(.) our 'latent_to_logit' model
        #
        # Use 2 different set of indexes to check the behavior for different activation
        # shapes ; (0, 1) produces an activation.shape of 4 dims, (-1, -1) leads to an
        # activation.shape of 2 dims for index_layer_g, index_layer_h in [(0, 1), (-1, -1)]

        for cut_layer_name in ['conv2d']:
            cut_layer = model.get_layer(cut_layer_name)
            g = tf.keras.Model(model.inputs, cut_layer.output)
            h = tf.keras.Model(Input(tensor=cut_layer.output), model.outputs)

            # The activations must be positives
            assert np.all(g(x) >= 0.0)

            # Initialize Craft
            number_of_concepts = 10
            patch_size = 15
            craft = Craft(input_to_latent_model = g,
                        latent_to_logit_model = h,
                        number_of_concepts = number_of_concepts,
                        patch_size = patch_size,
                        batch_size = 64)

            # Now we can fit the concept using our images
            # Focus on class id 0
            class_id = 0
            images_preprocessed = x[y.argmax(1)==class_id] # select only images of class 'class_id'
            crops, crops_u, w = craft.fit(images_preprocessed, class_id)

            # Checking shape of crops, crops_u, w
            assert crops.shape[1] == crops.shape[2] == patch_size # Check patch sizes
            assert crops.shape[0] == crops_u.shape[0] # Check numbers of patches
            assert crops_u.shape[1] == w.shape[0] == number_of_concepts

            # Importance estimation
            importances = craft.estimate_importance()
            assert len(importances) == number_of_concepts
            assert np.all(importances >= 0)

            # Checking the results of transform()
            images_u = craft.transform(images_preprocessed)
            if len(images_u.shape) == 4:
                assert images_u.shape == (images_preprocessed.shape[0],
                                          images_preprocessed.shape[1]-1,
                                          images_preprocessed.shape[2]-1,
                                          number_of_concepts)
            elif len(images_u.shape) == 2:
                assert images_u.shape == (images_preprocessed.shape[0], number_of_concepts)
            else:
                raise ValueError("images_u contains the wrong shape")

def test_wrong_layers():
    """Ensure that Craft complains when the input models are incompatible"""

    input_shape = (32, 32, 3)
    nb_labels = 3

    # Generate a fake dataset
    model = generate_model(input_shape, nb_labels)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(0.1))

    g = tf.keras.Model(model.input, model.layers[0].output)
    h = lambda x: 2*x

    # Initialize Craft
    number_of_concepts = 10
    patch_size = 15
    with pytest.raises(TypeError):
        Craft(input_to_latent_model = g,
                latent_to_logit_model = h,
                number_of_concepts = number_of_concepts,
                patch_size = patch_size,
                batch_size = 64)

def test_classifier():
    """ Check the Craft results on a small fake dataset """

    input_shape = (64, 64, 3)
    nb_labels = 3
    nb_samples = 200

    # Create a dataset of 'ABC', 'BCD', 'CDE' images
    x, y, nb_samples, _ = generate_txt_images_data(input_shape, nb_labels, nb_samples)

    # train a small classifier on the dataset
    def create_classifier_model(input_shape=(64, 64, 3), output_shape=10):
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(Conv2D(6, kernel_size=(2, 2)))
        model.add(Activation('relu'))
        model.add(Conv2D(6, kernel_size=(2, 2)))
        model.add(Activation('relu'))
        model.add(Conv2D(6, kernel_size=(2, 2)))
        model.add(Activation('relu', name='relu'))
        model.add(Flatten())
        model.add(Dense(output_shape))
        model.add(Activation('softmax'))
        opt = Adam(learning_rate=0.005)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        return model

    model = create_classifier_model(input_shape, nb_labels)

    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Retrieve checkpoints
    checkpoint_path = "tests/concepts/checkpoints/classifier_test_craft_tf.ckpt"
    if not os.path.exists(f"{checkpoint_path}.index"):
        os.makedirs("tests/concepts/checkpoints/", exist_ok=True)
        identifier = "1NLA7x2EpElzEEmyvFQhD6VS6bMwS_bCs"
        download_file(identifier, f"{checkpoint_path}.index")

        identifier = "1wDi-y9b-3I_a-ZtqRlfuib-D7Ox4j8pX"
        download_file(identifier, f"{checkpoint_path}.data-00000-of-00001")

    model.load_weights(checkpoint_path)

    acc = np.sum(np.argmax(model(x), axis=1) == np.argmax(y, axis=1)) / nb_samples
    assert acc == 1.0

    # cut the model in two parts (as explained in the paper)
    # first part is g(.) our 'input_to_latent' model, second part is h(.) our 'latent_to_logit' model
    cut_layer = model.get_layer('relu')
    g = tf.keras.Model(model.inputs, cut_layer.output)
    h = tf.keras.Model(Input(tensor=cut_layer.output), model.outputs)

    assert np.all(g(x) >= 0.0)

    # Init Craft on the full dataset
    craft = Craft(input_to_latent_model = g,
                latent_to_logit_model = h,
                number_of_concepts = 3,
                patch_size = 12,
                batch_size = 32)

    # Expected best crop for class 0 (ABC) is AB
    AB_str = """
        0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 1 1 1 1 1 1 1 1
        0 0 0 0 0 0 1 0 0 0 1 1
        1 0 0 0 0 0 1 0 0 0 0 1
        1 0 0 0 0 0 1 0 0 0 1 1
        1 0 0 0 0 0 1 1 1 1 1 1
        1 1 0 0 0 0 1 0 0 0 0 1
        0 1 0 0 0 0 1 0 0 0 0 0
        0 1 1 0 0 0 1 0 0 0 0 1
        1 1 1 1 1 1 1 1 1 1 1 1
        0 0 0 0 0 0 0 0 0 0 0 0
    """
    AB = np.genfromtxt(AB_str.splitlines())

    # Expected best crop for class 1 (BCD) is BC
    BC_str = """
        0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0
        1 1 1 1 1 1 0 0 0 0 1 1
        1 0 0 0 1 1 0 0 0 1 1 0
        1 0 0 0 0 1 0 0 0 1 0 0
        1 0 0 0 1 1 0 0 0 1 0 0
        1 1 1 1 1 1 0 0 0 1 0 0
        1 0 0 0 0 1 1 0 0 1 0 0
        1 0 0 0 0 0 1 0 0 1 0 0
        1 0 0 0 0 1 1 0 0 1 1 0
        1 1 1 1 1 1 0 0 0 0 1 1
    """
    BC = np.genfromtxt(BC_str.splitlines())

    # Expected best crop for class 2 (CDE) is DE
    DE_str = """
        0 0 0 0 0 0 0 0 0 0 0 0
        1 0 0 1 1 1 1 1 1 1 1 0
        1 1 0 0 0 1 0 0 0 0 1 0
        0 1 0 0 0 1 0 0 0 0 1 0
        0 1 1 0 0 1 0 0 1 0 0 0
        0 1 1 0 0 1 1 1 1 0 0 0
        0 1 1 0 0 1 0 0 1 0 0 0
        0 1 0 0 0 1 0 0 0 0 1 1
        1 1 0 0 0 1 0 0 0 0 1 1
        1 0 0 1 1 1 1 1 1 1 1 1
        0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0
        """
    DE = np.genfromtxt(DE_str.splitlines())

    DE2_str = """
        0 0 0 0 0 0 0 0 0 0 0 Z
        1 1 1 1 0 0 1 1 1 1 1 1
        0 0 1 1 1 0 0 0 1 0 0 0
        0 0 0 0 1 0 0 0 1 0 0 0
        0 0 0 0 1 1 0 0 1 0 0 1
        0 0 0 0 1 1 0 0 1 1 1 1
        0 0 0 0 1 1 0 0 1 0 0 1
        0 0 0 0 1 0 0 0 1 0 0 0
        0 0 1 1 1 0 0 0 1 0 0 0
        1 1 1 1 0 0 1 1 1 1 1 1
        0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0
        """
    DE2 = np.genfromtxt(DE2_str.splitlines())

    expected_best_crops = [[AB], [BC], [DE, DE2]]
    expected_best_crops_names = ['AB', 'BC', 'DE']

    # Run 3 Craft studies on each class, and in each case check if the best crop is the expected one
    class_check = [False, False, False]
    for class_id in range(3):
        # Focus on class class_id
        # Selecting subset for class {class_id} : {labels_str[class_id]}'
        x_subset = x[np.argmax(y, axis=1)==class_id,:,:,:]

        # fit craft on the selected class
        crops, crops_u, w = craft.fit(x_subset, class_id)

        # compute importances
        importances = craft.estimate_importance()
        assert importances[0] > 0.8

        # find the best crop and compare it to the expected best crop
        most_important_concepts = np.argsort(importances)[::-1]

        # Find the best crop for the most important concept
        c_id = most_important_concepts[0]
        best_crops_ids = np.argsort(crops_u[:, c_id])[::-1]
        best_crop = np.array(crops)[best_crops_ids[0]]

        # Compare this best crop to the expectation
        predicted_best_crop = np.where(best_crop.sum(axis=2) > 0.25, 1, 0)
        for expected_best_crop in expected_best_crops[class_id]:
            expected_best_crop = expected_best_crop.astype(np.uint8)

            comparison = predicted_best_crop == expected_best_crop
            acc = np.sum(comparison) / len(comparison.ravel())
            check = acc > 0.9
            if check:
                class_check[class_id] = True
                break
    assert np.all(class_check)
