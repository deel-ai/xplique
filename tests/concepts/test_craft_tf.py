import numpy as np
import tensorflow as tf
import pytest

from tensorflow.keras.layers import Input

from xplique.concepts import CraftTf as Craft

from ..utils import generate_data, generate_model


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
