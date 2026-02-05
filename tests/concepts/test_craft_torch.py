import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from xplique.concepts import CraftTorch as Craft


def generate_torch_data(x_shape=(3, 32, 32), num_labels=10, samples=100):
    x = torch.tensor(np.random.rand(samples, *x_shape).astype(np.float32))
    y = F.one_hot(
        torch.tensor(np.random.randint(0, num_labels, samples), dtype=torch.int64), num_labels
    )

    return x, y


def generate_torch_model(input_shape=(3, 32, 32, 3), output_shape=10):
    """Creates a basic torch model that can be used for testing purpose"""
    c_in = input_shape[0]
    h_in = input_shape[1]
    w_in = input_shape[2]

    model = nn.Sequential()

    model.append(nn.Conv2d(c_in, 4, (2, 2)))
    h_out = h_in - 1
    w_out = w_in - 1
    c_out = 4

    model.append(nn.ReLU())
    model.append(nn.MaxPool2d((2, 2)))
    h_out = int((h_out - 2) / 2 + 1)
    w_out = int((w_out - 2) / 2 + 1)

    model.append(nn.Dropout(0.25))
    model.append(nn.Flatten())
    flatten_size = c_out * h_out * w_out

    model.append(nn.Linear(int(flatten_size), output_shape))

    return model


def test_shape():
    """Ensure the output shape is correct"""

    input_shapes = [(3, 32, 32), (1, 32, 32), (3, 64, 64), (3, 64, 32)]
    nb_labels = 3
    nb_samples = 100

    for input_shape in input_shapes:
        # Generate a fake dataset
        x, y = generate_torch_data(input_shape, nb_labels, nb_samples)
        model = generate_torch_model(input_shape, nb_labels)

        # cut the model in two parts (as explained in the paper)
        # first part is g(.) our 'input_to_latent' model,
        # second part is h(.) our 'latent_to_logit' model
        #
        # Use 2 different set of indexes to check the behavior for
        # different activation shapes ; (0, 1) produces an activation.shape
        # of 4 dims, (-1, -1) leads to an activation.shape of 2 dims
        # for index_layer_g, index_layer_h in [(0, 1), (-1, -1)]:

        for index_layer_g, index_layer_h in [(2, 2), (-1, -1)]:
            g = nn.Sequential(*list(model.children())[:index_layer_g])
            h = nn.Sequential(*list(model.children())[index_layer_h:])

            # The activations must be positives
            assert torch.all(g(x) >= 0.0)

            # Initialize Craft
            number_of_concepts = 10
            patch_size = 15

            craft = Craft(
                input_to_latent_model=g,
                latent_to_logit_model=h,
                number_of_concepts=number_of_concepts,
                patch_size=patch_size,
                batch_size=64,
                device="cpu",
            )

            # Now we can fit the concept using our images
            # Focus on class id 0
            class_id = 0
            images_preprocessed = x[
                y.argmax(1) == class_id
            ]  # select only images of class 'class_id'
            crops, crops_u, w = craft.fit(images_preprocessed, class_id)

            # Checking shape of crops, crops_u, w
            assert crops.shape[2] == crops.shape[3] == patch_size  # Check patch sizes
            assert crops.shape[0] == crops_u.shape[0]  # Check numbers of patches
            assert crops_u.shape[1] == w.shape[0]

            # Importance estimation
            importances = craft.estimate_importance()
            assert len(importances) == number_of_concepts

            # Checking the results of transform()
            images_u = craft.transform(images_preprocessed)
            if len(images_u.shape) == 4:
                assert images_u.shape == (
                    images_preprocessed.shape[0],
                    images_preprocessed.shape[2] - 1,
                    images_preprocessed.shape[3] - 1,
                    number_of_concepts,
                )
            elif len(images_u.shape) == 2:
                assert images_u.shape == (images_preprocessed.shape[0], number_of_concepts)
            else:
                raise ValueError("images_u contains the wrong shape")


def test_wrong_layers():
    """Ensure that Craft complains when the input models are incompatible"""

    input_shape = (3, 32, 32)
    nb_labels = 3

    model = generate_torch_model(input_shape, nb_labels)

    # cut the model in two parts (as explained in the paper)
    # first part is g(.) our 'input_to_latent' model,
    # second part is h(.) our 'latent_to_logit' model
    g = nn.Sequential(*list(model.children())[:3])
    h = lambda x: 2 * x

    # Initialize Craft
    number_of_concepts = 10
    patch_size = 15
    with pytest.raises(TypeError):
        Craft(
            input_to_latent_model=g,
            latent_to_logit_model=h,
            number_of_concepts=number_of_concepts,
            patch_size=patch_size,
            batch_size=64,
        )
