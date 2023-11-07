import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import random

from xplique.concepts import CraftTorch as Craft
from ..utils import generate_txt_images_data
from ..utils import download_file

def generate_torch_data(x_shape=(3, 32, 32), num_labels=10, samples=100):
    x = torch.tensor(np.random.rand(samples, *x_shape).astype(np.float32))
    y = F.one_hot(torch.tensor(np.random.randint(0, num_labels, samples), dtype=torch.int64),
                  num_labels)

    return x, y

def generate_torch_model(input_shape=(3, 32, 32, 3), output_shape=10):
    """Creates a basic torch model that can be used for testing purpose"""
    c_in = input_shape[0]
    h_in = input_shape[1]
    w_in = input_shape[2]

    model = nn.Sequential()

    model.append(nn.Conv2d(c_in, 4, (2, 2)))
    h_out = h_in - 1
    w_out = w_in -1
    c_out = 4

    model.append(nn.ReLU())
    model.append(nn.MaxPool2d((2, 2)))
    h_out = int((h_out - 2)/2 + 1)
    w_out = int((w_out - 2)/2 + 1)

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

            craft = Craft(input_to_latent_model = g,
                        latent_to_logit_model = h,
                        number_of_concepts = number_of_concepts,
                        patch_size = patch_size,
                        batch_size = 64,
                        device = 'cpu')

            # Now we can fit the concept using our images
            # Focus on class id 0
            class_id = 0
            images_preprocessed = x[y.argmax(1)==class_id] # select only images of class 'class_id'
            crops, crops_u, w = craft.fit(images_preprocessed, class_id)

            # Checking shape of crops, crops_u, w
            assert crops.shape[2] == crops.shape[3] == patch_size # Check patch sizes
            assert crops.shape[0] == crops_u.shape[0] # Check numbers of patches
            assert crops_u.shape[1] == w.shape[0]

            # Importance estimation
            importances = craft.estimate_importance()
            assert len(importances) == number_of_concepts

            # Checking the results of transform()
            images_u = craft.transform(images_preprocessed)
            if len(images_u.shape) == 4:
                assert images_u.shape == (images_preprocessed.shape[0],
                                          images_preprocessed.shape[2]-1,
                                          images_preprocessed.shape[3]-1,
                                          number_of_concepts)
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

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    random.seed(0)
    np.random.seed(0)

    # Create a dataset of 'ABC', 'BCD', 'CDE' images
    x, y, nb_samples, _ = generate_txt_images_data(input_shape, nb_labels, nb_samples)
    x = np.moveaxis(x, -1, 1) # reorder the axis to match torch format
    x, y = torch.Tensor(x), torch.Tensor(y)

    # train a small classifier on the dataset
    def create_torch_classifier_model(input_shape=(3, 64, 64), output_shape=10):
        flatten_size = 6*(input_shape[1]-3)*(input_shape[2]-3)
        model = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.Flatten(1, -1),
            # nn.Dropout(p=0.2),
            nn.Linear(flatten_size, output_shape))
        for layer in model:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                layer.bias.data.fill_(0.01)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                layer.bias.data.fill_(0.01)
        return model

    model = create_torch_classifier_model((input_shape[-1], *input_shape[0:2]), nb_labels)

    # Retrieve checkpoints
    checkpoint_path = "tests/concepts/checkpoints/classifier_test_craft_torch.ckpt"
    if not os.path.exists(checkpoint_path):
        os.makedirs("tests/concepts/checkpoints/", exist_ok=True)
        identifier = "1vz6hMibMEN6_t9yAY9SS4iaMY8G8aAPQ"
        download_file(identifier, checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path))

    # check accuracy
    model.eval()
    acc = torch.sum(torch.argmax(model(x), axis=1) == torch.argmax(y, axis=1))/len(y)
    assert acc > 0.9

    # cut pytorch model
    g = nn.Sequential(*(list(model.children())[:6])) # input to penultimate layer
    h = nn.Sequential(*(list(model.children())[6:])) # penultimate layer to logits
    assert torch.all(g(x) >= 0.0)

    # Init Craft on the full dataset
    craft = Craft(input_to_latent_model = g,
                latent_to_logit_model = h,
                number_of_concepts = 3,
                patch_size = 12,
                batch_size = 32,
                device='cpu')

    # Expected best crop for class 0 (ABC) is AB
    AB_str = """
        0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0
        0 1 1 1 1 1 1 1 1 0 0 0
        0 0 0 1 0 0 0 1 1 0 0 1
        0 0 0 1 0 0 0 0 1 0 0 1
        0 0 0 1 0 0 0 1 1 0 0 1
        0 0 0 1 1 1 1 1 1 0 0 1
        0 0 0 1 0 0 0 0 1 1 0 1
        0 0 0 1 0 0 0 0 0 1 0 1
        0 0 0 1 0 0 0 0 1 1 0 1
        1 1 1 1 1 1 1 1 1 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0
        """
    AB = np.genfromtxt(AB_str.splitlines())

    # Expected best crop for class 1 (BCD) is BC
    BC_str = """
        1 1 1 1 1 1 0 0 0 0 1 1
        1 0 0 0 1 1 0 0 0 1 1 0
        1 0 0 0 0 1 0 0 0 1 0 0
        1 0 0 0 1 1 0 0 0 1 0 0
        1 1 1 1 1 1 0 0 0 1 0 0
        1 0 0 0 0 1 1 0 0 1 0 0
        1 0 0 0 0 0 1 0 0 1 0 0
        1 0 0 0 0 1 1 0 0 1 1 0
        1 1 1 1 1 1 0 0 0 0 1 1
        0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0
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
        0 0 0 0 0 0 0 0 0 0 0 0
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
        assert np.all(importances >= 0)

        # find the best crop and compare it to the expected best crop
        most_important_concepts = np.argsort(importances)[::-1]

        # Find the best crop for the most important concept
        c_id = most_important_concepts[0]
        best_crops_ids = np.argsort(crops_u[:, c_id])[::-1]
        best_crop = np.array(crops)[best_crops_ids[0]]
        best_crop = np.moveaxis(best_crop, 0, -1)

        # Compare this best crop to the expectation
        predicted_best_crop = np.where(best_crop.sum(axis=2) > 0.25, 1, 0)

        # Comparison between expected:
        for expected_best_crop in expected_best_crops[class_id]:
            expected_best_crop = expected_best_crop.astype(np.uint8)
            comparison = predicted_best_crop == expected_best_crop
            acc = np.sum(comparison) / len(comparison.ravel())
            check = acc > 0.9
            if check:
                class_check[class_id] = True
                break
    assert np.all(class_check)
