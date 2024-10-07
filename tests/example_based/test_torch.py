"""
Test example-based methods with PyTorch models and datasets.
"""

import pytest

import numpy as np
import tensorflow as tf
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from xplique.example_based import (
    SimilarExamples, Cole, MMDCritic, ProtoDash, ProtoGreedy,
    NaiveCounterFactuals, LabelAwareCounterFactuals, KLEORGlobalSim, KLEORSimMiss,
)
from xplique.example_based.projections import Projection, LatentSpaceProjection, HadamardProjection
from xplique.example_based.projections.commons import model_splitting

from xplique.example_based.datasets_operations.tf_dataset_operations import are_dataset_first_elems_equal
from xplique.example_based.datasets_operations.harmonize import harmonize_datasets

from tests.utils import almost_equal

    
def get_setup(input_shape, nb_samples=10, nb_labels=10):
    """
    Generate data and model for SimilarExamples
    """
    # Data generation
    x_train = torch.stack(
        [i * torch.ones(input_shape, dtype=torch.float32) for i in range(nb_samples)]
    )
    y_train = torch.arange(len(x_train), dtype=torch.int64) % nb_labels
    train_targets = F.one_hot(y_train, num_classes=nb_labels).to(torch.float32)
    
    x_test = x_train[1:-1]  # Exclude the first and last elements
    test_targets = train_targets[1:-1]  # Exclude the first and last elements

    return x_train, x_test, y_train, train_targets, test_targets


def create_cnn_model(input_shape, output_shape):
    in_channels, height, width = input_shape

    kernel_size = 3
    padding = 1
    stride = 1
    
    # Calculate the flattened size after the convolutional layers and pooling
    def conv_output_size(in_size):
        return (in_size - kernel_size + 2 * padding) // stride + 1

    height_after_conv1 = conv_output_size(height) // 2  # After first conv and pooling
    height_after_conv2 = conv_output_size(height_after_conv1) // 2  # After second conv and pooling

    width_after_conv1 = conv_output_size(width) // 2  # After first conv and pooling
    width_after_conv2 = conv_output_size(width_after_conv1) // 2  # After second conv and pooling

    flat_size = 8 * height_after_conv2 * width_after_conv2  # 8 is the number of filters in the last conv layer

    model = nn.Sequential(
        # Convolutional layer 1
        nn.Conv2d(in_channels=in_channels, out_channels=4, kernel_size=kernel_size, padding=padding),  # 4 filters
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),  # Pooling layer (2x2)
        
        # Convolutional layer 2
        nn.Conv2d(in_channels=4, out_channels=8, kernel_size=kernel_size, padding=padding),  # 8 filters
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),  # Pooling layer (2x2)
        
        # Flatten layer
        nn.Flatten(),
        
        # Fully connected layer 1
        nn.Linear(flat_size, 16),
        nn.ReLU(),
        
        # Output layer
        nn.Linear(16, output_shape)
    )

    # Initialize all weights to ones
    for layer in model:
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.constant_(layer.weight, 1.0)  # Set all weights to ones
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)  # Optionally set all biases to zero

    return model


def test_harmonize_datasets_with_torch():
    import torch
    
    cases = torch.rand(100, 10)
    labels = torch.randint(0, 2, (100, 1))
    batch_size = 10
    
    cases_out, labels_out, targets_out, batch_size_out = harmonize_datasets(cases, labels, batch_size=batch_size)
    
    assert targets_out is None, "Targets should be None when not provided."
    assert batch_size_out == batch_size, "Output batch size should match the input batch size."

    for case, label in zip(cases_out, labels_out):
        assert case.shape == (batch_size, cases.shape[1]), "Each case should have the same shape as the input cases."
        assert label.shape == (batch_size, labels.shape[1]), "Each label should have the same shape as the input labels."
        break


def test_inputs_combinations():
    """
    Test management of dataset init inputs
    """

    tf_tensor = tf.reshape(tf.range(90, dtype=tf.float32), (10, 3, 3))
    np_array = np.array(tf_tensor)
    tf_dataset = tf.data.Dataset.from_tensor_slices(tf_tensor)

    tf_dataset_b3 = tf_dataset.batch(3)
    tf_dataset_b5 = tf_dataset.batch(5)

    torch_tensor = torch.tensor(np_array)
    torch_dataset = TensorDataset(torch_tensor)
    zipped2 = TensorDataset(torch_tensor, torch_tensor)
    zipped3 = TensorDataset(torch_tensor, torch_tensor, torch_tensor)
    torch_dataloader_b3 = DataLoader(torch_dataset, batch_size=3, shuffle=False)
    torch_dataloader_b5 = DataLoader(torch_dataset, batch_size=5, shuffle=False)
    torch_zipped2_dataloader_b5 = DataLoader(zipped2, batch_size=5, shuffle=False)
    torch_zipped3_dataloader_b3 = DataLoader(zipped3, batch_size=3, shuffle=False)

    # Method initialization that should work
    cases_dataset, labels_dataset, targets_dataset, batch_size =\
        harmonize_datasets(torch_dataloader_b3, None, torch_dataloader_b3)
    assert are_dataset_first_elems_equal(cases_dataset, tf_dataset_b3)
    assert are_dataset_first_elems_equal(labels_dataset, None)
    assert are_dataset_first_elems_equal(targets_dataset, tf_dataset_b3)
    assert batch_size == 3

    cases_dataset, labels_dataset, targets_dataset, batch_size =\
        harmonize_datasets(torch_tensor, torch_tensor, None, batch_size=5)
    assert are_dataset_first_elems_equal(cases_dataset, tf_dataset_b5)
    assert are_dataset_first_elems_equal(labels_dataset, tf_dataset_b5)
    assert are_dataset_first_elems_equal(targets_dataset, None)
    assert batch_size == 5

    cases_dataset, labels_dataset, targets_dataset, batch_size =\
        harmonize_datasets(torch_zipped2_dataloader_b5, None, torch_dataloader_b5)
    assert are_dataset_first_elems_equal(cases_dataset, tf_dataset_b5)
    assert are_dataset_first_elems_equal(labels_dataset, tf_dataset_b5)
    assert are_dataset_first_elems_equal(targets_dataset, tf_dataset_b5)
    assert batch_size == 5

    cases_dataset, labels_dataset, targets_dataset, batch_size =\
        harmonize_datasets(torch_zipped3_dataloader_b3, batch_size=3)
    assert are_dataset_first_elems_equal(cases_dataset, tf_dataset_b3)
    assert are_dataset_first_elems_equal(labels_dataset, tf_dataset_b3)
    assert are_dataset_first_elems_equal(targets_dataset, tf_dataset_b3)
    assert batch_size == 3



def test_error_raising():
    """
    Test management of dataset init inputs
    """

    tf_tensor = tf.reshape(tf.range(90, dtype=tf.float32), (10, 3, 3))
    tf_dataset = tf.data.Dataset.from_tensor_slices(tf_tensor)

    torch_tensor = torch.reshape(torch.arange(90, dtype=torch.float32), (10, 3, 3))
    np_array = np.array(torch_tensor)

    torch_dataset = TensorDataset(torch_tensor)
    torch_dataloader = DataLoader(torch_dataset, batch_size=None, shuffle=False)
    torch_shuffled = DataLoader(torch_dataset, batch_size=4, shuffle=True)
    torch_dataloader_b3 = DataLoader(torch_dataset, batch_size=3, shuffle=False)
    torch_dataloader_b5 = DataLoader(torch_dataset, batch_size=5, shuffle=False)

    zipped2 = TensorDataset(torch_tensor, torch_tensor)
    zipped3 = TensorDataset(torch_tensor, torch_tensor, torch_tensor)
    torch_zipped2_dataloader_b5 = DataLoader(zipped2, batch_size=5, shuffle=False)
    torch_zipped3_dataloader_b3 = DataLoader(zipped3, batch_size=3, shuffle=False)

    too_long_torch_tensor = torch.cat([torch_tensor, torch_tensor], dim=0)
    too_long_torch_dataset = TensorDataset(too_long_torch_tensor)
    too_long_torch_dataloader_b10 = DataLoader(too_long_torch_dataset, batch_size=10, shuffle=False)


    # Method initialization that should not work

    # not input
    with pytest.raises(TypeError):
        harmonize_datasets()

    # shuffled
    with pytest.raises(AssertionError):
            harmonize_datasets(torch_shuffled,)

    # mismatching types
    with pytest.raises(AssertionError):
            harmonize_datasets(torch_dataloader_b3, torch_tensor,)
    with pytest.raises(AssertionError):
            harmonize_datasets(torch_tensor, tf_tensor,)
    with pytest.raises(AssertionError):
            harmonize_datasets(np_array, torch_tensor,)
    with pytest.raises(AssertionError):
            harmonize_datasets(np_array, torch_dataloader_b3,)
    with pytest.raises(AssertionError):
            harmonize_datasets(tf_dataset, torch_dataloader_b3,)
    with pytest.raises(AssertionError):
            harmonize_datasets(torch_zipped2_dataloader_b5, tf_tensor,)

    # labels or targets zipped
    with pytest.raises(AssertionError):
            harmonize_datasets(torch_dataloader_b5, torch_zipped2_dataloader_b5,)
    with pytest.raises(AssertionError):
            harmonize_datasets(torch_dataloader_b3, None, torch_zipped3_dataloader_b3,)

    # not batched and no batch size provided
    with pytest.raises(AssertionError):
            harmonize_datasets(torch_dataloader,)

    # not matching batch sizes
    with pytest.raises(AssertionError):
            harmonize_datasets(torch_dataloader_b3, torch_dataloader_b5,)
    with pytest.raises(AssertionError):
            harmonize_datasets(torch_zipped2_dataloader_b5, None, torch_dataloader_b3,)
    
    with pytest.raises(AssertionError):
        harmonize_datasets(
            too_long_torch_dataloader_b10,
            too_long_torch_dataloader_b10,
            torch_dataloader_b5,
        )

    # multiple datasets for labels or targets
    with pytest.raises(AssertionError):
            harmonize_datasets(torch_zipped2_dataloader_b5, torch_dataloader_b5,)
    with pytest.raises(AssertionError):
            harmonize_datasets(torch_zipped3_dataloader_b3, None, torch_dataloader_b3,)


def test_torch_model_splitting():

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    n_sample = 10
    torch_input_shape = (3, 32, 32)
    input_shape = (32, 32, 3)
    nb_labels = 10

    model = create_cnn_model(input_shape=torch_input_shape, output_shape=nb_labels)

    # generate data
    np_data = np.random.rand(n_sample, *input_shape).astype(np.float32)

    # inference with the initial model
    model.eval()
    model.to(device)
    torch_data = torch.tensor(np_data, device=device)
    with torch.no_grad():
        torch_channel_first_data = torch_data.permute(0, 3, 1, 2)
        np_predictions_1 = model(torch_channel_first_data).cpu().numpy()
    
    assert np_predictions_1.shape == (n_sample, nb_labels)

    # test splitting support different types
    _, _ = model_splitting(model, "flatten1")
    _, _ = model_splitting(model, -2)
    features_extractor, predictor = model_splitting(model, "last_conv")

    assert isinstance(features_extractor, tf.keras.Model)
    assert isinstance(predictor, tf.keras.Model)


    # inference with the splitted model
    tf_data = tf.convert_to_tensor(np_data)
    features = features_extractor(tf_data)
    tf_predictions = predictor(features)
    np_predictions_2 = tf_predictions.numpy()

    assert tf_predictions.shape == (n_sample, nb_labels)
    assert np.allclose(np_predictions_1, np_predictions_2, atol=1e-5)


def test_similar_examples_basic():
    """
    Test the SimilarExamples with an identity projection.
    """
    input_shape = (4, 4, 1)
    k = 3
    batch_size = 4

    x_train, x_test, y_train, _, _ = get_setup(input_shape)

    torch_dataset = TensorDataset(x_train, y_train)
    torch_dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=False)

    identity_projection = Projection(
        space_projection=lambda inputs, targets=None: inputs
    )

    # Method initialization
    method = SimilarExamples(
        cases_dataset=torch_dataloader,
        projection=identity_projection,
        k=k,
        batch_size=batch_size,
        distance="euclidean",
        case_returns=["examples", "labels"],
    )

    # Generate explanation
    outputs = method.explain(x_test)
    examples = outputs["examples"]
    labels = outputs["labels"]

    # Verifications
    # Shape should be (n, k, h, w, c)
    assert examples.shape == (len(x_test), k) + input_shape

    for i in range(len(x_test)):
        # test examples:
        assert almost_equal(np.array(examples[i, 0]), np.array(x_train[i + 1]))
        assert almost_equal(np.array(examples[i, 1]), np.array(x_train[i + 2]))\
            or almost_equal(np.array(examples[i, 1]), np.array(x_train[i]))
        assert almost_equal(np.array(examples[i, 2]), np.array(x_train[i]))\
            or almost_equal(np.array(examples[i, 2]), np.array(x_train[i + 2]))
        
        # test labels:
        assert almost_equal(np.array(labels[i, 0]), np.array(y_train[i + 1]))
        assert almost_equal(np.array(labels[i, 1]), np.array(y_train[i + 2]))\
            or almost_equal(np.array(labels[i, 1]), np.array(y_train[i]))
        assert almost_equal(np.array(labels[i, 2]), np.array(y_train[i]))\
            or almost_equal(np.array(labels[i, 2]), np.array(y_train[i + 2]))


def test_similar_examples_with_splitting():
    """
    Test the SimilarExamples with an identity projection.
    """
    # Setup
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    nb_samples = 10
    torch_input_shape = (3, 32, 32)
    input_shape = (32, 32, 3)
    nb_labels = 10
    k = 3
    batch_size = 4

    x_train, x_test, y_train, _, _ = get_setup(input_shape, nb_samples, nb_labels)
    torch_dataset = TensorDataset(x_train, y_train)
    torch_dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=False)

    model = create_cnn_model(input_shape=torch_input_shape, output_shape=nb_labels)
    projection = LatentSpaceProjection(model, "last_conv", device=device)

    # Method initialization
    method = SimilarExamples(
        cases_dataset=torch_dataloader,
        projection=projection,
        k=k,
        batch_size=batch_size,
        distance="euclidean",
        case_returns=["examples", "labels"],
    )

    # Generate explanation
    outputs = method.explain(x_test)
    examples = outputs["examples"]
    labels = outputs["labels"]

    # Verifications
    # Shape should be (n, k, h, w, c)
    assert examples.shape == (len(x_test), k) + input_shape

    for i in range(len(x_test)):
        # test examples:
        assert almost_equal(np.array(examples[i, 0]), np.array(x_train[i + 1]))
        assert almost_equal(np.array(examples[i, 1]), np.array(x_train[i + 2]))\
            or almost_equal(np.array(examples[i, 1]), np.array(x_train[i]))
        assert almost_equal(np.array(examples[i, 2]), np.array(x_train[i]))\
            or almost_equal(np.array(examples[i, 2]), np.array(x_train[i + 2]))
        
        # test labels:
        assert almost_equal(np.array(labels[i, 0]), np.array(y_train[i + 1]))
        assert almost_equal(np.array(labels[i, 1]), np.array(y_train[i + 2]))\
            or almost_equal(np.array(labels[i, 1]), np.array(y_train[i]))
        assert almost_equal(np.array(labels[i, 2]), np.array(y_train[i]))\
            or almost_equal(np.array(labels[i, 2]), np.array(y_train[i + 2]))


def test_all_methods_with_torch():
    # Setup
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    nb_samples = 13
    torch_input_shape = (3, 32, 32)
    input_shape = (32, 32, 3)
    nb_labels = 5
    batch_size = 4

    x_train, x_test, y_train, train_targets, test_targets = get_setup(input_shape, nb_samples, nb_labels)
    torch_dataset = TensorDataset(x_train, y_train)
    torch_dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=False)
    targets_dataloader = DataLoader(TensorDataset(train_targets), batch_size=batch_size, shuffle=False)

    model = create_cnn_model(input_shape=torch_input_shape, output_shape=nb_labels)
    projection = HadamardProjection(model, "last_conv", device=device)

    methods = [SimilarExamples, Cole, MMDCritic, ProtoDash, ProtoGreedy,
               NaiveCounterFactuals, LabelAwareCounterFactuals, KLEORGlobalSim, KLEORSimMiss,]

    for method_class in methods:
        if method_class == Cole:
            method = method_class(
                cases_dataset=torch_dataloader,
                targets_dataset=targets_dataloader,
                case_returns="all",
                model=model,
                latent_layer="last_conv",
                device=device,
            )
        else:
            method = method_class(
                cases_dataset=torch_dataloader,
                targets_dataset=targets_dataloader,
                projection=projection,
                case_returns="all",
            )

        # Generate explanation
        if method_class == LabelAwareCounterFactuals:
            outputs = method.explain(x_test, cf_expected_classes=test_targets)
        elif method_class in [NaiveCounterFactuals, KLEORGlobalSim, KLEORSimMiss]:
            outputs = method.explain(x_test, targets=test_targets)
        else:
            outputs = method.explain(x_test, targets=None)

        examples = outputs["examples"]
        labels = outputs["labels"]

        assert examples.shape == (len(x_test), 2) + input_shape
        assert labels.shape == (len(x_test), 1)
