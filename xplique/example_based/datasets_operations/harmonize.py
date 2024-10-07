"""
Allow Example-based methods to work with different types of datasets and tensors.
"""


import math

import numpy as np
import tensorflow as tf

from ...types import Optional, Tuple, DatasetOrTensor
from .tf_dataset_operations import sanitize_dataset, is_batched


def split_tf_dataset(cases_dataset: tf.data.Dataset,
                     labels_dataset: Optional[tf.data.Dataset] = None,
                     targets_dataset: Optional[tf.data.Dataset] = None
                     ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Splits a TensorFlow dataset into cases, labels, and targets datasets.
    The dataset is splitted only if it has multiple columns.
    If the dataset has 2 columns, the second column is assumed to be the labels.
    If the dataset has several columns but labels and targets are provided,
    there is a conflict and an error is raised.

    Parameters
    ----------
    cases_dataset
        The dataset to split.
    labels_dataset
        Labels associated with the cases in the `cases_dataset`.
        If this function is called, it should be `None`.
    targets_dataset
        Targets associated with the cases in the `cases_dataset`.
        If this function is called and `cases_dataset` has 3 columns, it should be `None`.
    
    Returns
    -------
    cases_dataset
        The dataset used to train the model.
    labels_dataset
        Labels associated with the `cases_dataset`.
    targets_dataset
        Targets associated with the `cases_dataset`.
    """

    assert isinstance(cases_dataset, tf.data.Dataset), (
        f"The dataset should be a `tf.data.Dataset`, got {type(cases_dataset)}."
    )

    if isinstance(cases_dataset.element_spec, tuple):
        if len(cases_dataset.element_spec) == 2:
            assert labels_dataset is None, (
                "The second column of `cases_dataset` is assumed to be the labels. "\
                + "Hence, `labels_dataset` should be empty."
            )
            labels_dataset = cases_dataset.map(lambda x, y: y)
            cases_dataset = cases_dataset.map(lambda x, y: x)
        elif len(cases_dataset.element_spec) == 3:
            assert labels_dataset is None and targets_dataset is None, (
                "The second and third columns of `cases_dataset` are assumed to be the labels "\
                "and targets. Hence, `labels_dataset` and `targets_dataset` should be empty."
            )
            targets_dataset = cases_dataset.map(lambda x, y, t: t)
            labels_dataset = cases_dataset.map(lambda x, y, t: y)
            cases_dataset = cases_dataset.map(lambda x, y, t: x)
        else:
            raise AttributeError(
                "`cases_dataset` cannot have more than 3 columns, "
                + f"{len(cases_dataset.element_spec)} were detected."
            )

    return cases_dataset, labels_dataset, targets_dataset


def harmonize_datasets(
        cases_dataset: DatasetOrTensor,
        labels_dataset: Optional[DatasetOrTensor] = None,
        targets_dataset: Optional[DatasetOrTensor] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, int]:
    """
    Harmonizes the provided datasets, transforming them to tf.data.Dataset if necessary.
    Datasets are also checked in case they are shuffled or do not match in batch_size.
    If the datasets have multiple columns, the function will split them into cases,
    labels, and targets datasets based on the number of columns.
    
    This function supports both TensorFlow and PyTorch datasets.
    
    Parameters
    ----------
    cases_dataset
        The dataset used to train the model, examples are extracted from this dataset.
        All datasets (cases, labels, and targets) should be of the same type.
        Supported types are: `tf.data.Dataset`, `torch.utils.data.DataLoader`,
        `tf.Tensor`, `np.ndarray`, `torch.Tensor`.
        For datasets with multiple columns, the first column is assumed to be the cases.
        While the second column is assumed to be the labels, and the third the targets.
        Warning: datasets tend to reshuffle at each iteration, ensure the datasets are
        not reshuffle as we use index in the dataset.
    labels_dataset
        Labels associated with the examples in the `cases_dataset`.
        It should have the same type as `cases_dataset`.
    targets_dataset
        Targets associated with the `cases_dataset` for dataset projection,
        oftentimes the one-hot encoding of a model's predictions. See `projection` for detail.
        It should have the same type as `cases_dataset`.
        It is not be necessary for all projections.
        Furthermore, projections which requires it compute it internally by default.
    batch_size : Optional[int]
        Number of samples treated simultaneously when using the datasets.
        It should match the batch size of the datasets if they are batched.

    Returns
    -------
    cases_dataset
        The harmonized dataset used to train the model.
    labels_dataset
        Harmonized labels associated with the `cases_dataset`.
    targets_dataset
        Harmonized targets associated with the `cases_dataset`.
    batch_size : int
        Number of samples treated simultaneously when using the datasets.
    """
    # pylint: disable=too-many-statements
    # pylint: disable=too-many-branches
    # Ensure the datasets are of the same type
    if labels_dataset is not None:
        if isinstance(cases_dataset, tf.data.Dataset):
            assert isinstance(labels_dataset, tf.data.Dataset), (
                "The labels_dataset should be a `tf.data.Dataset` if the cases_dataset is."
            )
            assert not isinstance(labels_dataset.element_spec, tuple), (
                "The labels_dataset should only have one column."
            )
        else:
            assert isinstance(cases_dataset, type(labels_dataset)), (
                "The cases_dataset and labels_dataset should be of the same type."\
                + f"Got {type(cases_dataset)} and {type(labels_dataset)}."
            )
    if targets_dataset is not None:
        if isinstance(cases_dataset, tf.data.Dataset):
            assert isinstance(targets_dataset, tf.data.Dataset), (
                "The targets_dataset should be a `tf.data.Dataset` if the cases_dataset is."
            )
            assert not isinstance(targets_dataset.element_spec, tuple), (
                "The targets_dataset should only have one column."
            )
        else:
            assert isinstance(cases_dataset, type(targets_dataset)), (
                "The cases_dataset and targets_dataset should be of the same type."\
                + f"Got {type(cases_dataset)} and {type(targets_dataset)}."
            )

    # Determine batch size and cardinality based on the dataset type
    # for torch elements, convert them to numpy arrays or tf datasets
    if isinstance(cases_dataset, tf.data.Dataset):
        # compute batch size and cardinality
        if is_batched(cases_dataset):
            if isinstance(cases_dataset.element_spec, tuple):
                batch_size = tf.shape(next(iter(cases_dataset))[0])[0].numpy()
            else:
                batch_size = tf.shape(next(iter(cases_dataset)))[0].numpy()
        else:
            assert batch_size is not None, (
                "The dataset is not batched, hence a `batch_size` should be provided."
            )
            cases_dataset = cases_dataset.batch(batch_size)
        cardinality = cases_dataset.cardinality().numpy()

        # handle multi-column datasets
        if isinstance(cases_dataset.element_spec, tuple):
            # split dataset if `cases_dataset` has multiple columns
            cases_dataset, labels_dataset, targets_dataset =\
                split_tf_dataset(cases_dataset, labels_dataset, targets_dataset)
    elif isinstance(cases_dataset, (np.ndarray, tf.Tensor)):
        # compute batch size and cardinality
        if batch_size is None:
            # no batching, one batch encompass all the dataset
            batch_size = cases_dataset.shape[0]
        else:
            batch_size = min(batch_size, cases_dataset.shape[0])
        cardinality = math.ceil(cases_dataset.shape[0] / batch_size)

        # tensors will be converted to tf.data.Dataset via the snitize function
    else:
        error_message = "Unknown cases dataset type, should be in: [tf.data.Dataset, tf.Tensor, "\
                        + "np.ndarray, torch.Tensor, torch.utils.data.DataLoader]. "\
                        + f"But got {type(cases_dataset)} instead."
        # try to import torch and torch.utils.data.DataLoader to treat possible input types
        try:
            # pylint: disable=import-outside-toplevel
            import torch
            from .convert_torch_to_tf import split_and_convert_column_dataloader
        except ImportError as exc:
            raise AttributeError(error_message) from exc

        if isinstance(cases_dataset, torch.Tensor):
            # compute batch size and cardinality
            if batch_size is None:
                # no batching, one batch encompass all the dataset
                batch_size = cases_dataset.shape[0]
            else:
                batch_size = min(batch_size, cases_dataset.shape[0])
            cardinality = math.ceil(cases_dataset.shape[0] / batch_size)

            # convert torch tensor to numpy array
            cases_dataset = cases_dataset.cpu().numpy()
            if labels_dataset is not None:
                labels_dataset = labels_dataset.cpu().numpy()
            if targets_dataset is not None:
                targets_dataset = targets_dataset.cpu().numpy()

            # tensors will be converted to tf.data.Dataset via the snitize function
        elif isinstance(cases_dataset, torch.utils.data.DataLoader):
            batch_size = cases_dataset.batch_size
            cardinality = len(cases_dataset)
            cases_dataset, labels_dataset, targets_dataset =\
                split_and_convert_column_dataloader(cases_dataset, labels_dataset, targets_dataset)
        else:
            raise AttributeError(error_message)

    # Sanitize datasets to ensure they are in the correct format
    cases_dataset = sanitize_dataset(cases_dataset, batch_size, cardinality)
    labels_dataset = sanitize_dataset(labels_dataset, batch_size, cardinality)
    targets_dataset = sanitize_dataset(targets_dataset, batch_size, cardinality)

    # Prefetch datasets
    cases_dataset = cases_dataset.prefetch(tf.data.AUTOTUNE)
    if labels_dataset is not None:
        labels_dataset = labels_dataset.prefetch(tf.data.AUTOTUNE)
    if targets_dataset is not None:
        targets_dataset = targets_dataset.prefetch(tf.data.AUTOTUNE)

    return cases_dataset, labels_dataset, targets_dataset, batch_size
