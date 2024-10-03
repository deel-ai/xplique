"""
Set of functions to convert `torch.utils.data.DataLoader` and `torch.Tensor` to `tf.data.Dataset`
"""
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import DataLoader


def convert_column_dataloader_to_tf_dataset(
        dataloader: torch.utils.data.DataLoader,
        elements_shape: Tuple[int],
        column_index: Optional[int] = None,
    ) -> tf.data.Dataset:
    """
    Converts a PyTorch torch.utils.data.DataLoader to a TensorFlow Dataset.

    Parameters
    ----------
    dataloader
        The DataLoader to convert.
    elements_shape
        The shape of the elements in the DataLoader.
    column_index
        The index of the column to convert.
        If `None`, the entire DataLoader is converted.

    Returns
    -------
    dataset
        The converted dataset.
    """

    # make generator from dataloader
    if column_index is None:
        def generator():
            for elements in dataloader:
                yield tf.cast(elements.numpy(), tf.float32)
    else:
        def generator():
            for elements in dataloader:
                tf_elements = tf.cast(elements[column_index].numpy(), tf.float32)
                yield tf.cast(elements[column_index].numpy(), tf.float32)

    # create tf dataset from generator
    dataset = tf.data.Dataset.from_generator(
        lambda: generator(),
        output_signature=tf.TensorSpec(shape=elements_shape, dtype=tf.float32),
    )

    return dataset


def split_and_convert_column_dataloader(
        cases_dataset: torch.utils.data.DataLoader,
        labels_dataset: Optional[torch.utils.data.DataLoader] = None,
        targets_dataset: Optional[torch.utils.data.DataLoader] = None,
        ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Splits a PyTorch DataLoader into cases, labels, and targets datasets.
    The DataLoader is splitted only if it has multiple columns.
    If the DataLoader has 2 columns, the second column is assumed to be the labels.
    If the DataLoader has several columns but labels and targets are provided,
    there is a conflict and an error is raised.
    The splitted parts are then converted to TensorFlow datasets.

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
    first_cases = next(iter(cases_dataset))

    if not (isinstance(first_cases, tuple) or isinstance(first_cases, list)):
        # the cases dataset only has one column

        # manage cases dataset
        cases_shape = (None,) + first_cases.shape[1:]
        new_cases_dataset = convert_column_dataloader_to_tf_dataset(cases_dataset, cases_shape)
    
    else:
        # manage cases dataset
        cases_shape = (None,) + first_cases[0].shape[1:]
        new_cases_dataset = convert_column_dataloader_to_tf_dataset(
            cases_dataset, cases_shape, column_index=0)
        
        if len(first_cases) >= 2:
            # the cases dataset has two columns
            assert labels_dataset is None, (
                "The second column of `cases_dataset` is assumed to be the labels. "\
                + "Hence, `labels_dataset` should be empty."
            )

            # manage labels dataset (extract them from the second column of `cases_dataset`)
            labels_shape = (None,) + first_cases[1].shape[1:]
            labels_dataset = convert_column_dataloader_to_tf_dataset(
                cases_dataset, labels_shape, column_index=1)

            if len(first_cases) == 3:
                # the cases dataset has three columns
                assert targets_dataset is None, (
                    "The second and third columns of `cases_dataset` are assumed to be the labels "\
                    "and targets. Hence, `labels_dataset` and `targets_dataset` should be empty."
                )
                # manage targets dataset (extract them from the third column of `cases_dataset`)
                targets_shape = (None,) + first_cases[2].shape[1:]
                targets_dataset = convert_column_dataloader_to_tf_dataset(
                    cases_dataset, targets_shape, column_index=2)

            elif len(first_cases) > 3:
                raise AttributeError(
                    "`cases_dataset` cannot have more than 3 columns, "
                    + f"{len(first_cases)} were detected."
                )

    # manage labels datasets
    if labels_dataset is not None:
        if isinstance(labels_dataset, tf.data.Dataset):
            pass
        elif isinstance(labels_dataset, torch.utils.data.DataLoader):
            first_labels = next(iter(labels_dataset))
            if isinstance(first_labels, tuple) or isinstance(first_labels, list):
                assert len(first_labels) == 1, (
                    "The `labels_dataset` should only have one column. "
                    + f"{len(first_labels)} were detected."
                )
                labels_shape = (None,) + first_labels[0].shape[1:]
                labels_dataset = convert_column_dataloader_to_tf_dataset(labels_dataset, labels_shape, column_index=0)
            else:
                labels_shape = (None,) + first_labels.shape[1:]
                labels_dataset = convert_column_dataloader_to_tf_dataset(labels_dataset, labels_shape)
        else:
            raise AttributeError(
                "The `labels_dataset` should be a PyTorch DataLoader or a TensorFlow Dataset. "
                + f"{type(labels_dataset)} was detected."
            )
    else:
        labels_dataset = None
    
    # manage targets datasets
    if targets_dataset is not None:
        if isinstance(targets_dataset, tf.data.Dataset):
            pass
        elif isinstance(targets_dataset, torch.utils.data.DataLoader):
            first_targets = next(iter(targets_dataset))
            if isinstance(first_targets, tuple) or isinstance(first_targets, list):
                assert len(first_targets) == 1, (
                    "The `targets_dataset` should only have one column. "
                    + f"{len(first_targets)} were detected."
                )
                targets_shape = (None,) + first_targets[0].shape[1:]
                targets_dataset = convert_column_dataloader_to_tf_dataset(targets_dataset, targets_shape, column_index=0)
            else:
                targets_shape = (None,) + first_targets.shape[1:]
                targets_dataset = convert_column_dataloader_to_tf_dataset(targets_dataset, targets_shape)
        else:
            raise AttributeError(
                "The `labels_dataset` should be a PyTorch DataLoader or a TensorFlow Dataset. "
                + f"{type(labels_dataset)} was detected."
            )
    else:
        targets_dataset = None

    return new_cases_dataset, labels_dataset, targets_dataset
