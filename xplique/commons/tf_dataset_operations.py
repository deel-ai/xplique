"""
Set of functions to manipulated `tf.data.Dataset`
"""
from itertools import product

import numpy as np
import tensorflow as tf

from ..types import Optional, Union


def _almost_equal(arr1, arr2, epsilon=1e-6):
    """Ensure two array are almost equal at an epsilon"""
    if isinstance(arr1, dict):
        assert ("input_ids" in arr1.keys()), f"As the input batch is a dictionnary we expect it to \
                be a dictionnary as expected by Hugging Face model thus containing 'input_ids'. The dict \
                keys are {arr1.keys()}."
        arr1 = arr1['input_ids']
        if isinstance(arr2, dict):
            assert ("input_ids" in arr2.keys()), f"As the input batch is a dictionnary we expect it to \
                    be a dictionnary as expected by Hugging Face model thus containing 'input_ids'. The dict \
                    keys are {arr2.keys()}."
            arr2 = arr2['input_ids']
            return np.shape(arr1) == np.shape(arr2) and np.sum(np.abs(arr1 - arr2)) < epsilon
    return np.shape(arr1) == np.shape(arr2) and np.sum(np.abs(arr1 - arr2)) < epsilon


def are_dataset_first_elems_equal(
    dataset1: Optional[tf.data.Dataset], dataset2: Optional[tf.data.Dataset]
) -> bool:
    """
    Test if the first batch of elements of two datasets are the same.
    It is used to verify equality between datasets in a lazy way.

    Parameters
    ----------
    dataset1
        First `tf.data.Dataset` to compare.
    dataset2
        Second `tf.data.Dataset` to compare.

    Returns
    -------
    test_result
        Boolean value of the equality.
    """
    if dataset1 is None:
        return dataset2 is None

    if dataset2 is None:
        return False

    next1 = next(iter(dataset1))
    next2 = next(iter(dataset2))
    if isinstance(next1, tuple):
        if isinstance(next1, dict): # for the case where input is a dict (HF)
            assert ("input_ids" in next1.keys()), f"As the input batch is a dictionnary we expect it to \
                    be a dictionnary as expected by Hugging Face model thus containing 'input_ids'. The dict \
                    keys are {next1.keys()}."
            next1 = next1['input_ids']
        next1 = next1[0]
        if isinstance(next2, tuple):
            if isinstance(next2, dict): # for the case where input is a dict (HF)
                assert ("input_ids" in next2.keys()), f"As the input batch is a dictionnary we expect it to \
                        be a dictionnary as expected by Hugging Face model thus containing 'input_ids'. The dict \
                        keys are {next2.keys()}."
                next2 = next2['input_ids']
            next2 = next2[0]
        else:
            return False

    return _almost_equal(next1, next2)


def is_not_shuffled(dataset: Optional[tf.data.Dataset]) -> bool:
    """
    Test if the provided dataset reshuffle at each iteration.
    Tensorflow do not provide clean way to verify it,
    hence we draw two times the first element and compare it.
    It may not always detect shuffled datasets, but this is enough of a safety net.

    Parameters
    ----------
    dataset
        Tensorflow dataset to test.

    Returns
    -------
    test_result
        Boolean value of the test.
    """
    return are_dataset_first_elems_equal(dataset, dataset)


def batch_size_matches(dataset: Optional[tf.data.Dataset], batch_size: int) -> bool:
    """
    Test if batch size of a tensorflow dataset matches the expected one.
    Tensorflow do not provide clean way to verify it,
    hence we draw a batch and check its first dimension.
    It may fail in some really precise cases, but this is enough of a safety net.

    Parameters
    ----------
    dataset
        Tensorflow dataset to test.
    batch_size
        The expected batch size of the dataset.

    Returns
    -------
    test_result
        Boolean value of the test.
    """
    if dataset is None:
        # ignored
        return True

    first_item = next(iter(dataset))
    if isinstance(first_item, tuple):
        if isinstance(first_item[0], dict): # for the case where input is a dict (HF)
            assert ("input_ids" in first_item[0].keys()), f"As the input batch is a dictionnary we expect it to \
                    be a dictionnary as expected by Hugging Face model thus containing 'input_ids'. The dict \
                    keys are {first_item[0].keys()}."
            first_item_inp = first_item[0]['input_ids']
            first_item_lab = first_item[1]
            return tf.shape(first_item_inp)[0].numpy() == batch_size and tf.shape(first_item_lab)[0].numpy() == batch_size
        else:
            return tf.reduce_all(
                [tf.shape(item)[0].numpy() == batch_size for item in first_item]
            )
    return tf.shape(first_item)[0].numpy() == batch_size


def sanitize_dataset(
    dataset: Union[tf.data.Dataset, tf.Tensor, np.array],
    batch_size: int,
    cardinality: Optional[int] = None,
) -> Optional[tf.data.Dataset]:
    """
    Function to ensure input dataset match expected format.
    It also transforms tensors in `tf.data.Dataset` and also verify the properties.
    This function verify that datasets do not reshuffle at each iteration and
    that their batch isze and cardinality match the expected ones.
    Note that, that Tensorflow do not provide easy way to make those tests, hence,
    for cost constraints, our tests are not perfect.

    Parameters
    ----------
    dataset
        Tensorflow dataset to verify or tensor to transform in `tf.data.Dataset` and verify.
    batch_size
        The expected batch size used either to verify the input dataset
        or batch the transformed tensor.
    cardinality
        Expected number of batch in the dataset or batched transformed tensor.

    Returns
    -------
    dataset
        Verified dataset or transformed tensor. In both case a `tf.data.Dataset`,
        that does not reshuffle at each iteration and
        with batch size and cardinality matching the expected ones.
    """
    if dataset is not None:
        if isinstance(dataset, tf.data.Dataset):
            assert is_not_shuffled(dataset), (
                "Datasets should not be shuffled, "
                + "the order of the element should stay the same at each iteration."
            )
            assert batch_size_matches(
                dataset, batch_size
            ), "The batch size should match between datasets."
        else:
            dataset = tf.data.Dataset.from_tensor_slices(dataset).batch(batch_size)

        if cardinality is not None and cardinality > 0:
            dataset_cardinality = dataset.cardinality().numpy()
            if dataset_cardinality > 0:
                assert dataset_cardinality == cardinality, (
                    "The number of batch should match between datasets. "
                    + f"Received {dataset.cardinality().numpy()} vs {cardinality}. "
                    + "You may have provided non-batched datasets or datasets with different length."
                )

    return dataset


def dataset_gather(dataset: tf.data.Dataset, indices: tf.Tensor) -> tf.Tensor:
    """
    Imitation of `tf.gather` for `tf.data.Dataset`,
    it extract elements from `dataset` at the given indices.
    We could see it as returning the `indices` tensor
    where each index was replaced by the corresponding element in `dataset`.
    The aim is to use it in the `example_based` module to extract examples form the cases dataset.
    Hence, `indices` expect dimensions of (n, k, 2),
    where n represent the number of inputs and k the number of corresponding examples.
    Here indices for each element are encoded by two values,
    the batch index and the index of the element in the batch.

    Example of application
    ```
    >>> dataset = tf.data.Dataset.from_tensor_slices(
    ...     tf.reshape(tf.range(20), (-1, 2, 2))
    ... ).batch(3)  # shape=(None, 2, 2)
    >>> indices = tf.constant([[[0, 0]], [[1, 0]]])  # shape=(2, 1, 2)
    >>> dataset_gather(dataset, indices)
    <tf.Variable 'Variable:0' shape=(2, 1, 2, 2) dtype=int32, numpy=
    array([[[[ 0,  1],
            [ 2,  3]]],
        [[[12, 13],
            [14, 15]]]])>
    ```

    Parameters
    ----------
    dataset
        Tensorflow dataset to verify or tensor to transform in `tf.data.Dataset` and verify.
    indices
        Tensor of indices of elements to extract from the `dataset`.
        `indices` should be of dimensions (n, k, 2),
        this is to match the format of indices in the `example_based` module.
        Indeed, n represent the number of inputs and k the number of corresponding examples.
        The index of each element is encoded by two values,
        the batch index and the index of the element in the batch.

    Returns
    -------
    results

    indices should be (n, k, 2)
    """
    if dataset is None:
        return None

    example = next(iter(dataset))
    if isinstance(example, dict):
        assert ("input_ids" in example.keys()), f"As the input batch is a dictionnary we expect it to \
                be a dictionnary as expected by Hugging Face model thus containing 'input_ids'. The dict \
                keys are {example.keys()}."
        results = {
            "input_ids": tf.Variable(
                tf.zeros(
                    indices.shape[:-1] + example["input_ids"][0].shape,
                    dtype=dataset.element_spec["input_ids"].dtype,
                )
            ),
            "attention_mask": tf.Variable(
                tf.zeros(
                    indices.shape[:-1] + example["attention_mask"][0].shape,
                    dtype=dataset.element_spec["attention_mask"].dtype,
                )
            ),
        }
    else:
        # (n, bs, ...)
        results = tf.Variable(
            tf.zeros(
                indices.shape[:-1] + example[0].shape, dtype=dataset.element_spec.dtype
            )
        )

    nb_results = product(indices.shape[:-1])
    current_nb_results = 0

    for i, batch in enumerate(dataset):
        # check if the batch is interesting
        if not tf.reduce_any(indices[..., 0] == i):
            continue

        # extract pertinent elements
        pertinent_indices_location = tf.where(indices[..., 0] == i)
        samples_index = tf.gather_nd(indices[..., 1], pertinent_indices_location)
        if isinstance(example, dict):
            samples = {
                "input_ids": tf.gather(batch["input_ids"], samples_index),
                "attention_mask": tf.gather(batch["attention_mask"], samples_index),
            }
            # put them at the right place in results
            for location, sample_id, sample_mask in zip(pertinent_indices_location, samples["input_ids"], samples["attention_mask"]):
                results["input_ids"][location[0], location[1]].assign(sample_id)
                results["attention_mask"][location[0], location[1]].assign(
                    sample_mask
                )
                current_nb_results += 1
        else:
            samples = tf.gather(batch, samples_index)
            # put them at the right place in results
            for location, sample in zip(pertinent_indices_location, samples):
                results[location[0], location[1]].assign(sample)
                current_nb_results += 1

        # test if results are filled to break the loop
        if current_nb_results == nb_results:
            break
    return results
