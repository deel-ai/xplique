"""
Test the different search methods.
"""
import pytest
import numpy as np
import tensorflow as tf

from xplique.example_based.search_methods import BaseKNN, KNN, FilterKNN, ORDER

def get_setup(input_shape, nb_samples=10, nb_labels=10):
    """
    Generate data and model for SimilarExamples
    """
    # Data generation
    x_train = tf.stack(
        [i * tf.ones(input_shape, tf.float32) for i in range(nb_samples)]
    )
    x_test = x_train[1:-1]
    y_train = tf.range(len(x_train), dtype=tf.float32) % nb_labels

    return x_train, x_test, y_train

class MockKNN(BaseKNN):
    """
    Mock KNN class for testing the find_examples method
    """
    def kneighbors(self, inputs, targets):
        """
        Define a mock kneighbors method for testing the find_examples method of
        the base class.
        """
        best_distances = tf.random.normal((inputs.shape[0], self.k), dtype=tf.float32)
        best_indices= tf.random.uniform((inputs.shape[0], self.k, 2), maxval=self.k, dtype=tf.int32)
        return best_distances, best_indices

def same_target_filter(inputs, cases, targets, cases_targets):
    """
    Filter function that returns a boolean mask with true when point-wise inputs and cases
    have the same target.
    """
    # get the labels predicted by the model
    # (n, )
    predicted_labels = tf.argmax(targets, axis=-1)

    # for each input, if the target label is the same as the predicted label
    # the mask as a True value and False otherwise
    label_targets = tf.argmax(cases_targets, axis=-1) # (bs,)
    mask = tf.equal(tf.expand_dims(predicted_labels, axis=1), label_targets) #(n, bs)
    return mask

def test_base_init():
    """
    Test the initialization of the base KNN class (not the super).
    Check if it raises the relevant errors when the input is invalid.
    """
    base_knn = MockKNN(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
        k=2,
        search_returns='distances',
    )
    assert base_knn.order == ORDER.ASCENDING
    assert base_knn.fill_value == np.inf

    # Test with reverse order
    order = ORDER.DESCENDING
    base_knn = MockKNN(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
        k=2,
        search_returns='distances',
        order=order
    )
    assert base_knn.order == order
    assert base_knn.fill_value == -np.inf

    # Test with invalid order
    with pytest.raises(AssertionError):
        base_knn = MockKNN(
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            k=2,
            search_returns='distances',
            order='invalid'
        )

def test_base_find_examples():
    """
    Test the find_examples method of the base KNN class.
    """
    returns = ["examples", "indices", "distances"]
    mock_knn = MockKNN(
        tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]], dtype=tf.float32),
        k = 2,
        search_returns = returns,
    )

    inputs = tf.random.normal((5, 3), dtype=tf.float32)
    return_dict = mock_knn.find_examples(inputs)
    assert set(return_dict.keys()) == set(returns)
    assert return_dict["examples"].shape == (5, 2, 3)
    assert return_dict["indices"].shape == (5, 2, 2)
    assert return_dict["distances"].shape == (5, 2)

    returns = ["examples", "include_inputs"]
    mock_knn = MockKNN(
        tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]], dtype=tf.float32),
        k = 2,
        search_returns = returns,
    )
    return_dict = mock_knn.find_examples(inputs)
    assert return_dict.shape == (5, 3, 3) 

    mock_knn = MockKNN(
        tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]], dtype=tf.float32),
        k = 2,
    )
    return_dict = mock_knn.find_examples(inputs)
    assert return_dict.shape == (5, 2, 3) 

def test_knn_init():
    """
    Test the initialization of the KNN class which are not linked to the super class.
    """
    cases_dataset = tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]], dtype=tf.float32)
    x1 = tf.random.normal((1, 3), dtype=tf.float32)
    x2 = tf.random.normal((3, 3), dtype=tf.float32)

    # Test with distances that are compatible with tf.norm
    distances = ["euclidean", 1, 2, np.inf, 5]
    for distance in distances:
        knn = KNN(
            cases_dataset,
            k=2,
            search_returns='distances',
            distance=distance,
        )
        assert tf.reduce_all(tf.equal(knn.distance_fn(x1, x2), tf.norm(x1 - x2, ord=distance, axis=-1)))
    
    # Test with a custom distance function
    def custom_distance(x1, x2):
        return tf.reduce_sum(tf.abs(x1 - x2), axis=-1)
    knn = KNN(
        cases_dataset,
        k=2,
        search_returns='distances',
        distance=custom_distance,
    )
    assert tf.reduce_all(tf.equal(knn.distance_fn(x1, x2), custom_distance(x1, x2)))

    # Test with invalid distance
    invalid_distances = [None, "invalid", 0.5]
    for distance in invalid_distances:
        with pytest.raises(AttributeError):
            knn = KNN(
                cases_dataset,
                k=2,
                search_returns='distances',
                distance=distance,
            )

def test_knn_compute_distances():
    """
    Test the private method _compute_distances_fn of the KNN class.
    """
    # Test with input and cases being 1D
    knn = KNN(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
        k=2,
        distance='euclidean',
        order=ORDER.ASCENDING
    )
    x1 = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=tf.float32)
    x2 = tf.constant([[7.0, 8.0], [9.0, 10.0]], dtype=tf.float32)

    expected_distance = tf.constant(
        [
            [np.sqrt(72), np.sqrt(128)],
            [np.sqrt(32), np.sqrt(72)],
            [np.sqrt(8), np.sqrt(32)]
        ], dtype=tf.float32
    )

    distances = knn._crossed_distances_fn(x1, x2)
    assert distances.shape == (x1.shape[0], x2.shape[0])
    assert tf.reduce_all(tf.equal(distances, expected_distance))

    # Test with higher dimensions
    data = np.array([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[10, 11, 12], [13, 14, 15], [16, 17, 18]]
    ])

    knn = KNN(
        data,
        k=2,
        distance="euclidean",
        order=ORDER.ASCENDING
    )

    x1 = tf.constant(
        [
            [[1, 2, 3],[4, 5, 6],[7, 8, 9]],
            [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
            [[19, 20, 21], [22, 23, 24], [25, 26, 27]]
        ], dtype=tf.float32
    )

    x2 = tf.constant(
        [
            [[28, 29, 30], [31, 32, 33], [34, 35, 36]],
            [[37, 38, 39], [40, 41, 42], [43, 44, 45]],
        ], dtype=tf.float32
    )

    expected_distance = tf.constant(
        [[np.sqrt(9)*27, np.sqrt(9)*36],
         [np.sqrt(9)*18, np.sqrt(9)*27],
         [np.sqrt(9)*9, np.sqrt(9)*18]], dtype=tf.float32)
    
    distances = knn._crossed_distances_fn(x1, x2)
    assert distances.shape == (x1.shape[0], x2.shape[0])
    assert tf.reduce_all(tf.equal(distances, expected_distance))
    

def test_knn_kneighbors():
    """
    Test the kneighbors method of the KNN class.
    """
    # Test with input and cases being 1D
    cases = tf.constant([[1.], [2.], [3.], [4.], [5.]], dtype=tf.float32)
    inputs = tf.constant([[1.5], [2.5], [4.5]], dtype=tf.float32)
    knn = KNN(
        cases,
        k=2,
        batch_size=2,
        distance="euclidean",
    )

    distances, indices = knn.kneighbors(inputs)
    assert distances.shape == (3, 2)
    assert indices.shape == (3, 2, 2)
    assert tf.reduce_all(tf.equal(distances, tf.constant([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=tf.float32)))
    assert tf.reduce_all(tf.equal(indices, tf.constant([[[0, 0], [0, 1]],[[0, 1], [1, 0]],[[1, 1], [2, 0]]], dtype=tf.int32)))

    # Test with reverse order
    knn = KNN(
        cases,
        k=2,
        batch_size=2,
        distance="euclidean",
        order=ORDER.DESCENDING
    )

    distances, indices = knn.kneighbors(inputs)
    assert distances.shape == (3, 2)
    assert indices.shape == (3, 2, 2)
    assert tf.reduce_all(tf.equal(distances, tf.constant([[3.5, 2.5], [2.5, 1.5], [3.5, 2.5]], dtype=tf.float32)))
    assert tf.reduce_all(tf.equal(indices, tf.constant([[[2, 0], [1, 1]],[[2, 0], [0, 0]],[[0, 0], [0, 1]]], dtype=tf.int32)))

    # Test with input and cases being 2D
    cases = tf.constant([[1., 2.], [2., 3.], [3., 4.], [4., 5.], [5., 6.]], dtype=tf.float32)
    inputs = tf.constant([[1.5, 2.5], [2.5, 3.5], [4.5, 5.5]], dtype=tf.float32)
    knn = KNN(
        cases,
        k=2,
        batch_size=2,
        distance="euclidean",
    )

    distances, indices = knn.kneighbors(inputs)
    assert distances.shape == (3, 2)
    assert indices.shape == (3, 2, 2)
    assert tf.reduce_all(tf.equal(distances, tf.constant([[np.sqrt(0.5), np.sqrt(0.5)], [np.sqrt(0.5), np.sqrt(0.5)], [np.sqrt(0.5), np.sqrt(0.5)]], dtype=tf.float32)))
    assert tf.reduce_all(tf.equal(indices, tf.constant([[[0, 0], [0, 1]],[[0, 1], [1, 0]],[[1, 1], [2, 0]]], dtype=tf.int32)))

    # Test with reverse order
    knn = KNN(
        cases,
        k=2,
        batch_size=2,
        distance="euclidean",
        order=ORDER.DESCENDING
    )

    distances, indices = knn.kneighbors(inputs)
    assert distances.shape == (3, 2)
    assert indices.shape == (3, 2, 2)
    expected_distances = tf.constant([[np.sqrt(2*3.5**2), np.sqrt(2*2.5**2)], [np.sqrt(2*2.5**2), np.sqrt(2*1.5**2)], [np.sqrt(2*3.5**2), np.sqrt(2*2.5**2)]], dtype=tf.float32)
    assert tf.reduce_all(tf.abs(distances - expected_distances) < 1e-5)
    assert tf.reduce_all(tf.equal(indices, tf.constant([[[2, 0], [1, 1]],[[2, 0], [0, 0]],[[0, 0], [0, 1]]], dtype=tf.int32)))

def test_filter_knn_compute_distances():
    """
    Test the private method _compute_distances_fn of the FilterKNN class.
    """
    # Test in Low dimension
    knn = FilterKNN(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
        k=2,
        distance='euclidean',
        order=ORDER.ASCENDING
    )
    x1 = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=tf.float32)
    x2 = tf.constant([[7.0, 8.0], [9.0, 10.0]], dtype=tf.float32)
    expected_distance = tf.constant(
        [
            [np.sqrt(72), np.sqrt(128)],
            [np.sqrt(32), np.sqrt(72)],
            [np.sqrt(8), np.sqrt(32)]
        ], dtype=tf.float32
    )
    mask = tf.ones((x1.shape[0], x2.shape[0]), dtype=tf.bool)
    distances = knn._crossed_distances_fn(x1, x2, mask)
    assert distances.shape == (x1.shape[0], x2.shape[0])
    assert tf.reduce_all(tf.equal(distances, expected_distance))

    mask = tf.constant([[True, False], [False, True], [True, True]], dtype=tf.bool)
    expected_distance = tf.constant([[np.sqrt(72), np.inf], [np.inf, np.sqrt(72)], [np.sqrt(8), np.sqrt(32)]], dtype=tf.float32)
    distances = knn._crossed_distances_fn(x1, x2, mask)
    assert tf.reduce_all(tf.equal(distances, expected_distance))

    # Test with higher dimensions
    data = np.array([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[10, 11, 12], [13, 14, 15], [16, 17, 18]]
    ])

    knn = FilterKNN(
        data,
        k=2,
        distance="euclidean",
        order=ORDER.ASCENDING
    )

    x1 = tf.constant(
        [
            [[1, 2, 3],[4, 5, 6],[7, 8, 9]],
            [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
            [[19, 20, 21], [22, 23, 24], [25, 26, 27]]
        ], dtype=tf.float32
    )

    x2 = tf.constant(
        [
            [[28, 29, 30], [31, 32, 33], [34, 35, 36]],
            [[37, 38, 39], [40, 41, 42], [43, 44, 45]],
        ], dtype=tf.float32
    )

    expected_distance = tf.constant(
        [[np.sqrt(9)*27, np.sqrt(9)*36],
         [np.sqrt(9)*18, np.sqrt(9)*27],
         [np.sqrt(9)*9, np.sqrt(9)*18]], dtype=tf.float32)
    
    mask = tf.ones((x1.shape[0], x2.shape[0]), dtype=tf.bool)
    distances = knn._crossed_distances_fn(x1, x2, mask)
    assert distances.shape == (x1.shape[0], x2.shape[0])
    assert tf.reduce_all(tf.equal(distances, expected_distance))

    mask = tf.constant([[True, False], [False, True], [True, True]], dtype=tf.bool)
    expected_distance = tf.constant([[np.sqrt(9)*27, np.inf], [np.inf, np.sqrt(9)*27], [np.sqrt(9)*9, np.sqrt(9)*18]], dtype=tf.float32)
    distances = knn._crossed_distances_fn(x1, x2, mask)
    assert distances.shape == (x1.shape[0], x2.shape[0])
    assert tf.reduce_all(tf.equal(distances, expected_distance))

def test_filter_knn_kneighbors():
    """
    """
    # Test with input and cases being 1D
    cases = tf.constant([[1.], [2.], [3.], [4.], [5.]], dtype=tf.float32)
    inputs = tf.constant([[1.5], [2.5], [4.5]], dtype=tf.float32)
    ## default filter and default order
    knn = KNN(
        cases,
        k=2,
        batch_size=2,
        distance="euclidean",
    )

    distances, indices = knn.kneighbors(inputs)
    assert distances.shape == (3, 2)
    assert indices.shape == (3, 2, 2)
    assert tf.reduce_all(tf.equal(distances, tf.constant([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=tf.float32)))
    assert tf.reduce_all(tf.equal(indices, tf.constant([[[0, 0], [0, 1]],[[0, 1], [1, 0]],[[1, 1], [2, 0]]], dtype=tf.int32)))

    cases_targets = tf.constant([[0, 1], [1, 0], [1, 0], [0, 1], [1, 0]], dtype=tf.float32)
    targets = tf.constant([[0, 1], [1, 0], [1, 0]], dtype=tf.float32)

    ## add a filter that is not the default
    knn = FilterKNN(
        cases,
        targets_dataset=cases_targets,
        k=2,
        batch_size=2,
        distance="euclidean",
        filter_fn=same_target_filter
    )
    mask = same_target_filter(inputs, cases, targets, cases_targets)
    print(mask)
    distances, indices = knn.kneighbors(inputs, targets)
    assert distances.shape == (3, 2)
    assert indices.shape == (3, 2, 2)
    assert tf.reduce_all(tf.equal(distances, tf.constant([[0.5, 2.5], [0.5, 0.5], [0.5, 1.5]], dtype=tf.float32)))
    assert tf.reduce_all(tf.equal(indices, tf.constant([[[0, 0], [1, 1]],[[0, 1], [1, 0]],[[2, 0], [1, 0]]], dtype=tf.int32)))

    ## test with reverse order
    knn = FilterKNN(
        cases,
        k=2,
        batch_size=2,
        distance="euclidean",
        order=ORDER.DESCENDING
    )

    distances, indices = knn.kneighbors(inputs, targets)
    assert distances.shape == (3, 2)
    assert indices.shape == (3, 2, 2)
    expected_distances = tf.constant([[3.5, 2.5], [2.5, 1.5], [3.5, 2.5]], dtype=tf.float32)
    assert tf.reduce_all(tf.equal(distances, expected_distances))
    assert tf.reduce_all(tf.equal(indices, tf.constant([[[2, 0], [1, 1]],[[2, 0], [0, 0]],[[0, 0], [0, 1]]], dtype=tf.int32)))
  
    ## add a filter that is not the default one and reverse order
    knn = FilterKNN(
        cases,
        targets_dataset=cases_targets,
        k=2,
        batch_size=2,
        distance="euclidean",
        order=ORDER.DESCENDING,
        filter_fn=same_target_filter
    )

    distances, indices = knn.kneighbors(inputs, targets)
    assert distances.shape == (3, 2)
    assert indices.shape == (3, 2, 2)
    assert tf.reduce_all(tf.equal(distances, tf.constant([[2.5, 0.5], [2.5, 0.5], [2.5, 1.5]], dtype=tf.float32)))
    assert tf.reduce_all(tf.equal(indices, tf.constant([[[1, 1], [0, 0]],[[2, 0], [0, 1]],[[0, 1], [1, 0]]], dtype=tf.int32)))

    # Test with input and cases being 2D
    cases = tf.constant([[1., 2.], [2., 3.], [3., 4.], [4., 5.], [5., 6.]], dtype=tf.float32)
    inputs = tf.constant([[1.5, 2.5], [2.5, 3.5], [4.5, 5.5]], dtype=tf.float32)
    ## default filter and default order
    knn = FilterKNN(
        cases,
        k=2,
        batch_size=2,
        distance="euclidean",
    )

    distances, indices = knn.kneighbors(inputs)
    assert distances.shape == (3, 2)
    assert indices.shape == (3, 2, 2)
    assert tf.reduce_all(tf.equal(distances, tf.constant([[np.sqrt(0.5), np.sqrt(0.5)], [np.sqrt(0.5), np.sqrt(0.5)], [np.sqrt(0.5), np.sqrt(0.5)]], dtype=tf.float32)))
    assert tf.reduce_all(tf.equal(indices, tf.constant([[[0, 0], [0, 1]],[[0, 1], [1, 0]],[[1, 1], [2, 0]]], dtype=tf.int32)))

    cases_targets = tf.constant([[0, 1], [1, 0], [1, 0], [0, 1], [1, 0]], dtype=tf.float32)
    targets = tf.constant([[0, 1], [1, 0], [1, 0]], dtype=tf.float32)
    ## add a filter that is not the default
    knn = FilterKNN(
        cases,
        targets_dataset=cases_targets,
        k=2,
        batch_size=2,
        distance="euclidean",
        filter_fn=same_target_filter
    )

    distances, indices = knn.kneighbors(inputs, targets)
    assert distances.shape == (3, 2)
    assert indices.shape == (3, 2, 2)
    expected_distances = tf.constant([[np.sqrt(0.5), np.sqrt(2*2.5**2)], [np.sqrt(0.5), np.sqrt(0.5)], [np.sqrt(0.5), np.sqrt(2*1.5**2)],], dtype=tf.float32)
    assert tf.reduce_all(tf.abs(distances - expected_distances) < 1e-5)
    assert tf.reduce_all(tf.equal(indices, tf.constant([[[0, 0], [1, 1]],[[0, 1], [1, 0]],[[2, 0], [1, 0]]], dtype=tf.int32)))
    
    ## test with reverse order and default filter
    knn = FilterKNN(
        cases,
        k=2,
        batch_size=2,
        distance="euclidean",
        order=ORDER.DESCENDING
    )

    distances, indices = knn.kneighbors(inputs)
    assert distances.shape == (3, 2)
    assert indices.shape == (3, 2, 2)
    expected_distances = tf.constant([[np.sqrt(2*3.5**2), np.sqrt(2*2.5**2)], [np.sqrt(2*2.5**2), np.sqrt(2*1.5**2)], [np.sqrt(2*3.5**2), np.sqrt(2*2.5**2)]], dtype=tf.float32)
    assert tf.reduce_all(tf.abs(distances - expected_distances) < 1e-5)
    assert tf.reduce_all(tf.equal(indices, tf.constant([[[2, 0], [1, 1]],[[2, 0], [0, 0]],[[0, 0], [0, 1]]], dtype=tf.int32)))

    ## add a filter that is not the default one and reverse order
    knn = FilterKNN(
        cases,
        targets_dataset=cases_targets,
        k=2,
        batch_size=2,
        distance="euclidean",
        order=ORDER.DESCENDING,
        filter_fn=same_target_filter
    )

    distances, indices = knn.kneighbors(inputs, targets)
    assert distances.shape == (3, 2)
    assert indices.shape == (3, 2, 2)
    expected_distances = tf.constant([[np.sqrt(2*2.5**2), np.sqrt(0.5)], [np.sqrt(2*2.5**2), np.sqrt(0.5)], [np.sqrt(2*2.5**2), np.sqrt(2*1.5**2)]], dtype=tf.float32)
    assert tf.reduce_all(tf.abs(distances - expected_distances) < 1e-5)
    assert tf.reduce_all(tf.equal(indices, tf.constant([[[1, 1], [0, 0]],[[2, 0], [0, 1]],[[0, 1], [1, 0]]], dtype=tf.int32)))
