"""
Tests for the contrastive methods.
"""
import tensorflow as tf
import numpy as np

from xplique.example_based.search_methods import KLEORSimMissSearch, KLEORGlobalSimSearch

def test_kleor_base_and_sim_miss():
    """
    Test suite for both the BaseKLEOR and KLEORSimMiss class. Indeed, the KLEORSimMiss class is a subclass of the
    BaseKLEOR class with a very basic implementation of the only abstract method (identity function).
    """
    # setup the tests
    cases = tf.constant([[1., 2.], [2., 3.], [3., 4.], [4., 5.], [5., 6.]], dtype=tf.float32)
    cases_targets = tf.constant([[0, 1], [1, 0], [1, 0], [0, 1], [1, 0]], dtype=tf.float32)

    cases_dataset = tf.data.Dataset.from_tensor_slices(cases).batch(2)
    cases_targets_dataset = tf.data.Dataset.from_tensor_slices(cases_targets).batch(2)
    
    inputs = tf.constant([[1.5, 2.5], [2.5, 3.5], [4.5, 5.5]], dtype=tf.float32)
    targets = tf.constant([[0, 1], [1, 0], [1, 0]], dtype=tf.float32)

    # build the kleor object
    kleor = KLEORSimMissSearch(cases_dataset, cases_targets_dataset, k=1, search_returns=["examples", "indices", "distances", "include_inputs", "nuns"], batch_size=2)

    # test the _filter_fn method
    fake_targets = tf.constant([[0, 1], [1, 0], [1, 0], [0, 1], [1, 0]], dtype=tf.float32)
    fake_cases_targets = tf.constant([[0, 1], [1, 0], [0, 1], [1, 0], [1, 0]], dtype=tf.float32)
    # the mask should be True when the targets are the same i.e we keep those cases
    expected_mask = tf.constant([[True, False, True, False, False],
                                 [False, True, False, True, True],
                                 [False, True, False, True, True],
                                 [True, False, True, False, False],
                                 [False, True, False, True, True]], dtype=tf.bool)
    mask = kleor._filter_fn(inputs, cases, fake_targets, fake_cases_targets)
    assert tf.reduce_all(tf.equal(mask, expected_mask))

    # test the _filter_fn_nun method, this time the mask should be True when the targets are different
    expected_mask = tf.constant([[False, True, False, True, True],
                                 [True, False, True, False, False],
                                 [True, False, True, False, False],
                                 [False, True, False, True, True],
                                 [True, False, True, False, False]], dtype=tf.bool)
    mask = kleor._filter_fn_nun(inputs, cases, fake_targets, fake_cases_targets)
    assert tf.reduce_all(tf.equal(mask, expected_mask))

    # test the _get_nuns method
    nuns, _, nuns_distances = kleor._get_nuns(inputs, targets)
    expected_nuns = tf.constant([
        [[2., 3.]],
        [[1., 2.]],
        [[4., 5.]]], dtype=tf.float32)
    expected_nuns_distances = tf.constant([
        [np.sqrt(2*0.5**2)],
        [np.sqrt(2*1.5**2)],
        [np.sqrt(2*0.5**2)]], dtype=tf.float32)
    assert tf.reduce_all(tf.equal(nuns, expected_nuns))
    assert tf.reduce_all(tf.equal(nuns_distances, expected_nuns_distances))

    # test the _initialize_search method
    sf_indices, input_sf_distances, nun_sf_distances, batch_indices = kleor._initialize_search(inputs)
    assert sf_indices.shape == (3, 1, 2) # (n, k, 2)
    assert input_sf_distances.shape == (3, 1) # (n, k)
    assert nun_sf_distances.shape == (3, 1) # (n, k)
    assert batch_indices.shape == (3, 2) # (n, bs)
    expected_sf_indices = tf.constant([[[-1, -1]],[[-1, -1]],[[-1, -1]]], dtype=tf.int32)
    assert tf.reduce_all(tf.equal(sf_indices, expected_sf_indices))
    assert tf.reduce_all(tf.math.is_inf(input_sf_distances))
    assert tf.reduce_all(tf.math.is_inf(nun_sf_distances))
    expected_batch_indices = tf.constant([[0, 1], [0, 1], [0, 1]], dtype=tf.int32)
    assert tf.reduce_all(tf.equal(batch_indices, expected_batch_indices))

    # test the kneighbors method
    input_sf_distances, sf_indices, nuns, _, __ = kleor.kneighbors(inputs, targets)

    assert input_sf_distances.shape == (3, 1) # (n, k)
    assert sf_indices.shape == (3, 1, 2) # (n, k, 2)
    assert nuns.shape == (3, 1, 2) # (n, k, 2)

    assert tf.reduce_all(tf.equal(nuns, expected_nuns))

    expected_distances = tf.constant([[np.sqrt(2*0.5**2)], [np.sqrt(2*0.5**2)], [np.sqrt(2*1.5**2)]], dtype=tf.float32)
    assert tf.reduce_all(tf.abs(input_sf_distances - expected_distances) < 1e-5)

    expected_indices = tf.constant([[[0, 0]],[[0, 1]],[[1, 0]]], dtype=tf.int32)
    assert tf.reduce_all(tf.equal(sf_indices, expected_indices))

    # test the find_examples method
    return_dict = kleor.find_examples(inputs, targets)
    assert set(return_dict.keys()) == set(["examples", "indices", "distances", "nuns"])

    examples = return_dict["examples"]
    distances = return_dict["distances"]
    indices = return_dict["indices"]
    nuns = return_dict["nuns"]

    assert tf.reduce_all(tf.equal(nuns, expected_nuns))
    assert tf.reduce_all(tf.equal(expected_indices, indices))
    assert tf.reduce_all(tf.abs(distances - expected_distances) < 1e-5)

    expected_examples = tf.constant([
        [[1.5, 2.5], [1., 2.]],
        [[2.5, 3.5], [2., 3.]],
        [[4.5, 5.5], [3., 4.]]], dtype=tf.float32)
    assert tf.reduce_all(tf.equal(examples, expected_examples))
 
def test_kleor_global_sim():
    """
    Test suite for the KleorGlobalSim class. As only the kneighbors, format_output are impacted by the
    _additionnal_filtering method we test those 3 methods.
    """
    # setup the tests
    cases = tf.constant([[1., 2.], [2., 3.], [3., 4.], [4., 5.], [5., 6.]], dtype=tf.float32)
    cases_targets = tf.constant([[0, 1], [1, 0], [1, 0], [0, 1], [1, 0]], dtype=tf.float32)

    cases_dataset = tf.data.Dataset.from_tensor_slices(cases).batch(2)
    cases_targets_dataset = tf.data.Dataset.from_tensor_slices(cases_targets).batch(2)
    
    inputs = tf.constant([[1.5, 2.5], [2.5, 3.5], [4.5, 5.5]], dtype=tf.float32)
    targets = tf.constant([[0, 1], [1, 0], [1, 0]], dtype=tf.float32)

    # build the kleor object
    kleor = KLEORGlobalSimSearch(cases_dataset, cases_targets_dataset, k=1, search_returns=["examples", "indices", "distances", "include_inputs", "nuns"], batch_size=2)

    # test the _additionnal_filtering method
    # (n, bs)
    fake_nun_sf_distances = tf.constant([[1., 2.], [2., 3.], [3., 4.]])
    # (n, bs)
    fake_input_sf_distances = tf.constant([[2., 1.], [3., 2.], [2., 5.]])
    # (n,1)
    fake_nuns_input_distances = tf.constant([[3.], [1.], [4.]])
    # the expected filtering should be such that we keep the distance of a sf candidates
    # when the input is closer to the sf than the nun, otherwise we set it to infinity
    expected_nun_sf_distances = tf.constant([[1., 2.], [np.inf, np.inf], [3., np.inf]], dtype=tf.float32)
    expected_input_sf_distances = tf.constant([[2., 1.], [np.inf, np.inf], [2., np.inf]], dtype=tf.float32)

    nun_sf_distances, input_sf_distances = kleor._additional_filtering(fake_nun_sf_distances, fake_input_sf_distances, fake_nuns_input_distances)
    assert nun_sf_distances.shape == (3, 2)
    assert input_sf_distances.shape == (3, 2)

    inf_mask_expected_nun_sf = tf.math.is_inf(expected_nun_sf_distances)
    inf_mask_nun_sf = tf.math.is_inf(nun_sf_distances)
    assert tf.reduce_all(tf.equal(inf_mask_expected_nun_sf, inf_mask_nun_sf))
    assert tf.reduce_all(
        tf.abs(tf.where(inf_mask_nun_sf, 0.0, nun_sf_distances) - tf.where(inf_mask_expected_nun_sf, 0.0, expected_nun_sf_distances)
               ) < 1e-5)

    inf_mask_expected_input_sf = tf.math.is_inf(expected_input_sf_distances)
    inf_mask_input_sf = tf.math.is_inf(input_sf_distances)
    assert tf.reduce_all(tf.equal(inf_mask_expected_input_sf, inf_mask_input_sf))
    assert tf.reduce_all(
        tf.abs(tf.where(inf_mask_input_sf, 0.0, input_sf_distances) - tf.where(inf_mask_expected_input_sf, 0.0, expected_input_sf_distances)
               ) < 1e-5)

    # test the kneighbors method
    input_sf_distances, sf_indices, nuns, _, __ = kleor.kneighbors(inputs, targets)

    expected_nuns = tf.constant([
        [[2., 3.]],
        [[1., 2.]],
        [[4., 5.]]], dtype=tf.float32)
    assert tf.reduce_all(tf.equal(nuns, expected_nuns))

    assert input_sf_distances.shape == (3, 1) # (n, k)
    assert sf_indices.shape == (3, 1, 2) # (n, k, 2)

    expected_indices = tf.constant([[[-1, -1]],[[0, 1]],[[-1, -1]]], dtype=tf.int32)
    assert tf.reduce_all(tf.equal(sf_indices, expected_indices))

    expected_distances = tf.constant([[kleor.fill_value], [np.sqrt(2*0.5**2)], [kleor.fill_value]], dtype=tf.float32)
    
    # create masks for inf values
    inf_mask_input = tf.math.is_inf(input_sf_distances)
    inf_mask_expected = tf.math.is_inf(expected_distances)
    assert tf.reduce_all(tf.equal(inf_mask_input, inf_mask_expected))

    # compare finite values
    assert tf.reduce_all(
        tf.abs(tf.where(inf_mask_input, 0.0, input_sf_distances) - tf.where(inf_mask_expected, 0.0, expected_distances)
               ) < 1e-5)
    
    # test the find_examples
    return_dict = kleor.find_examples(inputs, targets)

    indices = return_dict["indices"]
    nuns = return_dict["nuns"]
    distances = return_dict["distances"]
    examples = return_dict["examples"]

    assert tf.reduce_all(tf.equal(nuns, expected_nuns))
    assert tf.reduce_all(tf.equal(expected_indices, indices))

    # create masks for inf values
    inf_mask_dist = tf.math.is_inf(distances)
    assert tf.reduce_all(tf.equal(inf_mask_dist, inf_mask_expected))
    assert tf.reduce_all(
        tf.abs(tf.where(inf_mask_dist, 0.0, distances) - tf.where(inf_mask_expected, 0.0, expected_distances)
               ) < 1e-5)

    expected_examples = tf.constant([
        [[1.5, 2.5], [np.inf, np.inf]],
        [[2.5, 3.5], [2., 3.]],
        [[4.5, 5.5], [np.inf, np.inf]]], dtype=tf.float32)

    # mask for inf values
    inf_mask_examples = tf.math.is_inf(examples)
    inf_mask_expected_examples = tf.math.is_inf(expected_examples)
    assert tf.reduce_all(tf.equal(inf_mask_examples, inf_mask_expected_examples))
    assert tf.reduce_all(
        tf.abs(tf.where(inf_mask_examples, 0.0, examples) - tf.where(inf_mask_expected_examples, 0.0, expected_examples)
               ) < 1e-5)
