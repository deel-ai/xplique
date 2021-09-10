"""
Tests for Lime Module
"""
import numpy as np
import tensorflow as tf
from sklearn import linear_model
from xplique.attributions import Lime
from ..utils import generate_data, generate_model, almost_equal


def test_pertub_func():
    """Ensure the default pertub function has the right output"""
    num_features = 10
    pertub_func = Lime._get_default_pertub_function(prob = 0.5)
    samples = pertub_func(num_features, 12)

    assert samples.shape == (12,10)
    assert samples.dtype == tf.int32


def test_get_masks():
    """
    Ensure the get_masks function behave as expected (shape,type) and
    return correct values on toy examples
    """
    # say we have a mapping grouping an image by 2x2 pixels
    mapping_to_interpret_space = tf.constant([[0,0,1,1],[0,0,1,1],[2,2,3,3],[2,2,3,3]])
    # have four interpretation
    # first is the one representing the upper right of the image
    interpret_sample1 = [0,1,0,0]
    # second is the one representing the bottom left of the image
    interpret_sample2 = [0,0,1,0]
    # third contain the upper left and bottom right of the image
    interpret_sample3 = [1,0,0,1]
    # last one just miss the upper right
    interpret_sample4 = [1,0,1,1]

    interpret_samples = tf.constant([interpret_sample1,
                                     interpret_sample2,
                                     interpret_sample3,
                                     interpret_sample4])

    masks = Lime._get_masks(interpret_samples,mapping_to_interpret_space)
    assert masks.shape == (4,4,4)
    assert masks.dtype == tf.int32
    masks.numpy()
    expected_mask1 = np.array([[0,0,1,1],[0,0,1,1],[0,0,0,0],[0,0,0,0]])
    assert np.array_equal(masks[0],expected_mask1)

    expected_mask2 = np.array([[0,0,0,0],[0,0,0,0],[1,1,0,0],[1,1,0,0]])
    assert np.array_equal(masks[1],expected_mask2)

    expected_mask3 = np.array([[1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]])
    assert np.array_equal(masks[2],expected_mask3)

    expected_mask4 = np.array([[1,1,0,0],[1,1,0,0],[1,1,1,1],[1,1,1,1]])
    assert np.array_equal(masks[3],expected_mask4)

def test_apply_masks():
    """
    Ensure the apply_masks function behave as expected (shape,type) and
    work correctly on toy examples
    """
    # have a real artificial image to see if masks applies correctly
    img = tf.ones((2,2,3))
    # define 2 easy mask
    # we grayed all pixels excepts for the upper right pixel
    mask1 = [[0,1],[0,0]]
    # we grayed only the upper right pixel
    mask2 = [[1,0],[1,1]]
    # define the RGB value for pixel which will be masked
    baseline = tf.constant([125.0,255.0,0.0])

    masks = tf.constant([mask1,mask2])
    samples = Lime._apply_masks(img,masks,baseline)
    assert samples.shape == (2,2,2,3)
    assert samples.dtype == tf.float32

    samples.numpy()
    expected_sample1 = np.array([[[125,255,0],[1,1,1]],[[125,255,0],[125,255,0]]])
    assert np.array_equal(samples[0],expected_sample1)

    expected_sample2 = np.array([[[1,1,1],[125,255,0]],[[1,1,1],[1,1,1]]])
    assert np.array_equal(samples[1],expected_sample2)

def test_similarities():
    """
    Ensure that the compute similarity function behave as expected (shape, type)
    and return expected values on toy examples
    """
    original_input = tf.ones((2,2,3),dtype=tf.float32)
    mapping = tf.constant([[0,1],[2,3]], dtype=tf.int32)
    ref_value = tf.zeros(3, dtype=tf.float32)

    int_sample1 = np.array([1, 0, 0, 1])
    int_sample2 = np.array([0, 1, 0, 1])
    int_sample3 = np.array([0, 1, 0, 0])
    int_sample4 = np.array([0, 1, 1, 1])

    int_samples = tf.constant(
        [int_sample1, int_sample2, int_sample3, int_sample4],
        dtype=tf.int32
    )
    masks = Lime._get_masks(int_samples, mapping)
    pertubed_samples = Lime._apply_masks(original_input, masks, ref_value)

    similarity_kernel = Lime._get_exp_kernel_func()
    similarities = similarity_kernel(original_input, int_samples, pertubed_samples)

    assert similarities.shape == 4
    assert similarities.dtype == tf.float32

    expected_outcome = np.array([np.exp(-6),np.exp(-6),np.exp(-9),np.exp(-3)])

    similarities = similarities.numpy()
    assert almost_equal(similarities,expected_outcome)

    similarity_kernel2 = Lime._get_exp_kernel_func(distance_mode='cosine')
    similarities2 = similarity_kernel2(original_input, int_samples, pertubed_samples)

    assert similarities2.shape == 4
    assert similarities2.dtype == tf.float32

def test_compute():
    """The output shape must be the same as the input shape, except for the channels"""
    input_shapes = [(28, 28, 1), (32, 32, 3)]
    nb_labels = 10

    def map_four_by_four(inp):

        width = inp.shape[0]
        height = inp.shape[1]

        mapping = np.zeros((width,height))
        for i in range(width):
            if i%2 != 0:
                mapping[i] = mapping[i-1]
            else:
                for j in range(height):
                    mapping[i][j] = (width/2) * (i//2) + (j//2)

        mapping = tf.cast(mapping, dtype=tf.int32)
        return mapping

    for input_shape in input_shapes:
        samples, labels = generate_data(input_shape, nb_labels, 20)
        model = generate_model(input_shape, nb_labels)

        method = Lime(model,
                      interpretable_model=linear_model.Lasso(alpha=0.1),
                      map_to_interpret_space=map_four_by_four,
                      nb_samples=10,
                      kernel_width=10)

        explanations = method.explain(samples, labels)
        assert samples.shape[:3] == explanations.shape

def test_inputs_batching():
    """ Ensure, that we can call explain with batched inputs """
    nb_labels = 10

    samples, labels = generate_data((32, 32, 3), nb_labels, 200)
    model = generate_model((32, 32, 3), nb_labels)

    method = Lime(model,
                    batch_size=15,
                    interpretable_model=linear_model.Ridge(alpha=0.1),
                    nb_samples=20,
                    kernel_width=10)

    explanations = method.explain(samples, labels)
    assert samples.shape[:3] == explanations.shape
