"""
Stochastic transformations
"""

import tensorflow as tf


def random_blur(kernel_size=10, sigma_range=(1.0, 2.0)):
    """
    Generate a function that apply a random gaussian blur to the batch.

    Parameters
    ----------
    kernel_size : int, optional
        Size of the gaussian kernel
    sigma_range : tuple, optional
        Min and max sigma (or scale) of the gaussian kernel.

    Returns
    -------
    blur : function
        Transformation function applying random blur.
    """
    uniform = tf.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2.,
                          kernel_size)
    uniform_xx, uniform_yy = tf.meshgrid(uniform, uniform)

    kernel_size = tf.cast(kernel_size, tf.float32)
    sigma_min = tf.cast(max(sigma_range[0], 0.1), tf.float32)
    sigma_max = tf.cast(max(sigma_range[1], 0.1), tf.float32)

    def blur(images):
        sigma = tf.random.uniform([], minval=sigma_min, maxval=sigma_max,
                                  dtype=tf.float32)

        kernel = tf.exp(-0.5 * (uniform_xx ** 2 + uniform_yy ** 2) / sigma ** 2)
        kernel /= tf.reduce_sum(kernel)

        kernel = tf.reshape(kernel, (kernel_size, kernel_size, 1, 1))
        kernel = tf.tile(kernel, [1, 1, 3, 1])

        return tf.nn.depthwise_conv2d(images, kernel, strides=[1, 1, 1, 1],
                                      padding='SAME')

    return blur


def random_jitter(delta=6):
    """
    Generate a function that perform a random jitter on batch of images.

    Parameters
    ----------
    delta : int, optional
        Max of the shift

    Returns
    -------
    jitter : function
        Transformation function applying random jitter.

    """

    def jitter(images):
        shape = tf.shape(images)
        images = tf.image.random_crop(images, (shape[0], shape[1] - delta, shape[2] - delta,
                                               shape[-1]))
        return images

    return jitter


def random_scale(scale_range=(0.95, 1.05)):
    """
    Generate a function that apply a random scaling to the batch. Preserve
    aspect ratio.

    Parameters
    ----------
    scale_range : tuple, optional
        Min and max scaling factor.

    Returns
    -------
    scale : function
        Transformation function applying random scaling.
    """
    min_scale = tf.cast(scale_range[0], tf.float32)
    max_scale = tf.cast(scale_range[1], tf.float32)

    def scale(images):
        _, width, height, _ = images.shape
        scale_factor = tf.random.uniform([], minval=min_scale, maxval=max_scale,
                                         dtype=tf.float32)
        return tf.image.resize(images, tf.cast([width * scale_factor,
                                                height * scale_factor], tf.int32))

    return scale


def random_flip(horizontal=True, vertical=False):
    """
    Generate a function that apply random flip (horizontal/vertical) to the
    batch.

    Parameters
    ----------
    horizontal : bool,
        Whether to perform random horizontal flipping (left/right)
    vertical : bool,
        Whether to perform random vertical flipping (top/down)

    Returns
    -------
    flip : function
        Transformation function applying random flipping.
    """

    def flip(images):
        if horizontal:
            images = tf.image.random_flip_left_right(images)
        if vertical:
            images = tf.image.random_flip_up_down(images)
        return images

    return flip


def pad(size=6, pad_value=0.0):
    """
    Generate a function that apply padding to a batch of images.

    Parameters
    ----------
    size : int, optional
        Size of the padding
    pad_value : float, optional
        Value of the padded pixels

    Returns
    -------
    pad_func : function
        Transformation function applying padding.
    """
    pad_array = [(0, 0), (size, size), (size, size), (0, 0)]
    pad_value = tf.cast(pad_value, tf.float32)

    def pad_func(images):
        return tf.pad(images, pad_array, mode="CONSTANT", constant_values=pad_value)

    return pad_func


def compose_transformations(transformations):
    """
    Return a function that combine all the transformations passed and resize
    the images at the end.

    Parameters
    ----------
    transformations : list
        List of transformations, like the one in this module.

    Returns
    -------
    composed_func : function
        The combinations of the functions passed and a resize.
    """

    def composed_func(images):
        for func in transformations:
            images = func(images)
        return images

    return composed_func


standard_transformations = compose_transformations([
    pad(24, 0.0),
    random_jitter(6),
    random_jitter(6),
    random_jitter(12),
    random_jitter(12),
    random_scale([0.95, 0.99]),
    random_jitter(12),
    random_jitter(12),
    random_jitter(12),
    random_jitter(12),
    random_jitter(12),
])
