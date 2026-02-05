import numpy as np
import tensorflow as tf

from xplique.features_visualizations import (
    Objective,
    init_maco_buffer,
    maco,
    maco_image_parametrization,
)


def dummy_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((8, 8, 3)),
            tf.keras.layers.Conv2D(4, (3, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile()
    return model


def dummy_model_grayscale():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((8, 8, 1)),
            tf.keras.layers.Conv2D(4, (3, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile()
    return model


def test_init_maco_buffer():
    """Ensure we can init the magnitude and phase for any size and dataset"""
    img_size_to_magnitude_size = {
        (8, 8, 3): (8, 5),
        (16, 24, 3): (16, 13),
        (32, 32, 3): (32, 17),
        (128, 128, 3): (128, 65),
        (256, 512, 3): (256, 257),
        (1024, 2048, 3): (1024, 1025),
        (8, 8, 1): (8, 5),
        (16, 24, 1): (16, 13),
        (32, 32, 1): (32, 17),
        (128, 128, 1): (128, 65),
        (256, 512, 1): (256, 257),
        (1024, 2048, 1): (1024, 1025),
    }

    # Test without dataset
    for img_size, spectrum_size in img_size_to_magnitude_size.items():
        magnitude, phase = init_maco_buffer(img_size)
        assert magnitude.shape[1:] == spectrum_size
        assert phase.shape[1:] == spectrum_size

    # Test with dataset
    dummy_dataset_rgb = tf.data.Dataset.from_tensor_slices(tf.random.normal((10, 32, 32, 3))).batch(
        2
    )
    dummy_dataset_gray = tf.data.Dataset.from_tensor_slices(
        tf.random.normal((10, 32, 32, 1))
    ).batch(2)

    for img_size, spectrum_size in img_size_to_magnitude_size.items():
        if img_size[-1] == 3:
            magnitude, phase = init_maco_buffer(img_size, dataset=dummy_dataset_rgb)
        else:
            magnitude, phase = init_maco_buffer(img_size, dataset=dummy_dataset_gray)
        assert magnitude.shape[1:] == spectrum_size
        assert phase.shape[1:] == spectrum_size


def test_maco_image_param():
    """Ensure we can reconstruct an image from magnitude and phase"""
    img_size_to_magnitude_size = {
        (8, 8, 3): (8, 5),
        (16, 24, 3): (16, 13),
        (32, 32, 3): (32, 17),
        (128, 128, 3): (128, 65),
        (256, 512, 3): (256, 257),
        (1024, 2048, 3): (1024, 1025),
        (8, 8, 1): (8, 5),
        (16, 24, 1): (16, 13),
        (32, 32, 1): (32, 17),
        (128, 128, 1): (128, 65),
        (256, 512, 1): (256, 257),
        (1024, 2048, 1): (1024, 1025),
    }

    dummy_dataset_rgb = tf.data.Dataset.from_tensor_slices(tf.random.normal((10, 32, 32, 3))).batch(
        2
    )
    dummy_dataset_gray = tf.data.Dataset.from_tensor_slices(
        tf.random.normal((10, 32, 32, 1))
    ).batch(2)

    for img_size, spectrum_size in img_size_to_magnitude_size.items():
        if img_size[-1] == 3:
            magnitude, phase = init_maco_buffer(img_size, dataset=dummy_dataset_rgb)
        else:
            magnitude, phase = init_maco_buffer(img_size, dataset=dummy_dataset_gray)
        img = maco_image_parametrization(magnitude, phase, (0, 1))
        assert img.shape == img_size


def test_maco():
    """Ensure the optimization process is returning a valid image"""
    model_rgb = dummy_model()
    model_grayscale = dummy_model_grayscale()

    objectives_rgb = [
        Objective.neuron(model_rgb, -1, 0),
        Objective.direction(model_rgb, -1, tf.one_hot(1, 10)),
        Objective.channel(model_rgb, -3, 0),
    ]

    objectives_grayscale = [
        Objective.neuron(model_grayscale, -1, 0),
        Objective.direction(model_grayscale, -1, tf.one_hot(1, 10)),
        Objective.channel(model_grayscale, -3, 0),
    ]

    dummy_dataset_rgb = tf.data.Dataset.from_tensor_slices(tf.random.normal((10, 8, 8, 3))).batch(2)
    dummy_dataset_gray = tf.data.Dataset.from_tensor_slices(tf.random.normal((10, 8, 8, 1))).batch(
        2
    )

    for objective in objectives_rgb:
        # Test for RGB ImageNet images
        img, transparency = maco(
            objective, nb_steps=10, custom_shape=(8, 8), values_range=(-127, 127)
        )
        assert img.shape == (8, 8, 3)
        assert transparency.shape == (8, 8, 3)
        assert np.min(img) >= -127
        assert np.max(img) <= 127

        # Test for RGB images
        img, transparency = maco(
            objective,
            nb_steps=10,
            custom_shape=(8, 8),
            values_range=(-127, 127),
            maco_dataset=dummy_dataset_rgb,
        )
        assert img.shape == (8, 8, 3)
        assert transparency.shape == (8, 8, 3)
        assert np.min(img) >= -127
        assert np.max(img) <= 127

    for objective in objectives_grayscale:
        # Test for grayscale images
        img, transparency = maco(
            objective,
            nb_steps=10,
            custom_shape=(8, 8),
            values_range=(-127, 127),
            maco_dataset=dummy_dataset_gray,
        )
        assert img.shape == (8, 8, 1)
        assert transparency.shape == (8, 8, 1)
        assert np.min(img) >= -127
        assert np.max(img) <= 127
