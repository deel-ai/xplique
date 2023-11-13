"""
Images preconditionners

Adaptation of the original Lucid library :
https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/color.py
Credit is due to the original Lucid authors.
"""


import numpy as np
import tensorflow as tf

from ..types import Tuple, Union, Callable


IMAGENET_SPECTRUM_URL = "https://storage.googleapis.com/serrelab/loupe/"\
                        "spectrums/imagenet_decorrelated.npy"


def recorrelate_colors(images: tf.Tensor) -> tf.Tensor:
    """
    Map uncorrelated colors to 'normal colors' by using empirical color
    correlation matrix of ImageNet (see https://distill.pub/2017/feature-visualization/)

    Parameters
    ----------
    images
        Input samples , with N number of samples, W & H the sample dimensions,
        and C the number of channels.

    Returns
    -------
    images
        Images recorrelated.
    """

    # constant
    imagenet_color_correlation = tf.cast(
      [[0.56282854, 0.58447580, 0.58447580],
       [0.19482528, 0.00000000,-0.19482528],
       [0.04329450,-0.10823626, 0.06494176]], tf.float32
    )

    images_flat = tf.reshape(images, [-1, 3])
    images_flat = tf.matmul(images_flat, imagenet_color_correlation)
    return tf.reshape(images_flat, tf.shape(images))


def to_valid_rgb(images: tf.Tensor,
                 normalizer: Union[str, Callable] = 'sigmoid',
                 values_range: Tuple[float, float] = (0, 1)) -> tf.Tensor:
    """
    Apply transformations to map tensors to valid rgb images.


    Parameters
    ----------
    images
        Input samples, with N number of samples, W & H the sample dimensions,
        and C the number of channels.
    normalizer
        Transformation to apply to map pixels in the range [0, 1]. Either 'clip' or 'sigmoid'.
    values_range
        Range of values of the inputs that will be provided to the model, e.g (0, 1) or (-1, 1).

    Returns
    -------
    images
        Images after correction
    """
    images = recorrelate_colors(images)

    if normalizer == 'sigmoid':
        images = tf.nn.sigmoid(images)
    elif normalizer == 'clip':
        images = tf.clip_by_value(images, values_range[0], values_range[1])
    else:
        images = normalizer(images)

    # rescale according to value range
    images = images - tf.reduce_min(images, (1, 2, 3), keepdims=True)
    images = images / tf.reduce_max(images, (1, 2, 3), keepdims=True)
    images *= values_range[1] - values_range[0]
    images += values_range[0]

    return images


def fft_2d_freq(width: int, height: int) -> np.ndarray:
    """
    Return the fft samples frequencies for a given width/height.
    As we deal with real values (pixels), the Discrete Fourier Transform is
    Hermitian symmetric, tensorflow's reverse operation requires only
    the unique components (width, height//2+1).

    Parameters
    ----------
    width
        Width of the image.
    height
        Height of the image.

    Returns
    -------
    frequencies
        Array containing the samples frequency bin centers in cycles per pixels
    """
    freq_y = np.fft.fftfreq(height)[:, np.newaxis]

    cut_off = int(width % 2 == 1)
    freq_x = np.fft.fftfreq(width)[:width//2+1+cut_off]

    return np.sqrt(freq_x**2 + freq_y**2)


def get_fft_scale(width: int, height: int, decay_power: float = 1.0) -> tf.Tensor:
    """
    Generate 'scaler' to normalize spectrum energy. Also scale the energy by the
    dimensions to use similar learning rate regardless of image size.
    adaptation of : https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/spatial.py
    #L73

    Parameters
    ----------
    width
        Width of the image.
    height
        Height of the image.
    decay_power
        Control the allowed energy of the high frequency, a high value
        suppresses high frequencies.

    Returns
    -------
    fft_scale
        Scale factor of the fft spectrum
    """
    frequencies = fft_2d_freq(width, height)
    fft_scale = 1.0 / np.maximum(frequencies, 1.0 / max(width, height)) ** decay_power
    fft_scale = fft_scale * np.sqrt(width * height)

    return tf.convert_to_tensor(fft_scale, dtype=tf.complex64)


def fft_to_rgb(shape: Tuple, buffer: tf.Tensor, fft_scale: tf.Tensor) -> tf.Tensor:
    """
    Convert a fft buffer into images.

    Parameters
    ----------
    shape
        Shape of the images with N number of samples, W & H the sample
        dimensions, and C the number of channels.
    buffer
        Image buffer in the fourier basis.
    fft_scale
        Scale factor of the fft spectrum

    Returns
    -------
    images
        Images in the 'pixels' basis.
    """
    batch, width, height, channels = shape
    spectrum = tf.complex(buffer[0], buffer[1]) * fft_scale

    image = tf.signal.irfft2d(spectrum)
    image = tf.transpose(image, (0, 2, 3, 1))
    image = image[:batch, :width, :height, :channels]

    return image / 4.0


def fft_image(shape: Tuple, std: float = 0.01) -> tf.Tensor:
    """
    Generate the preconditioned image buffer

    Parameters
    ----------
    shape
        Shape of the images with N number of samples, W & H the sample
        dimensions, and C the number of channels.
    std
        Standard deviation of the normal for the buffer initialization
    Returns
    -------
    buffer
        Image buffer in the fourier basis.
    """
    batch, width, height, channels = shape
    frequencies = fft_2d_freq(width, height)

    buffer = tf.random.normal((2, batch, channels)+frequencies.shape,
                              stddev=std)

    return buffer


def init_maco_buffer(image_shape, std=1.0):
    """
    Initialize the buffer for the MACO algorithm.

    Parameters
    ----------
    image_shape
        Shape of the images with N number of samples, W & H the sample
        dimensions, and C the number of channels.
    std
        Standard deviation of the normal for the buffer initialization

    Returns
    -------
    magnitude
        Magnitude of the spectrum
    phase
        Phase of the spectrum
    """
    spectrum_shape = (image_shape[0], image_shape[1]//2+1)

    # init randomly the phase and load the constrained spectrum (average spectrum)
    phase = np.random.normal(size=(3, *spectrum_shape), scale=std).astype(np.float32)

    magnitude_path = tf.keras.utils.get_file("spectrum_decorrelated.npy",
                                             IMAGENET_SPECTRUM_URL,
                                             cache_subdir="spectrums")
    magnitude = np.load(magnitude_path)
    magnitude = tf.image.resize(np.moveaxis(magnitude, 0, -1), spectrum_shape).numpy()
    magnitude = np.moveaxis(magnitude, -1, 0)

    return tf.cast(magnitude, tf.float32), tf.cast(phase, tf.float32)

@tf.function
def maco_image_parametrization(magnitude, phase, values_range):
    """
    Generate the image from the magnitude and phase using MaCo method.

    Parameters
    ----------
    magnitude
        Magnitude of the spectrum
    phase
        Phase of the spectrum
    values_range
        Range of the values of the image

    Returns
    -------
    img
        Image in the 'pixels' basis.
    """
    phase = phase - tf.reduce_mean(phase)
    phase = phase / (tf.math.reduce_std(phase) + 1e-5)

    buffer = tf.complex(tf.cos(phase) * magnitude, tf.sin(phase) * magnitude)
    img = tf.signal.irfft2d(buffer)
    img = tf.transpose(img, [1,2,0])

    img = img - tf.reduce_mean(img)
    img = img / (tf.math.reduce_std(img) + 1e-5)

    img = recorrelate_colors(img)
    img = tf.nn.sigmoid(img)

    img = img * (values_range[1] - values_range[0]) + values_range[0]

    return img
