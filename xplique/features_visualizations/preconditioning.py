"""
Images preconditionners

Adaptation of the original Lucid library :
https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/color.py
Credit is due to the original Lucid authors.
"""


import numpy as np
import tensorflow as tf


imagenet_color_correlation = tf.cast(
      [[0.56282854, 0.58447580, 0.58447580],
       [0.19482528, 0.00000000,-0.19482528],
       [0.04329450,-0.10823626, 0.06494176]], tf.float32
)


def recorrelate_colors(images):
    """
    Map uncorrelated colors to 'normal colors' by using empirical color
    correlation matrix of ImageNet (see https://distill.pub/2017/feature-visualization/)

    Parameters
    ----------
    images : tf.tensor (N, W, H, C)
        Input samples , with N number of samples, W & H the sample dimensions,
        and C the number of channels.

    Returns
    -------
    images : tf.tensor (N, W, H, C)
        Images recorrelated.
    """
    images_flat = tf.reshape(images, [-1, 3])
    images_flat = tf.matmul(images_flat, imagenet_color_correlation)
    return tf.reshape(images_flat, tf.shape(images))


def to_valid_rgb(images, normalizer='sigmoid'):
    """
    Apply transformations to map tensors to valid rgb images.

    Parameters
    ----------
    images : tf.tensor (N, W, H, C)
        Input samples, with N number of samples, W & H the sample dimensions,
        and C the number of channels.
    normalizer : None, 'sigmoid' or 'clip', optional
        Transformation to apply to map pixels in the range [0, 1].

    Returns
    -------
    images : tf.tensor (N, W, H, C)
        Images after transformations
    """
    images = recorrelate_colors(images)
    images = tf.nn.sigmoid(images) if normalizer == 'sigmoid' else images
    images = tf.clip_by_norm(images, 1, axes=-1) if normalizer == 'clip' else images
    return images


def fft_2d_freq(width, height):
    """
    Return the fft samples frequencies for a given width/height.
    As we deal with real values (pixels), the Discrete Fourier Transform is
    Hermitian symmetric, tensorflow's reverse operation requires only
    the unique components (width, height//2+1).

    Parameters
    ----------
    width : int
        Width of the image.
    height : int
        Height of the image.

    Returns
    -------
    frequencies : ndarray
        Array containing the samples frequency bin centers in cycles per pixels
    """
    freq_y = np.fft.fftfreq(height)[:, np.newaxis]

    cut_off = int(width % 2 == 1)
    freq_x = np.fft.fftfreq(width)[:width//2+1+cut_off]

    return np.sqrt(freq_x**2 + freq_y**2)


def get_fft_scale(width, height, decay_power=1.0):
    """
    Generate 'scaler' to normalize spectrum energy. Also scale the energy by the
    dimensions to use similar learning rate regardless of image size.
    adaptation of : https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/spatial.py
    #L73

    Parameters
    ----------
    width : int,
        Width of the image.
    height : int,
        Height of the image.
    decay_power : float, optional
        Control the allowed energy of the high frequency, a high value
        suppresses high frequencies.

    Returns
    -------
    fft_scale : tf.tensor
        Scale factor of the fft spectrum
    """
    frequencies = fft_2d_freq(width, height)
    fft_scale = 1.0 / np.maximum(frequencies, 1.0 / max(width, height)) ** decay_power
    fft_scale = fft_scale * np.sqrt(width * height)

    return tf.convert_to_tensor(fft_scale, dtype=tf.complex64)


def fft_to_rgb(shape, buffer, fft_scale):
    """
    Convert a fft buffer into images.

    Parameters
    ----------
    shape : tuple (N, W, H, C)
        Shape of the images with N number of samples, W & H the sample
        dimensions, and C the number of channels.
    buffer : tf.tensor
        Image buffer in the fourier basis.
    fft_scale : tf.tensor
        Scale factor of the fft spectrum

    Returns
    -------
    images : tf.tensor (N, W, H, C)
        Images in the 'pixels' basis.
    """
    batch, width, height, channels = shape
    spectrum = tf.complex(buffer[0], buffer[1]) * fft_scale

    image = tf.signal.irfft2d(spectrum)
    image = tf.transpose(image, (0, 2, 3, 1))
    image = image[:batch, :width, :height, :channels]

    return image / 4.0


def fft_image(shape, std=0.01):
    """
    Generate the preconditioned image buffer

    Parameters
    ----------
    shape : tuple (N, W, H, C)
        Shape of the images with N number of samples, W & H the sample
        dimensions, and C the number of channels.
    std : float, optional
        Standard deviation of the normal for the buffer initialization
    Returns
    -------
    buffer : tf.tensor (2, N, C, W, H//2+1)
        Image buffer in the fourier basis.
    """
    batch, width, height, channels = shape
    frequencies = fft_2d_freq(width, height)

    buffer = tf.random.normal((2, batch, channels)+frequencies.shape,
                              stddev=std)

    return buffer
