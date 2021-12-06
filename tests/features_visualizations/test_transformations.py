# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import numpy as np
import tensorflow as tf

from ..utils import almost_equal
from xplique.features_visualizations.transformations import pad
from xplique.features_visualizations.transformations import random_blur


def test_blur():
    """Test basic behaviour of the blur transformations"""
    imgs = np.zeros((3, 3, 3, 3), np.float32)
    # green dot
    imgs[:, 1, 1, 1] = 1.0
    # blue dot
    imgs[:, 2, 2, 2] = 1.0

    # assuming a gaussian kernel of size=3, sigma=1
    gaussian_kernel_sum = np.sum(np.exp(0) * 1 + np.exp(-0.5) * 4 + np.exp(-1) * 4)
    c0 = np.exp(0) / gaussian_kernel_sum  # center
    c1 = np.exp(-0.5) / gaussian_kernel_sum  # delta 1
    c2 = np.exp(-1.0) / gaussian_kernel_sum  # delta 2

    blur_fn = random_blur(kernel_size=3, sigma_range=(1.0, 1.0))
    blur_imgs = blur_fn(imgs)

    # check we have the same images (batch blur is working correctly)
    assert almost_equal(blur_imgs[0], blur_imgs[1])
    assert almost_equal(blur_imgs[1], blur_imgs[2])
    # check that the channel are independant (not regular conv!)
    assert (
        blur_imgs[0, 1, 1, 0] != blur_imgs[0, 1, 1, 1]
        and blur_imgs[0, 1, 1, 1] != blur_imgs[0, 1, 1, 2]
    )
    # assert the gaussian kernel is correct
    assert almost_equal(
        blur_imgs[0, :, :, 1], np.array([[c2, c1, c2], [c1, c0, c1], [c2, c1, c2]])
    )  # green
    assert almost_equal(
        blur_imgs[0, :, :, 2], np.array([[0.0, 0.0, 0.0], [0.0, c2, c1], [0.0, c1, c0]])
    )  # blue
    assert almost_equal(
        blur_imgs[0, :, :, 0],
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )  # red


def test_pad():
    """Ensure the padding is consistent"""
    imgs = np.ones((3, 2, 2, 3), np.float32)
    pad_fn = pad(2, 0.0)
    imgs_padded = pad_fn(imgs)

    # check we have the same images (batch pad is working correctly)
    assert almost_equal(imgs_padded[0], imgs_padded[1])
    assert almost_equal(imgs_padded[1], imgs_padded[2])
    # check the channel are treated the same way
    assert almost_equal(imgs_padded[0, :, :, 0], imgs_padded[0, :, :, 1])
    assert almost_equal(imgs_padded[0, :, :, 1], imgs_padded[0, :, :, 2])
    # check the padding value is ok
    assert almost_equal(
        imgs_padded[0, :, :, 0],
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )
