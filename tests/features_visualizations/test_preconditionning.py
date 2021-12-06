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
from xplique.features_visualizations.preconditioning import fft_image
from xplique.features_visualizations.preconditioning import fft_to_rgb
from xplique.features_visualizations.preconditioning import get_fft_scale


def test_shape():
    """Ensure the shape of the image generated is correct"""

    input_shapes = [(2, 8, 8, 3), (2, 64, 64, 1), (2, 256, 256, 3), (2, 512, 512, 1)]

    for input_shape in input_shapes:
        n, w, h, c = input_shape
        buffer = fft_image(input_shape)
        scaler = get_fft_scale(w, h)
        img = fft_to_rgb(input_shape, buffer, scaler)

        # image generated from inverse fourier transformation should have
        # the desired shape
        assert img.shape == input_shape
