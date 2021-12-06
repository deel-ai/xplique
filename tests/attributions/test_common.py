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

from ..utils import generate_data
from ..utils import generate_model
from xplique.attributions import DeconvNet
from xplique.attributions import GradCAM
from xplique.attributions import GradCAMPP
from xplique.attributions import GradientInput
from xplique.attributions import GuidedBackprop
from xplique.attributions import IntegratedGradients
from xplique.attributions import KernelShap
from xplique.attributions import Lime
from xplique.attributions import Occlusion
from xplique.attributions import Rise
from xplique.attributions import Saliency
from xplique.attributions import SmoothGrad
from xplique.attributions import SquareGrad
from xplique.attributions import VarGrad
from xplique.attributions.base import BlackBoxExplainer


def _default_methods(model, output_layer_index):
    return [
        Saliency(model, output_layer_index),
        GradientInput(model, output_layer_index),
        SmoothGrad(model, output_layer_index),
        VarGrad(model, output_layer_index),
        SquareGrad(model, output_layer_index),
        IntegratedGradients(model, output_layer_index),
        GradCAM(model, output_layer_index),
        Occlusion(model),
        Rise(model),
        GuidedBackprop(model, output_layer_index),
        DeconvNet(model, output_layer_index),
        GradCAMPP(model, output_layer_index),
        Lime(model),
        KernelShap(model),
    ]


def test_common():
    """Test applied to all the attributions"""

    input_shape, nb_labels, samples = ((32, 32, 3), 10, 20)
    model = generate_model(input_shape, nb_labels)
    output_layer_index = -2

    inputs_np, targets_np = generate_data(input_shape, nb_labels, samples)
    inputs_tf, targets_tf = tf.cast(inputs_np, tf.float32), tf.cast(
        targets_np, tf.float32
    )
    dataset = tf.data.Dataset.from_tensor_slices((inputs_np, targets_np))
    batched_dataset = tf.data.Dataset.from_tensor_slices((inputs_np, targets_np)).batch(
        3
    )

    methods = _default_methods(model, output_layer_index)

    for inputs, targets in [
        (inputs_np, targets_np),
        (inputs_tf, targets_tf),
        (dataset, None),
        (batched_dataset, None),
    ]:
        for method in methods:
            explanations = method.explain(inputs, targets)

            # all explanation must have an explain method
            assert hasattr(method, "explain")

            # all explanations returned must be numpy array
            assert isinstance(explanations, tf.Tensor)


def test_batch_size():
    """Ensure the functioning of attributions for special batch size cases"""

    input_shape, nb_labels, samples = ((10, 10, 3), 5, 20)
    inputs, targets = generate_data(input_shape, nb_labels, samples)
    model = generate_model(input_shape, nb_labels)
    output_layer_index = -1

    batch_sizes = [None, 1, 32]

    for bs in batch_sizes:

        methods = [
            Saliency(model, output_layer_index, bs),
            GradientInput(model, output_layer_index, bs),
            SmoothGrad(model, output_layer_index, bs),
            VarGrad(model, output_layer_index, bs),
            SquareGrad(model, output_layer_index, bs),
            IntegratedGradients(model, output_layer_index, bs),
            GradCAM(model, output_layer_index, bs),
            Occlusion(model, bs),
            Rise(model, bs),
            GuidedBackprop(model, output_layer_index, bs),
            DeconvNet(model, output_layer_index, bs),
            GradCAMPP(model, output_layer_index, bs),
            Lime(model, bs),
            KernelShap(model, bs),
        ]

        for method in methods:
            try:
                explanations = method.explain(inputs, targets)
            except:
                raise AssertionError(
                    "Explanation failed for method ",
                    method.__class__.__name__,
                    " batch size ",
                    bs,
                )


def test_model_caching():
    """Test the caching engine, used to avoid re-tracing"""

    model = generate_model()
    output_layer_index = -1

    # the key used for caching is the following tuple
    cache_key = (id(model.input), id(model.output))

    cache_len_before = len(BlackBoxExplainer._cache_models.keys())  # pylint:
    # disable=protected-access

    assert (
        cache_key not in BlackBoxExplainer._cache_models
    )  # pylint: disable=protected-access

    _ = _default_methods(model, output_layer_index)

    # check that the key is now in the cache
    assert (
        cache_key in BlackBoxExplainer._cache_models
    )  # pylint: disable=protected-access

    # ensure that there no more than one key has been added
    assert (
        len(BlackBoxExplainer._cache_models) == cache_len_before + 1
    )  # pylint: disable=protected-access
