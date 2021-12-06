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

from ..utils import generate_data
from ..utils import generate_model
from xplique.attributions import DeconvNet
from xplique.attributions import GradCAM
from xplique.attributions import GradCAMPP
from xplique.attributions import GradientInput
from xplique.attributions import GuidedBackprop
from xplique.attributions import IntegratedGradients
from xplique.attributions import Occlusion
from xplique.attributions import Rise
from xplique.attributions import Saliency
from xplique.attributions import SmoothGrad
from xplique.attributions import SquareGrad
from xplique.attributions import VarGrad
from xplique.metrics import AverageStability
from xplique.metrics import Deletion
from xplique.metrics import Insertion
from xplique.metrics import MuFidelity
from xplique.metrics.base import ExplainerMetric
from xplique.metrics.base import ExplanationMetric


def _default_methods(model, output_layer_index=-2):
    return [
        Saliency(model, output_layer_index),
        GradientInput(model, output_layer_index),
        SmoothGrad(model, output_layer_index, nb_samples=5),
        VarGrad(model, output_layer_index),
        SquareGrad(model, output_layer_index),
        IntegratedGradients(model, output_layer_index, steps=5),
        GradCAM(model, output_layer_index),
        Occlusion(model, patch_size=4, patch_stride=4),
        Rise(model, nb_samples=10),
        GuidedBackprop(model, output_layer_index),
        DeconvNet(model, output_layer_index),
        GradCAMPP(model, output_layer_index),
    ]


def test_common():
    """Test that all the attributions method works as explainer"""

    input_shape, nb_labels, samples = ((16, 16, 3), 10, 20)
    x, y = generate_data(input_shape, nb_labels, samples)
    model = generate_model(input_shape, nb_labels)

    explainers = _default_methods(model)

    metrics = [
        Deletion(model, x, y, steps=3),
        Insertion(model, x, y, steps=3),
        MuFidelity(model, x, y, nb_samples=3),
        AverageStability(model, x, y, nb_samples=3),
    ]

    for explainer in explainers:
        explanations = explainer(x, y)
        for metric in metrics:
            assert hasattr(metric, "evaluate")
            if isinstance(metric, ExplainerMetric):
                score = metric(explainer)
            else:
                score = metric(explanations)
            print(f"\n\n\n {type(score)} \n\n\n")
            assert type(score) in [np.float32, np.float64, float]
