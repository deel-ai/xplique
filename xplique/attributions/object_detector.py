"""
Module related to Object detector method
"""

from deprecated import deprecated

import numpy as np
import tensorflow as tf

from ..types import Optional, Callable, Union

from .base import BlackBoxExplainer, WhiteBoxExplainer
from ..commons import get_gradient_functions
from ..commons.operators import object_detection_operator
from ..utils_functions.object_detection import _box_iou, _format_objects


OLD_OBJECT_DETECTION_DEPRECATION_MESSAGE = """
\n
The method to compute attribution explanation explanations changed drastically after version 1.0.0. 
For more information please refer to the documentation:
https://deel-ai.github.io/xplique/latest/api/attributions/object_detection/

Nonetheless, here is a quick example of how it is used now:
```
from xplique.attributions import AnyMethod
explainer = AnyMethod(model, operator="object detection", ...)
explanation = explainer(inputs, targets)  # be aware, targets are specific to object detection here
```
"""


class BoundingBoxesExplainer(BlackBoxExplainer):
    """
    For a given black box explainer, this class allows to find explications of an object detector
    model. The object model detector shall return a list (length of the size of the batch)
    containing a tensor of 2 dimensions.
    The first dimension of the tensor is the number of bounding boxes found in the image
    The second dimension is:
    [xmin, ymin, xmax, ymax, probability_detection, ones_hot_classif_result]

    This work is a generalization of the following article at any kind of black box explainer and
    also can be used for other kind of object detector (like segmentation)

    Ref. Petsiuk & al., Black-box Explanation of Object Detectors via Saliency Maps (2021).
    https://arxiv.org/pdf/2006.03204.pdf

    Parameters
    ----------
    explainer
        the black box explainer used to explain the object detector model
    _
        inheritance from old versions
    intersection_score
        the iou calculator used to compare two objects.
    """

    @deprecated(version="1.0.0", reason=OLD_OBJECT_DETECTION_DEPRECATION_MESSAGE)
    def __init__(self,
                 explainer: BlackBoxExplainer,
                 _: Optional[Callable] = _format_objects,
                 intersection_score: Optional[Callable]  = _box_iou):
        # make operator function based on arguments
        operator = lambda model, inputs, targets: \
            object_detection_operator(model, inputs, targets, intersection_score)

        # BlackBoxExplainer init to set operator
        super().__init__(model=explainer.model, batch_size=explainer.batch_size, operator=operator)
        self.explainer = explainer

        # update explainer inference functions for explain method
        self.explainer.inference_function = self.inference_function
        self.explainer.batch_inference_function = self.batch_inference_function

        if isinstance(self.explainer, WhiteBoxExplainer):
            # check and get gradient function from model and operator
            self.gradient, self.batch_gradient = get_gradient_functions(self.model, operator)
            self.explainer.gradient = self.gradient
            self.explainer.batch_gradient = self.batch_gradient

    def explain(self,
                inputs: Union[tf.data.Dataset, tf.Tensor, np.array],
                targets: Optional[Union[tf.Tensor, np.array]] = None) -> tf.Tensor:
        """
        Compute the explanation of the object detection through the explainer

        Parameters
        ----------
        inputs
            Dataset, Tensor or Array. Input samples to be explained.
            If Dataset, targets should not be provided (included in Dataset).
            Expected shape (N, H, W, C).
            More information in the documentation.
        targets
            Tensor or Array. One-hot encoding of the model's output from which an explanation
            is desired. One encoding per input and only one output at a time. Therefore,
            the expected shape is (N, ...). With features matching the object formatting.
            See object detection operator documentation for more information
            More information in the documentation.

        Returns
        -------
        explanation
            The resulting object detection explanation
        """
        if len(tf.shape(targets)) == 1:
            targets = tf.expand_dims(targets, axis=0)

        return self.explainer.explain(inputs, targets)
