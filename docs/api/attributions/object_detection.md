# Object detection with Xplique

[Attributions: Object Detection tutorial](https://colab.research.google.com/drive/1X3Yq7BduMKqTA0XEheoVIpOo3IvOrzWL) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1X3Yq7BduMKqTA0XEheoVIpOo3IvOrzWL) </sub>





## Which kind of tasks are supported by Xplique?

With the [operator's api](../api/attributions/operator) you can treat many different problems with Xplique. There is one operator for each task.

| Task and Documentation link                        | `operator` parameter value <br/> from `xplique.Tasks` Enum  | Tutorial link |
| :------------------------------------------------- | :---------------------------------------------------------- | :------------ |
| [Classification](../classification/)               | `CLASSIFICATION`        | <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) </sub> |
| **Object Detection**                               | `OBJECT_DETECTION`      | <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1X3Yq7BduMKqTA0XEheoVIpOo3IvOrzWL) </sub> |
| [Regression](../regression/)                       | `REGRESSION`            | <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pjDJmAa9oeSquYtbYh6tksU6eTmObIcq) </sub> |
| [Semantic Segmentation](../semantic_segmentation/) | `SEMANTIC_SEGMENTATION` | <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AHg7KO1fCOX5nZLGZfxkZ2-DLPPdSfbX) </sub> |

!!!info
    They all share the [API for Xplique attribution methods](../api_attributions/).





## Simple example

```python
import xplique
from xplique.attributions import Saliency
from xplique.metrics import Deletion

# load images and model
# ...

predictions = model(images)
explainer = Saliency(model, operator=xplique.Tasks.OBJECT_DETECTION)

# explain each image - bounding-box pair separately
for all_bbx_for_one_image, image in zip(predictions, images):
    # an image is needed per bounding box, so we tile them
    repeated_image = tf.tile(tf.expand_dims(image, axis=0),
                             (tf.shape(all_bbx_for_one_image)[0], 1, 1, 1))

    explanations = explainer(repeated_image, all_bbx_for_one_image)

    # either compute several score or
    # concatenate repeated images and corresponding boxes in one tensor
    metric_for_one_image = Deletion(model, repeated_image, all_bbx_for_one_image,
                                    operator=xplique.Tasks.OBJECT_DETECTION)
    score_saliency = metric(explanations)
```





## How to use it?

To apply attribution methods, the [**common API documentation**](../api_attributions/) describes the parameters and how to fix them. However, depending on the task and thus on the `operator`, there are three points that vary:

- **[The `operator` parameter](#the-operator)** value, it is an Enum or a string identifying the task,

- **[The model's output](#models-output)** specification, as `model(inputs)` is used in the computation of the operators, and

- **[The `targets` parameter](#the-targets-parameter)** format, indeed, the `targets` parameter specifies what to explain and the format of such specification depends on the task.





## The `operator` ##

### How to specify it

In Xplique, to adapt attribution methods, you should specify the task to the `operator` parameter. In the case of object detection, with either:
```python
Method(model, operator="object detection")
# or
Method(model, operator=xplique.Tasks.OBJECT_DETECTION)
```

!!!info
    There are several [variants of the object detection operator](#the-different-operators-variant-and-what-they-explain) to explain part of the prediction.



### The computation

This operator is a generalization of DRise method introduced by Petsiuk & al. [^1] to most attribution methods. The computation is the same as the one described in the DRise paper. The DRise can be divided into two principles:

- **The matching**: DRise extends Rise (described in detail in [the Rise tutorial](https://colab.research.google.com/drive/1icu2b1JGfpTRa-ic8tBSXnqqfuCGW2mO)) to explain object detection. Rise is a perturbation-based method, hence current predictions are compared to predictions on perturbed inputs. However, object detectors predict several boxes with no consistency in the order, thus DRise chooses to match the current bounding box to the most similar one and use the similarity metric as the perturbation score.
- **The similarity metric**: This is the score used by DRise to match bounding boxes. It uses the three parts of a bounding box prediction, the position of the box, the box objectness, and the associated class. A score is computed for each of those three parts and these scores are multiplied:

$$
score = intersection\_score * detection\_probability * classification\_score
$$

With:
$$
intersection\_score = IOU(coordinates_{ref}, coordinates_{pred})
$$

$$
detection\_probability = objectness_{pred}
$$

$$
classification\_score = \frac{\sum(classes_{ref} * classes_{pred})}{||classes_{ref}|| * ||classes_{pred}||}
$$

!!!info
    The intersection score of the operator is the IOU (Intersection Over Union) by default but can be modified by specifying as [custom intersection score](#custom-intersection-score).

!!!info
    With the DRise formula the methods explain the box position, the box objectness, and the class prediction at the same time. However, the user may want to explain them separately, therefore several variants of this operator are available in Xplique and described in [What can we explain and how? section](#what-can-we-explain-and-how).



### The behavior

- In the case of [perturbation-based methods](../api_attributions/#gradient-based-approaches), the perturbation score is the similarity metric aforementioned.
- For [gradient-based methods](../api_attributions/#perturbation-based-approaches), the gradient of the similarity metric is given, but no matching is necessary as no perturbation is made.





## Model's output ##

We expect `model(inputs)` to yield a $(n, nb\_boxes, 4 + 1 + nb\_classes)$ tensors or array where:

- $n$: the number of inputs, it should match the first dimension of `inputs`.
- $nb\_boxes$: a fixed number of bounding boxes predicted for a given image (no NMS).
- $(4 + 1 + nb\_classes)$: the encoding of a bounding box prediction
- $4$: the bounding box coordinates $(x_{top\_left}, y_{top\_left}, x_{bottom\_right}, y_{bottom\_right})$, with $x_{top\_left} < x_{bottom\_right}$ and $y_{top\_left} < y_{bottom\_right}$.
- $1$: the objectness or detection probability of the bounding box,
- $nb\_classes$: the class of the bounding box, a soft class predictions not a one-hot encoding.

!!!warning
    Object detection models provided to the explainer should not include NMS and classification should be soft classification not one-hot encoding. Furthermore, if the model does not match the expected format, a wrapper may be needed. (see [the tutorial](https://colab.research.google.com/drive/1X3Yq7BduMKqTA0XEheoVIpOo3IvOrzWL) for an example).

!!!info
    PyTorch models are not natively treated by Xplique, however, a simple wrapper is available in [pytorch documentation](../pytorch/).





## The `targets` parameter ##

### Role

The `targets` parameter specifies what is to explain in the `inputs`, it is passed to the `explain` or to the `__call__` method of an explainer or metric and used by the operators. In the case of object detection, it indicates which box to explain, furthermore, it gives the initial predictions to the operator as the reference for perturbation-based methods.



### Format

The `targets` parameter in the case of semantic segmentation should have the same shape as the [model's output](#models-output) as the same computation are made. Concretely, the `targets` parameter should have a shape of $(n, 4 + 1 + nb\_classes)$ to explain a bounding box for each input (detail in [model's output description](#models-output)).

Additionally, there is a possibility to explain a group of bounding boxes at the same time described in the [explaining several bounding boxes section](#explain-several-bounding-boxes-simultaneously) which requires a different shape.



### In practice

To explain each bounding box individually, the images need to be repeated. Indeed, object detector predict several bounding boxes per image and the first dimension of `inputs` and `targets` should match as it corresponds to the sample dimension. Therefore, the easiest way to obtain this is for each image to repeat it so that it matches the number of bounding boxes to explain for this image.

In the [simple example](#simple-example), there is a loop on the images - predictions pair, then images are repeated to match the number of predicted bounding boxes, and finally, the `targets` parameter takes the predicted bounding boxes.

!!!tip
    AS specified in the [model's output specification](#models-output), the NMS (Non Maximum Suppression) should not be included in the model. However, it can be used to select the bounding boxes to explain.

!!!warning
    Repeating images may create a tensor that exceeds memory for large images and/or when many bounding boxes are to be explained. In this case, we advise to make a loop on the images, then a loop on the boxes.



### Explain several bounding boxes simultaneously

The user may not want to explain each bounding box individually but several bounding boxes at the same time (*i.e* a set of pedestrian bounding boxes on a sidewalk). In this case, the `targets` parameter shape will not be $(n, 4 + 1 + nb\_classes)$ but $(n, nb\_boxes, 4 + 1 + nb\_classes)$, with $nb\_boxes$ the number of boxes to explain simultaneously. In this case, $nb\_boxes$ bounding boxes are associated to each sample and a single attribution map is returned. However, for different images, $nb\_boxes$ may not be fix and it may not be possible to make a single tensor in this case. Thus, we recommend to treat each group of bounding boxes with a different call to the attribution method with $n=1$.

To return one explanation for several bounding boxes, Xplique takes the mean of the bounding boxes individual explanations and returns it.

For a concrete example, please refer to the [Attributions: Object detection](https://colab.research.google.com/drive/1X3Yq7BduMKqTA0XEheoVIpOo3IvOrzWL) tutorial.





## What can be explained and how?

### The different elements in object detection

In object detection, the prediction for a given bounding box include several pieces of information: The **box position**, the **box probability of containing something**, and the **class of the detected object**. Therefore we may want to explain each of them separately, however, the DRise method of matching bounding boxes should be kept in mind. Indeed, the box position cannot be removed from the score, otherwise, the explanation may not correspond to the same object.



### The different operator's variants and what they explain

The Xplique library allows the specification of which part of the prediction to explain via a set 4 operators: the one as defined by the DRise formula and three variants:

- `"object detection"`: the one described in [the operator section](#the-operator):

    $$score = intersection\_score * detection\_probability * classification\_score$$

- `"object detection box position"`: explains only the bounding box position:

    $$score = intersection\_score$$

- `"object detection box proba"`: explains the probability of a bounding box to contain something:

    $$score = intersection\_score * detection\_probability$$

- `"object detection box class"`: explains the class of a bounding box:

    $$score = intersection\_score * classification\_score$$





## Custom intersection score

The default intersection score is IOU, but it is possible to define a custom intersection score. The only constraint is that it should follow `xplique.commons.object_detection_operator._box_iou` signature for it to work.

```python
from xplique.attributions import Saliency
from xplique.commons.operators import object_detection_operator

custom_intersection_score = ...

custom_operator = lambda model, inputs, targets: object_detection_operator(
    model, inputs, targets, intersection_score=custom_intersection_score
)

explainer = Saliency(model, operator=custom_operator)

...  # All following steps are the same as the examples
```

[^1] [Black-box Explanation of Object Detectors via Saliency Maps (2021)](https://arxiv.org/pdf/2006.03204.pdf)