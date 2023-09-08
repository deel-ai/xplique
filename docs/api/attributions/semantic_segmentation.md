# Semantic segmentation explanations with Xplique

[Attributions: Semantic segmentation tutorial](https://colab.research.google.com/drive/1AHg7KO1fCOX5nZLGZfxkZ2-DLPPdSfbX) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) </sub>





## Which kind of tasks are supported by Xplique?

With the [operator's api](../api/attributions/operator) you can treat many different problems with Xplique. There is one operator for each task.

| Task and Documentation link                        | `operator` parameter value <br/> from `xplique.Tasks` Enum  | Tutorial link |
| :------------------------------------------------- | :---------------------------------------------------------- | :------------ |
| [Classification](../classification/)               | `CLASSIFICATION`        | <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) </sub> |
| [Object Detection](../object_detection/)           | `OBJECT_DETECTION`      | <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1X3Yq7BduMKqTA0XEheoVIpOo3IvOrzWL) </sub> |
| [Regression](../regression/)                       | `REGRESSION`            | <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pjDJmAa9oeSquYtbYh6tksU6eTmObIcq) </sub> |
| **Semantic Segmentation**                          | `SEMANTIC_SEGMENTATION` | <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AHg7KO1fCOX5nZLGZfxkZ2-DLPPdSfbX) </sub> |

!!!info
    They all share the [API for Xplique attribution methods](../api_attributions/).





## Simple example

```python
import xplique
from xplique.utils_functions.segmentation import get_connected_zone
from xplique.attributions import Saliency
from xplique.metrics import Deletion

# load images and model
# ...

# extract targets individually
coordinates_of_object = (42, 42)
predictions = model(image)
target = get_connected_zone(predictions, coordinates_of_object)
inputs = tf.expand_dims(image, 0)
targets = tf.expand_dims(target, 0)

explainer = Saliency(model, operator=xplique.Tasks.SEMANTIC_SEGMENTATION)
explanations = explainer(inputs, targets)

metric = Deletion(model, inputs, targets, operator=xplique.Tasks.SEMANTIC_SEGMENTATION)
score_saliency = metric(explanations)
```





## How to use it?

To apply attribution methods, the [**common API documentation**](../api_attributions/) describes the parameters and how to fix them. However, depending on the task and thus on the `operator`, there are three points that vary:

- **[The `operator` parameter](#the-operator)** value, it is an Enum or a string identifying the task,

- **[The model's output](#models-output)** specification, as `model(inputs)` is used in the computation of the operators, and

- **[The `targets` parameter](#the-targets-parameter)** format, indeed, the `targets` parameter specifies what to explain and the format of such specification depends on the task.

!!!info
    Applying attribution methods to semantic segmentation with Xplique has a particularity: a set of functions from `utils_functions.segmentation` are used to define `targets` and are documented in the a [specific section](#the-segmentation-utils-functions).



## The `operator` ##

### How to specify it

In Xplique, to adapt attribution methods, you should specify the task to the `operator` parameter. In the case of semantic segmentation, with either:
```python
Method(model, operator="semantic segmentation")
# or
Method(model, operator=xplique.Tasks.SEMANTIC_SEGMENTATION)
```



### The computation

The operator for semantic segmentation is similar to the classification one, but the output is not a class but a matrix of class. The operator should take this position into account, thus it manipulates two elements:

- **The zone of interest**: it represents the zone/pixels on which we want the explanation to be made. It could be a single object like a person, a group of objects like trees, a part of an object that has been wrongly classified, or even the border of an object. Note that the concept of object here only makes sense for us as the model only classifies pixels, which is why Xplique includes the [segmentation utils function](#the-segmentation-utils-functions).

- **The class of interest**: it represents the channel of the prediction we want to explain. Similarly to classification, we could either want to explain a cat or a dog in the same image. Note that in some case, providing several classes could make sense, see the example of applications with [explanations of the borders between two objects](#the-border-between-two-objects).

Indeed, the semantic segmentation operator multiplies the model's predictions by the targets, which can be considered a mask. Then the operator divide the sum of the remaining predictions over the size of the mask. In some, the operator take the mean predictions over the zone and class of interest

$$
score = mean_{over\ the\ zone\ and\ class\ of\ interest}(model(inputs))
$$

Note that the two information need to be communicated through the [`targets` parameter](#the-targets-parameter).



### The behavior

- In the case of [perturbation-based methods](../api_attributions/#gradient-based-approaches), the perturbation score is the difference between the operator's output for the studied `inputs` and the perturbed inputs. Where the operator's output is the mean logits value over the class and zone of interest.
- For [gradient-based methods](../api_attributions/#perturbation-based-approaches), the gradient of the mean of model's predictions limited to the zone and class of interest.





## Model's output ##

We expect `model(inputs)` to yield a $(n, h, w, c)$ tensor or array where:

- $n$: the number of inputs, it should match the first dimension of `inputs`
- $h$: the height of the images
- $w$: the width of the images
- $c$: the number of classes

!!!warning
    The model's output for each pixel is expected to be a soft output and not the class prediction or a one hot encoding of the class. Otherwise the attribution methods will not be able to compare predictions efficiently.

!!!warning
    Contrary to classification, here a softmax or comparable last layer is necessary as zeros are interpreted by the operator as non-zone of interest. In this sense, strictly positive values are required.





## The `targets` parameter ##

### Role

The `targets` parameter specifies what is to explain in the `inputs`, it is passed to the `explain` or to the `__call__` method of an explainer or metric and used by the operators. In the case of semantic segmentation, the `targets` parameter enables the communication of the two necessary information for the [semantic segmentation operator](#the-operator):

- **The zone of interest**: to communicate the zone of interest via the `targets` parameter, the `targets` value on pixels that are not in the zone of interest should be set to zero. In this way `tf.math.sign(targets)` creates a mask of the zone of interest. This operation should be done along the $h$ and $w$ dimensions of `targets`.

- **The class of interest**: similarly to the zone of interest, the class of interest is communicated by setting other classes along the $c$ dimension to zero.



### Format

The `targets` parameter in the case of semantic segmentation should have the same shape as the [model's output](#models-output) as a difference is made between the two. Hence, the shape is $(n, h, w,c )$ with:
    - $n$ is the number of inputs, it should match the first dimension of `inputs`
    - $h$ is the height of the images
    - $w$ is the width of the images
    - $c$ is the number of classes

Then it should take values in $\{-1, 0, 1\}$, $1$ in the zone of interest (zone on the $h$ and $w$ dimension) and $0$ elsewhere. Similarly, values not on the channel corresponding to the class of interest (dimension $c$) should be $0$. In the case of the explanation of a border or with contrastive explanations, $-1$ values might be used.



### In practice

The `targets` parameter is computed via the [xplique.utils_functions.segmentation](#the-segmentation-utils-functions) set of functions. They manipulate model's prediction individually, as explanation requests are different between each image. Please refer to the [segmentation utils functions](#the-segmentation-utils-functions) for detail on how to design `targets`.

!!!tip
    You should not worry about such specification as the [segmentation utils functions](#the-segmentation-utils-functions) will do the work in your stead.

!!!warning
    The `targets` parameter for each sample should be defined individually. Then the batch dimension should be added manually or individual values should be stacked.





## The segmentation utils functions ##

[Source](https://github.com/deel-ai/xplique/tree/master/xplique/utils_functions/segmentation.py)

The segmentation utils functions are a set a utility functions used to compute the [`targets` parameter](#the-targets-parameter) values. They should be applied to each image separately as each segmentation is different want the things to explain differs between images. Nonetheless, you could use `tf.map_fn` to apply the same function to several images.

An example of application of those functions can be found in the [Attribution: Semantic segmentation](https://colab.research.google.com/drive/1AHg7KO1fCOX5nZLGZfxkZ2-DLPPdSfbX) tutorial.

For now, there are four functions:

### `get_class_zone`

The most simple, where the class of interest is `class_id` and the zone of interest corresponds to pixels where the class is the argmax along the classes dimension of the model's prediction. This function can be used to design `targets` to explain:

- the [class of a crowd of objects](#the-class-of-a-crowd-of-objects)
- the [class of an object](#the-class-of-an-object), if there is only one object in the image.
- the [class of a set of objects](#the-class-of-a-set-of-object), if there are few and locally close objects of the same class.

{{xplique.utils_functions.segmentation.get_class_zone}}



### `get_connected_zone`

Here `coordinates` is a $(h, w)$ tuple that indicates the indices of a pixel of the image. The class of interest is the argmax along the classes dimension for this given pixel. Then the zone of interest is the set of pixels with the same argmax class that forms a connected zone with the indicated pixel. This function can be seen as selecting a zone with a point in this zone. This function can be used to design `targets` to explain:

- the [class of an object](#the-class-of-an-object).
- the [class of a set of objects](#the-class-of-a-set-of-object), if they are connected.
- the [class of part of an object](#the-class-of-part-of-an-object), if this part have been classified differently than the object and the other surrounding objects.

{{xplique.utils_functions.segmentation.get_connected_zone}}



### `list_class_connected_zones`

A mix of `get_class_zone` and `get_connected_zone`. `class_id` indicates the class of interest and each connected zone for this class becomes a zone of interest (apart from zones with size under `zone_minimum_size`). It is useful for automatized treatment of explainability, but may generate explanations for zones we may not want to explain. Nonetheless, it can be used to design `targets` to explain similar elements as `get_connected_zone`.

!!!warning
    Contrarily to the other utils function for segmentation, here output is a list of tensors.

{{xplique.utils_functions.segmentation.list_class_connected_zones}}



### `get_in_out_border`

This function allows to compute the `targets` needed to explain the [border of an object](#the-border-of-an-object). For this function, `class_target_mask` encodes the class and the zone of interest. From this zone, the in-border (all pixels of the zone with contact to non-zone pixels) and the out-border (all non-zone pixels with contact to pixels of the zone) are computed. Then, the in-borders pixels are set with the predictions values, and out-borders with the opposite of the predictions values. Therefore, explaining this border corresponds to explaining what increased the class predictions inside the zone and decreased it outside, but along the borders of the zone.

{{xplique.utils_functions.segmentation.get_in_out_border}}



### `get_common_border`

This function uses two borders computed via the previous function and limits the zone of interest to the common part between both zone of interest. The classes of interest are merged, thus creating a second class of interest. Therefore, this function enables the creation of `targets` to explain the [border between two objects](#the-border-between-two-objects).

{{xplique.utils_functions.segmentation.get_common_border}}






## What can be explained with it? ##

There are many things that we may want to explain in semantic segmentation, and in this section present different possibilities. The [segmentation utils functions](#the-segmentation-utils-functions) allow the design of the [`targets` parameter](#the-targets-parameter) to specify what to explain.

!!!warning
    The concept **object** does not make sense for the model, a semantic segmentation model only classifies pixels. However, what humans want to explain are mainly objects, sets of objects or parts of them.

!!!info
    As objects do not make sense for the model, to stay coherent when manipulating objects. The only condition is that the predicted class on this connected zone is the same for all pixels.

For a concrete example, please refer to the [Attributions: Semantic segmentation](https://colab.research.google.com/drive/1AHg7KO1fCOX5nZLGZfxkZ2-DLPPdSfbX) tutorial.



### The class of an object ###

Here an object can be a person walking on a street, the dog by his side or a car.

However, what humans call an object does not make sense for model, hence explaining an object corresponds to explaining a zone of interest where pixels have the same classification.

!!!warning
    The zone should be extracted from the model's prediction and not the labels.

To explain the difference between labels and predictions there are two possibilities:

- either the difference is a single zone with a different class than the surroundings, then this zone can be considered an object.
- or the difference is more complex or mixed with other objects. Then the zones in the union but not in the intersection of both should be iteratively considered objects and explained. It is not recommended to treat them simultaneously.



### The class of a set of objects ###

A set of objects can be a group of people walking down a street or a set of trees on one side of the road.

There are three cases that can be considered set of objects:

- Connected set of objects, it can be seen as only one big zone and treated the same as in [1.](#the-class-of-an-object)
- Locally close set of objects, this could also considered a big zone, but it is harder to compute.
- Set of objects dispersed on the image and hardly countable, if there are a multitude of objects then, it can be seen as a [crowd of objects](#the-class-of-a-crowd-of-objects). Otherwise, it should not be treated together.



### The class of part of an object ###

A part of an object can be the leg of a person, the head of a dog, or a person in a group of people. This is interesting when the part and the object have been classified differently by the model. It should be considered an object as in [1.](#the-class-of-an-object)



### The class of a crowd of objects ###

A crowd is a set of hardly countable objects, it can be a set of clouds, a multitude of people on the sidewalk or trees in a landscape.



### The border of an object ###

The border of an object is the limit between the pixels inside the object and those outside of it. Here the object should correspond to a connected zone of pixels where the model predicts the same class.

It can be the contour of three people on the side walk or of trees on a landscape. It is interesting when the border is hard to define between similarly colored pixels or when the model prediction is not precise.



### The border between two objects ###

The border between two objects is the common part between two borders of objects when those two are connected. This can be the border between a person and his wrongly classified leg.





## Binary semantic segmentation ##

As described in the [operator description](#the-operator), the output of the model should have a shape of $(n, h, w, c)$. However, in binary semantic segmentation, the two classes are often encoded by positive and negative value along only one channel with shape $(n, h, w)$.

The easiest way to apply xplique on such model is to wrap the model to match the expected format. If we suppose that the output of the binary semantic segmentation model have a shape of $(n, h, w)$, that negative values encode class $0$, and that positive values encode class $1$. Then the wrapper can take the form:

```python
class Wrapper():
    def __init__(model):
        self.model = model

    def __call__(inputs):
        binary_segmentation = self.model(inputs)
        class_0_mask = binary_segmentation < 0
        divided = tf.stack([-binary_segmentation * tf.cast(class_0_mask, tf.float32),
                            binary_segmentation * tf.cast(tf.logical_not(class_0_mask), tf.float32)],
                           axis=-1)
        return tf.nn.softmax(divided, axis=-1)

wrapped_model = wrap(binary_seg_model)
```