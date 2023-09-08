# Classification explanations with Xplique

[Attributions: Getting started tutorial](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) </sub>





## Which kind of tasks are supported by Xplique?

With the [operator's api](../api/attributions/operator) you can treat many different problems with Xplique. There is one operator for each task.

| Task and Documentation link                        | `operator` parameter value <br/> from `xplique.Tasks` Enum  | Tutorial link |
| :------------------------------------------------- | :---------------------------------------------------------- | :------------ |
| **Classification**                                 | `CLASSIFICATION`        | <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) </sub> |
| [Object Detection](../object_detection/)           | `OBJECT_DETECTION`      | <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1X3Yq7BduMKqTA0XEheoVIpOo3IvOrzWL) </sub> |
| [Regression](../regression/)                       | `REGRESSION`            | <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pjDJmAa9oeSquYtbYh6tksU6eTmObIcq) </sub> |
| [Semantic Segmentation](../semantic_segmentation/) | `SEMANTIC_SEGMENTATION` | <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AHg7KO1fCOX5nZLGZfxkZ2-DLPPdSfbX) </sub> |

!!!info
    They all share the [API for Xplique attribution methods](../api_attributions/).





## Simple example

```python
import xplique
from xplique.attributions import Saliency
from xplique.metrics import Deletion

# load inputs and model
# ...

# for classification it is recommended to remove softmax layer if there is one
# model.layers[-1].activation = tf.keras.activations.linear

# for classification, `targets` are the one hot encoding of the predicted class
targets = tf.one_hot(tf.argmax(model(inputs), axis=-1), depth=nb_classes, axis=-1)

# compute explanations by specifying the classification operator
explainer = Saliency(model, operator=xplique.Tasks.CLASSIFICATION)
explanations = explainer(inputs, targets)

# compute metrics on those explanations
# if the softmax was removed,
# it is possible to specify it to obtain more interpretable metrics
metric = Deletion(model, inputs, targets,
                  operator=xplique.Tasks.CLASSIFICATION, activation="softmax")
score_saliency = metric(explanations)
```

!!!tip
    In general, if you are doing classification tasks, it is better to not include the final softmax layer in your model but to work with logits instead!





## How to use it?

To apply attribution methods, the [**common API documentation**](../api_attributions/) describes the parameters and how to fix them. However, depending on the task and thus on the `operator`, there are three points that vary:

- **[The `operator` parameter](#the-operator)** value, it is an Enum or a string identifying the task,

- **[The model's output](#models-output)** specification, as `model(inputs)` is used in the computation of the operators, and

- **[The `targets` parameter](#the-targets-parameter)** format, indeed, the `targets` parameter specifies what to explain and the format of such specification depends on the task.





## The `operator` ##

### How to specify it

In Xplique, to adapt attribution methods, you should specify the task to the `operator` parameter. In the case of classification, with either:
```python
Method(model)
# or
Method(model, operator="classification")
# or
Method(model, operator=xplique.Tasks.CLASSIFICATION)
```

!!!info
    Classification if the default behavior of Xplique attribution methods, hence there is no need to specify it. Nonetheless, it is recommended to still do so to ensure a good comprehension of what is explained.



### The computation

The classification operator multiplies model's predictions on `inputs` with `targets` and sum it for each input to explain. However, only one value should be non-zero in `targets`, thus, the classification operator returns the model output for the specified (via `targets`) class.
```python
scores = tf.reduce_sum(model(inputs) * targets, axis=-1)
```



### The behavior

- In the case of [perturbation-based methods](../api_attributions/#gradient-based-approaches), the perturbation score corresponds to the difference between the initial logits value for the predicted classes and the same logits for predictions over perturbed inputs.
- For [gradient-based methods](../api_attributions/#perturbation-based-approaches), the gradient of logits of interest with respect to the inputs.

The logits of interest are specified via the `targets` parameter described in [the related section](#the-targets-parameter).





## Model's output ##

We expect `model(inputs)` to yield a $(n, c)$ tensor or array where $n$ is the number of input samples and $c$ is the number of classes. 





## The `targets` parameter ##

### Role

The `targets` parameter specifies what is to explain in the `inputs`, it is passed to the `explain` or to the `__call__` method of an explainer or metric and used by the operators. In the case of classification, it indicates the class to explain, or specifies [contrastive explanations](#contrastive-explanations).



### Format

The `targets` parameter in the case of classification should have the same shape as the [model's output](#models-output) as they are multiplied point-wise. Hence, the shape is $(n, c)$ with $n$ the number of samples to be explained (it should match the first dimension of `inputs`) and $c$ the number of classes. The `targets` parameter expects values among ${-1, 0, 1}$ but most values should be $0$ and most of the time only one should be $1$ for each sample. $-1$ are only used for [contrastive explanations](#contrastive-explanations).



### In practice

In the [simple example](#simple-example), the `targets` value provided is computed with `tf.one_hot(tf.argmax(model(inputs), axis=-1), axis=-1)`. Literally, the one hot encoding of the predicted class, this specifies which class to explain.

!!!tip
    It is better to explain the predicted class than the expected class as the goal is to explain the model's prediction.





## What can be explained with it? ##

### Explain the predicted class ###

By specifying `targets` with a one hot encoding of the predicted class, the explanation will highlight which features were important for this prediction.



### Contrastive explanations ###

By specifying `targets` with zeros everywhere, `1` for the first class, and `-1` for the second class. The explanation will show which features were important to predict the first and and not the second one.

!!!tip
    If the model made a mistake, an interesting explanation is predicted class versus expected class.