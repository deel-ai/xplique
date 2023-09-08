# Regression explanations with Xplique

[Attributions: Regression and Tabular data tutorial](https://colab.research.google.com/drive/1pjDJmAa9oeSquYtbYh6tksU6eTmObIcq) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pjDJmAa9oeSquYtbYh6tksU6eTmObIcq) </sub>





## Which kind of tasks are supported by Xplique?

With the [operator's api](../api/attributions/operator) you can treat many different problems with Xplique. There is one operator for each task.

| Task and Documentation link                        | `operator` parameter value <br/> from `xplique.Tasks` Enum  | Tutorial link |
| :------------------------------------------------- | :---------------------------------------------------------- | :------------ |
| [Classification](../classification/)               | `CLASSIFICATION`        | <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) </sub> |
| [Object Detection](../object_detection/)           | `OBJECT_DETECTION`      | <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1X3Yq7BduMKqTA0XEheoVIpOo3IvOrzWL) </sub> |
| **Regression**                                     | `REGRESSION`            | <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pjDJmAa9oeSquYtbYh6tksU6eTmObIcq) </sub> |
| [Semantic Segmentation](../semantic_segmentation/) | `SEMANTIC_SEGMENTATION` | <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AHg7KO1fCOX5nZLGZfxkZ2-DLPPdSfbX) </sub> |

!!!info
    They all share the [API for Xplique attribution methods](../api_attributions/).

!!!warning
    In Xplique, for now with regression, predictions can only be explained output by output. Indeed, explaining several output simultaneously brings new problematic and we are currently working on an operator to solve this.





## Simple example

```python
import xplique
from xplique.attributions import Saliency
from xplique.metrics import Deletion

# load inputs and model
# ...

# for regression, `targets` indicates the output of interest, here output 3
targets = tf.one_hot([2], depth=nb_outputs, axis=-1)

# compute explanations by specifying the regression operator
explainer = Saliency(model, operator=xplique.Tasks.REGRESSION)
explanations = explainer(inputs, targets)

# compute metrics on these explanations
metric = Deletion(model, inputs, targets, operator=xplique.Tasks.REGRESSION)
score_saliency = metric(explanations)
```





## How to use it?

To apply attribution methods, the [**common API documentation**](../api_attributions/) describes the parameters and how to fix them. However, depending on the task and thus on the `operator`, there are three points that vary:

- **[The `operator` parameter](#the-operator)** value, it is an Enum or a string identifying the task,

- **[The model's output](#models-output)** specification, as `model(inputs)` is used in the computation of the operators, and

- **[The `targets` parameter](#the-targets-parameter)** format, indeed, the `targets` parameter specifies what to explain and the format of such specification depends on the task.





## The `operator` ##

### How to specify it

In Xplique, to adapt attribution methods, you should specify the task to the `operator` parameter. In the case of regression, with either:
```python
Method(model, operator="regression")
# or
Method(model, operator=xplique.Tasks.REGRESSION)
```



### The computation

The regression operator works similarly to the classification operator, it asks for the output of interest via `targets` and returns this output. See [targets section](#the-targets-parameter) for more detail.
```python
scores = tf.reduce_sum(model(inputs) * targets, axis=-1)
```



### The behavior

- In the case of [perturbation-based methods](../api_attributions/#gradient-based-approaches), the perturbation score corresponds to the difference between the initial value of the output of interest and the same output for predictions over perturbed inputs.
- For [gradient-based methods](../api_attributions/#perturbation-based-approaches), the gradient of the model's predictions for the output of interest.





## Model's output ##

We expect `model(inputs)` to yield a $(n, d)$ tensor or array where $n$ is the number of input samples and $d$ is the number of variables the model should predict (possibly one). 





## The `targets` parameter ##

### Role

The `targets` parameter specifies what is to explain in the `inputs`, it is passed to the `explain` or to the `__call__` method of an explainer or metric and used by the operators. In the case of regression it indicates which of the output should be explained.



### Format

The `targets` parameter in the case of regression should have the same shape as the [model's output](#models-output) as they are multiplied. Hence, the shape is $(n, d)$ with $n$ the number of samples to be explained (it should match the first dimension of `inputs`) and $d$ is the number of variables (possibly one).


### In practice

In the [simple example](#simple-example), the `targets` value provided is computed with `tf.one_hot`. Indeed, the regression operator takes as `targets` the one hot encoding of the index of the output to explain.
