# API: Attributions Methods

- [**Attribution Methods**: Getting started](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) </sub>



## Context
In 2013, [Simonyan et al.](http://arxiv.org/abs/1312.6034) proposed a first attribution method, opening the way to a wide range of approaches which could be defined as follow:

!!!definition
    The main objective in attributions techniques is to highlight the discriminating variables for decision-making. For instance, with Computer Vision (CV) tasks, the main goal is to underline the pixels contributing the most in the input image(s) leading to the model’s output(s).

## Common API

All attribution methods inherit from the Base class `BlackBoxExplainer`. This base class can be instanciated with three parameters:

- `model`: the model from which we want to obtain attributions (e.g: InceptionV3, ResNet, ...), see the [model expectations](../model/) for more details
- `batch_size`: an integer which allows to either process inputs per batch (gradient-based methods) or process perturbed samples of an input per batch (inputs are therefore process one by one)
- `operator`: function g to explain, see the [Operator documentation](../operator/) for more details

In addition, all class inheriting from `BlackBoxExplainer` should implement an `explain` method:

```python
@abstractmethod
def explain(self,
            inputs: Union[tf.data.Dataset, tf.Tensor, np.array],
            targets: Optional[Union[tf.Tensor, np.array]] = None) -> tf.Tensor:
    raise NotImplementedError()

def __call__(self,
             inputs: tf.Tensor,
             labels: tf.Tensor) -> tf.Tensor:
    """Explain alias"""
    return self.explain(inputs, labels)
```

`inputs`: Must be one of the following: a `tf.data.Dataset` (in which case you should not provide targets), a `tf.Tensor` or a `np.ndarray`.

- If inputs are images, the expected shape of `inputs` is $(N, H, W, C)$ following the TF's conventions where:
    - $N$ is the number of inputs
    - $H$ is the height of the images
    - $W$ is the width of the images
    - $C$ is the number of channels (works for $C=3$ or $C=1$, other values might not work or need further customization)

- If inputs are tabular data, the expected shape of `inputs` is $(N, W)$ where:
    - $N$ is the number of inputs
    - $W$ is the feature dimension of a single input

    !!!tip
        Please refer to the [table](../../../#whats-included) to see which methods might work with Tabular Data

- (Experimental) If inputs are Time Series, the expected shape of `inputs` is $(N, T, W)$
    - $N$ is the number of inputs
    - $T$ is the temporal dimension of a single input
    - $W$ is the feature dimension of a single input

        !!!warning
            By default `Lime` & `KernelShap` will treat such inputs as grey images. You will need to define a custom `map_to_interpret_space` function when instantiating those methods in order to create a meaningful mapping of Time-Series data into an interpretable space when building such explainers.

    !!!note
        If your model is not following the same conventions, please refer to the [Model documentation](../model/).

`targets`: Must be one of the following: a `tf.Tensor` or a `np.ndarray`. Its format depends on the task at end and what you want an explanation of. Thus, if you want to know more about `targets` you should read the documentation of [Model](../model/#tasks) or [Operator](../operator/#tasks).

Even though we made an harmonized API for all attributions methods it might be relevant for the user to distinguish Gradient based and Perturbation based methods, also often referenced respectively as white-box and black-box methods, as their hyperparameters settings might be quite different.

## Perturbation-based approaches

Perturbation based methods focus on perturbing an input with a variety of techniques and, with the analysis of the resulting outputs, define an attribution representation. Thus, **there is no need to explicitly know the model architecture** as long as forward pass is available, which explain why they are also referenced as black-box methods.

Therefore, to use perturbation-based approaches you do not need a TF model. To know more, please see the [Callable](../../../callable) documentation.

Xplique includes the following black-box attributions:

| Method Name      | **Tutorial**             |
|:---------------- | :----------------------: |
| KernelShap              | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1frholXRE4XQQ3W5yZuPQ2-xqc-LTczfT) |
| Lime                    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1frholXRE4XQQ3W5yZuPQ2-xqc-LTczfT) |
| Occlusion               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15xmmlxQkNqNuXgHO51eKogXvLgs-sG4q) |
| Rise                    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1icu2b1JGfpTRa-ic8tBSXnqqfuCGW2mO) |
| Sobol Attribution       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) |
| Hsic Attribution        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) |

## Gradient-based approaches

Those approaches are also called white-box methods as **they require a full access to the model architecture**, notably it should **allow computing gradients**. Indeed, the core idea with the gradient-based approach is to use back-propagation, along other techniques, not to update the model’s weights (which is already trained) but to reveal the most contributing inputs, potentially in a specific layer. All methods are available when the model work with TensorFlow but most methods also works with PyTorch (see [Xplique for PyTorch documentation](../../../pytorch))

| Method Name      | **Tutorial**             | Available with TF | Available with PyTorch* |
|:---------------- | :----------------------: | :---------------: | :---------------------: |
| DeconvNet               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19eB3uwAtCKZgkoWtMzrF0LTJ-htF_KE7) | ✔ |❌ |
| GradCAM                 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nsB7xdQbU0zeYQ1-aB_D-M67-RAnvt4X) | ✔ |❌ |
| GradCAM++               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nsB7xdQbU0zeYQ1-aB_D-M67-RAnvt4X) | ✔ |❌ |
| GradientInput           | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19eB3uwAtCKZgkoWtMzrF0LTJ-htF_KE7) | ✔ | ✔ |
| GuidedBackpropagation   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19eB3uwAtCKZgkoWtMzrF0LTJ-htF_KE7) | ✔ | ❌ |
| IntegratedGradients     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UXJYVebDVIrkTOaOl-Zk6pHG3LWkPcLo) | ✔ | ✔ |
| Saliency                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19eB3uwAtCKZgkoWtMzrF0LTJ-htF_KE7) | ✔ | ✔ |
| SmoothGrad              | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12-tlM_TdZ12oc5lNL2S2g-hcMJV8tZUD) | ✔ | ✔ |
| SquareGrad              | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12-tlM_TdZ12oc5lNL2S2g-hcMJV8tZUD) | ✔ | ✔ |
| VarGrad                 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12-tlM_TdZ12oc5lNL2S2g-hcMJV8tZUD) | ✔ | ✔ |

*: Before using a PyTorch's model it is highly recommended to read the [dedicated documentation](../../../pytorch)

In addition, those methods inherits from `WhiteBoxExplainer` (itself inheriting from `BlackBoxExplainer`). Thus, an additional `__init__` argument is added: `output_layer`. It is the layer to target for the output (e.g logits or after softmax). If an `int` is provided, it will be interpreted as a layer index, if a `string` is provided it will look for the layer name. Default to the last layer.

!!!tip
    It is recommended to use the layer before Softmax.

!!!warning
    The `output_layer` parameter will work well with TensorFlow's model. However, it will not work with PyTorch's model. For PyTorch, one should directly manipulate the model to focus on the layers of interest.

!!!info
    The "white-box" explainers that work with Pytorch are those that only require the gradient of the model without having to "modify" some part of the model (e.g. Deconvnet will commute all original ReLU by a custom ReLU policy)