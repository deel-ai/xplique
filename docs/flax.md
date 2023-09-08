# PyTorch's model with Xplique

- [**PyTorch's model**: Getting started](https://colab.research.google.com/drive/1bMlO29_0K3YnTQBbbyKQyRfo8YjvDbhe)<sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bMlO29_0K3YnTQBbbyKQyRfo8YjvDbhe) </sub>

- [**Metrics**: With Pytorch's model](https://colab.research.google.com/drive/16bEmYXzLEkUWLRInPU17QsodAIbjdhGP) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16bEmYXzLEkUWLRInPU17QsodAIbjdhGP) </sub>

!!!note
    We should point out that what we did with Flax should be possible for other frameworks. Do not hesitate to give it a try and to make a PR if you have been successful!

## Is it possible to use Xplique with Jax/Flax's model ?

**Yes**, it is! Even though the library was mainly designed to be a Tensorflow toolbox we have been working on a very practical wrapper to facilitate the integration of your Flax's model into Xplique's framework!

### Quickstart
```python
import flax.linen as nn

from xplique.wrappers import FlaxWrapper
from xplique.attributions import Saliency
from xplique.metrics import Deletion

# load images, targets, model architecture and model parameters
# ...
flax_module = nn.Sequential(...)
params = flax_module.init(jax.random.PRNGKey(0), inputs)
wrapped_model = FlaxWrapper(flax_module, params)

explainer = Saliency(wrapped_model)
explanations = explainer(inputs, targets)

metric = Deletion(wrapped_model, inputs, targets)
score_saliency = metric(explanations)
```

## Does it work for every modules ?

It has been tested on both the `attributions` and the `metrics` modules.

## Does it work for all attribution methods ?

Not yet, but it works for most of them (even for gradient-based ones!):

| **Attribution Method** | Flax/Jax compatible |
| :--------------------- | :----------------: |
| Deconvolution          | ❌                |
| Grad-CAM               | ❌                |
| Grad-CAM++             | ❌                |
| Gradient Input         | ✅                |
| Guided Backprop        | ❌                |
| Hsic Attribution       | ✅                |
| Integrated Gradients   | ✅                |
| Kernel SHAP            | ✅                |
| Lime                   | ✅                |
| Occlusion              | ✅                |
| Rise                   | ✅                |
| Saliency               | ✅                |
| SmoothGrad             | ✅                |
| Sobol Attribution      | ✅                |
| SquareGrad             | ✅                |
| VarGrad                | ✅                |

##  Steps to make Xplique work on pytorch

###  1. Make sure the inputs follow the Xplique API (and not what the model expects).

One thing to keep in mind is that **attribution methods expect a specific inputs format** as described in the [API Description](api/attributions/api_attributions.md). Especially, for images `inputs` should be $(N, H, W, C)$ following the TF's conventions where:

- $N$ is the number of inputs
- $H$ is the height of the images
- $W$ is the width of the images
- $C$ is the number of channels
  
Fortunately, the default format expected by Flax by default is $(N, H, W, C)$ just like Tensorflow.  
  
If you are using PyTorch's preprocessing functions what you should do is:

- preprocess as usual
- convert the data to numpy array

!!!tip
    If you want to be sure how this work you can look at the [**PyTorch's model**: Getting started](https://colab.research.google.com/drive/1bMlO29_0K3YnTQBbbyKQyRfo8YjvDbhe) notebook and compare it to the [**Attribution methods**:Getting Started](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2)

### 2. Wrap your model

A `FlaxWrapper` object can be initialized with 2 parameters:

- `flax_module: flax.linen.Module`: A Flax module with `.apply()` method that expects inputs and parameters as arguments
- `parameters: Any`: A Pytree (i.e a structure of nested dicts, lists, and tuples) containing the parameters of the model, as returned by `flax_module.init()`. This pytree is expected to contain at least a `params` key, but it may contain other keys as well (e.g. `batch_stats` for batch normalization layers).

Indeed, contrary to Tensorflow and Pytorch that are stateful frameworks, Jax/Flax is stateless. This means that the model's parameters are not stored in the model itself but in a separate object. This is why we need to provide the parameters to the wrapper in order to be able to use the model.
  
!!!info
    It is possible that you used special treatments for your models or that it does not follow typical convention. In that case, we encourage you to have a look at the <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20"></sub>[Source Code](https://github.com/deel-ai/xplique/blob/master/xplique/wrappers/flax.py) to adapt it to your needs.

### 3. Use this wrapped model as a TF's one

## What are the limitations ?

As it was previously mentionned this does not work with: Deconvolution, Grad-CAM, Grad-CAM++ and Guided Backpropagation.

Furthermore, when one use any white-box explainers one have the possibility to provide an `output_layer` parameter. This functionnality will not work with PyTorch models. The user will have to manipulate itself its model!

!!!warning
    The `output_layer` parameter does not work for PyTorch's model!

It is possible that all failure cases were not covered in the tests, in that case please open an issue so the team will work on it!