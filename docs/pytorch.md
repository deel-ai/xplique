# PyTorch's model with Xplique

- [**PyTorch's model**: Getting started](https://colab.research.google.com/drive/1bMlO29_0K3YnTQBbbyKQyRfo8YjvDbhe)<sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bMlO29_0K3YnTQBbbyKQyRfo8YjvDbhe) </sub>

- [**Metrics**: With Pytorch's model](https://colab.research.google.com/drive/16bEmYXzLEkUWLRInPU17QsodAIbjdhGP) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16bEmYXzLEkUWLRInPU17QsodAIbjdhGP) </sub>

!!!note
    We should point out that what we did with Pytorch should be possible for other frameworks. Do not hesitate to give it a try and to make a PR if you have been successful!

## Is it possible to use Xplique with PyTorch's model ?

**Yes**, it is! Even though the library was mainly designed to be a Tensorflow toolbox we have been working on a very practical wrapper to facilitate the integration of your PyTorch's model into Xplique's framework!

### Quickstart
```python
import torch

from xplique.wrappers import TorchWrapper
from xplique.attributions import Saliency
from xplique.metrics import Deletion

# load images, targets and model
# ...

device = 'cuda' if torch.cuda.is_available() else 'cpu'
wrapped_model = TorchWrapper(torch_model, device)

explainer = Saliency(wrapped_model)
explanations = explainer(inputs, targets)

metric = Deletion(wrapped_model, inputs, targets)
score_saliency = metric(explanations)
```

## Does it work for every modules ?

It has been tested on both the `attributions` and the `metrics` modules.

## Does it work for all attribution methods ?

Not yet, but it works for most of them (even for gradient-based ones!):

| **Attribution Method** | PyTorch compatible |
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

## How does it work ?

### 1. Pre-process the inputs

One thing to keep in mind is that **attribution methods expect a specific inputs format** as described in the [API Description](api/attributions/api_attributions.md). Especially, for images `inputs` should be $(N, H, W, C)$ following the TF's conventions where:

- $N$ is the number of inputs
- $H$ is the height of the images
- $W$ is the width of the images
- $C$ is the number of channels

However, if you are using a PyTorch's model it is most likely expecting images' shape to be $(N, C, H, W)$. So what should you do ?

If you are using PyTorch's preprocessing functions what you should do is:

- preprocess as usual
- convert the data to numpy array
- use `np.moveaxis(np_inputs, [1, 2, 3], [3, 1, 2])` to change shape from $(N, C, H, W)$ to $(N, H, W, C)$

!!!notes
    The third step is necessary only if your data has a `channel` dimension which is not in the place expected with Tensorflow

!!!tip
    If you want to be sure how this work you can look at the [**PyTorch's model**: Getting started](https://colab.research.google.com/drive/1bMlO29_0K3YnTQBbbyKQyRfo8YjvDbhe)<sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bMlO29_0K3YnTQBbbyKQyRfo8YjvDbhe) </sub> notebook and compare it to the [**Attribution methods**:Getting Started](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) </sub>

### 2. Create your wrapper

A `TorchWrapper` object can be initialized with 3 parameters:

- `torch_model: torch.nn.Module`: A torch's model that inherits from nn.Module
- `device: Union['torch.device', str]`: The device on which the torch's model and inputs should be mounted
- `is_channel_first: Optional[bool] = None`: A boolean that is true if the torch's model expect a channel dim and if this one come first

The last parameter is the one that needs special care. Indeed, if it is set to `True` **we assume** that the **torch model** expects its inputs to be $(N, C, H, W)$. As the explainer **requires** inputs to be $(N, H, W, C)$ we change the inputs' axis order when a call is made to the wrapped model (transparently for the user). If it is set to `False` we do not move the axis at all. By default the wrapper is looking for `torch.nn.Conv2d` layers in the torch model and consider it is channel first if it finds one and not otherwise.

!!!info
    It is possible that you used special treatments for your models or that it does not follow typical convention. In that case, we encourage you to have a look at the <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20"></sub>[Source Code](https://github.com/deel-ai/xplique/blob/master/xplique/wrappers/pytorch.py) to adapt it to your needs.

### 3. Use this wrapped model as a TF's one

## What are the limitations ?

As it was previously mentionned this does not work with: Deconvolution, Grad-CAM, Grad-CAM++ and Guided Backpropagation.

Furthermore, when one use any white-box explainers one have the possibility to provide an `output_layer` parameter. This functionnality will not work with PyTorch models. The user will have to manipulate itself its model!

!!!warning
    The `output_layer` parameter does not work for PyTorch's model!

It is possible that all failure cases were not covered in the tests, in that case please open an issue so the team will work on it!