# Gradient $\odot$ Input

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20">
</sub>[View colab tutorial](https://colab.research.google.com/drive/19eB3uwAtCKZgkoWtMzrF0LTJ-htF_KE7) |
<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source](https://github.com/deel-ai/xplique/blob/master/xplique/attributions/gradient_input.py) |
📰 [Paper](https://arxiv.org/abs/1605.01713)


Gradient $\odot$ Input is a visualization techniques based on the gradient of a class score relative to
the input, element-wise with the input. This method was introduced by Shrikumar et al., 2016[^1], in
an old version of their DeepLIFT paper[^2].

!!! quote
    Gradient inputs was at first proposed as a technique to improve the sharpness of the attribution maps.
    The attribution is computed taking the (signed) partial derivatives of the output with respect to
    the input and multiplying them with the input itself.

    -- <cite>[Towards better understanding of the gradient-based attribution methods for Deep Neural Networks (2017)](https://arxiv.org/abs/1711.06104)</cite>[^3]

A theoretical analysis conducted by Ancona et al, 2018[^3] showed that Gradient $\odot$ Input is
equivalent to $\epsilon$-LRP and DeepLift methods under certain conditions: using a baseline of zero, and with
all biases to zero.

More precisely, the explanation $\phi$ for an input $x$ and a classifier $f$ is defined as

$$ \phi = x \odot \nabla_x f(x) $$

with $\odot$ the Hadamard product.

## Example

```python
from xplique.attributions import GradientInput

# load images, labels and model
# ...

method = GradientInput(model)
explanations = method.explain(images, labels)
```

## Notebooks

- [**Attribution Methods**: Getting started](https://colab.research.google.com/drive
/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) </sub>
- [**Gradient $\odot$ Input**: Going Further](https://colab.research.google.com/drive/19eB3uwAtCKZgkoWtMzrF0LTJ-htF_KE7) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19eB3uwAtCKZgkoWtMzrF0LTJ-htF_KE7) </sub>

{{xplique.attributions.gradient_input.GradientInput}}

[^1]: [Not Just a Black Box: Learning Important Features Through Propagating Activation Differences](https://arxiv.org/abs/1605.01713)
[^2]: [Learning Important Features Through Propagating Activation Differences](https://arxiv.org/abs/1704.02685)
[^3]: [Towards better understanding of gradient-based attribution methods for Deep Neural Networks](https://arxiv.org/abs/1711.06104)
