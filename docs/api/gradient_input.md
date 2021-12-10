# Gradient $\odot$ Input

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

More precisely, the explanation $\phi_x$ for an input $x$, for a given class $c$ is defined as

$$ \phi_x = x \odot \frac{\partial{S_c(x)}}{\partial{x}} $$

with $S_c$ the unormalized class score (layer before softmax).

## Example

```python
from xplique.attributions import GradientInput

# load images, labels and model
# ...

method = GradientInput(model)
explanations = method.explain(images, labels)
```

{{xplique.attributions.gradient_input.GradientInput}}

[^1]: [Not Just a Black Box: Learning Important Features Through Propagating Activation Differences](https://arxiv.org/abs/1605.01713)
[^2]: [Learning Important Features Through Propagating Activation Differences](https://arxiv.org/abs/1704.02685)
[^3]: [Towards better understanding of gradient-based attribution methods for Deep Neural Networks](https://arxiv.org/abs/1711.06104)
