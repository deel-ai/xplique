# Gradient $\odot$ Input

Gradient $\odot$ Input is a visualization techniques based on the gradient of a class score relative to 
the input, element-wise with the input. This method was introduced by Shrikumar et al., 2016[^1], in
an old version of their DeepLIFT paper[^2].

> Gradient inputs was at first proposed as a technique to improve the sharpness of the attribution maps.
> The attribution is computed taking the (signed) partial derivatives of the output with respect to 
> the input and multiplying them with the input itself.
>
> -- <cite>[Towards better understanding of the gradient-based attribution methods for Deep Neural Networks (2017)](https://arxiv.org/abs/1711.06104)</cite>[^3]

A theoretical analysis conducted by Ancona et al, 2018[^3] showed that Gradient $\odot$ Input is 
equivalent to $\epsilon$-LRP and DeepLift methods under certain conditions: using a baseline of zero, and with
all biases to zero.

More precisely, the explanation $E_x$ for an input $x$, for a given class $c$ is defined as

$$ E_x = x \odot \frac{\partial{S_c(x)}}{\partial{x}} $$

with $S_c$ the unormalized class score (layer before softmax).

## Examples

```python
from xplique.methods import GradientInput

# load images, labels and model
# ...

method = GradientInput(model)
explanations = method.explain(images, labels)
```

Using Gradient $\odot$ Input method on the layer before softmax (as recommended).
```python
from xplique.methods import GradientInput

"""
load images, labels and model
...

#Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, None, 512)     401920     
_________________________________________________________________
dense_2 (Dense)              (None, None, 10)      5130      
_________________________________________________________________
activation_1 (Activation)    (None, None, 10)      0         
=================================================================
Total params: 407,050
Trainable params: 407,050
Non-trainable params: 0
"""

# model target layer is dense_2, before activation
method = GradientInput(model, output_layer_index=-2)
explanations = method.explain(images, labels)
```

{{xplique.methods.gradient_input.GradientInput}}

[^1]: [Not Just a Black Box: Learning Important Features Through Propagating Activation Differences](https://arxiv.org/abs/1605.01713)
[^2]: [Learning Important Features Through Propagating Activation Differences](https://arxiv.org/abs/1704.02685)
[^3]: [Towards better understanding of gradient-based attribution methods for Deep Neural Networks](https://arxiv.org/abs/1711.06104)