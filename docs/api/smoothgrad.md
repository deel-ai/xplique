# SmoothGrad

SmoothGrad is a gradient-based explanation method, which, as the name suggests, averages the 
gradient at several points corresponding to small perturbations around the point of interest. 
The smoothing effect induced by the average help reducing the visual noise, and hence improve the
explanations.

> \[...] The gradient at any given point will be less meaningful than a local average of gradient 
> values. This suggests a new way to create improved sensitivity maps: instead of basing a 
> visualization directly on the gradient, we could base it on a smoothing of the gradients with a 
> Gaussian kernel.
>
> --<cite>[SmoothGrad: removing noise by adding noise (2017)](https://arxiv.org/abs/1706.03825)</cite>[^1]


More precisely, the explanation $E_x$ for an input $x$, for a given class $c$ is defined as

$$ E_x = \mathbb{E}_{\epsilon \tilde\ \mathcal{N}(0, \sigma^2)} \Big{[} \frac { \partial{S_c(x + \epsilon)} } { \partial{x + \epsilon} } \Big{]} $$

with $S_c$ the unormalized class score (layer before softmax).

## Examples

```python
from xplique.methods import IntegratedGradients

# load images, labels and model
# ...

method = IntegratedGradients(model, steps=50, baseline_value=0.0)
explanations = method.explain(images, labels)
```

Using Integrated gradients method on the layer before softmax (as recommended).
```python
from xplique.methods import IntegratedGradients

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
# leaving baseline_value as default (0.0)
method = IntegratedGradients(model, output_layer_index=-2, steps=50)
explanations = method.explain(images, labels)
```


{{xplique.methods.smoothgrad.SmoothGrad}}

[^1]: [SmoothGrad: removing noise by adding noise (2017)](https://arxiv.org/abs/1706.03825)