# Occlusion sensitivity

The Occlusion sensitivity method sweep a patch that occludes pixels over the
images, and use the variations of the model prediction to deduce critical areas.[^1]

!!! quote
    \[...] this method, referred to as Occlusion, replacing one feature $x_i$ at the time with a
     baseline and measuring the effect of this perturbation on the target output.
     
     -- <cite>[Towards better understanding of the gradient-based attribution methods for Deep Neural Networks (2017)](https://arxiv.org/abs/1711.06104)</cite>[^2]


with $S_c$ the unormalized class score (layer before softmax) and $\bar{x}$ a baseline, the Occlusion
sensitivity map $\phi$ is defined as :

$$ \phi_i = S_c(x) - S_c(x_{[x_i = \bar{x}]}) $$

## Example

```python
from xplique.methods import Occlusion

# load images, labels and model
# ...

method = Occlusion(model, patch_size=(10, 10), 
                   patch_stride=(2, 2), occlusion_value=0.5)
explanations = method.explain(images, labels)
```

{{xplique.methods.occlusion.Occlusion}}

[^1]:[Ref. Visualizing and Understanding Convolutional Networks (2014).](https://arxiv.org/abs/1311.2901)
[^2]: [Towards better understanding of gradient-based attribution methods for Deep Neural Networks](https://arxiv.org/abs/1711.06104)