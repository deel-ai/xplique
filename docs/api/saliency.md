# Saliency Maps

Saliency is visualization techniques based on the gradient of a class score relative to the
input.

!!! quote
    An interpretation of computing the image-specific class saliency using the class score derivative
    is that the magnitude of the derivative indicates which pixels need to be changed the least
    to affect the class score the most. One can expect that such pixels correspond to the object location
    in the image. 
    
    -- <cite>[Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps (2013)](https://arxiv.org/abs/1312.6034)</cite>[^1]

More precisely, the explanation $\phi_x$ for an input $x$, for a given class $c$ is defined as

$$ \phi_x = \Big{|}\frac{\partial{S_c(x)}}{\partial{x}}\Big{|} $$

with $S_c$ the unormalized class score (layer before softmax).

## Example

```python
from xplique.methods import Saliency

# load images, labels and model
# ...

method = Saliency(model)
explanations = method.explain(images, labels)
```

{{xplique.methods.saliency.Saliency}}

[^1]:[Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034)