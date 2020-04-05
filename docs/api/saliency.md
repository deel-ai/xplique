# Saliency Maps

Saliency is visualization techniques based on the gradient of a class score relative to the
input.

> An interpretation of computing the image-specific class saliency using the class score derivative
> is that the magnitude of the derivative indicates which pixels need to be changed the least
> to affect the class score the most. One can expect that such pixels correspond to the object location
> in the image. 
>
> -- <cite>[Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps (2013)](https://arxiv.org/abs/1312.6034)</cite>[^1]

More precisely, the explanation $E_x$ for an input $x$, for a given class $c$ is defined as

$$ E_x = \Big{|}\frac{\partial{S_c(x)}}{\partial{x}}\Big{|} $$

with $S_c$ the unormalized class score (layer before softmax).
  
!!! note
    It should noted that for most methods of explanation it is highly recommended to use the 
    layer before softmax, as once the score has been normalized by softmax, maximizing the ouput can
    be achieved not only by increasing the score of the class in question, but also minimising 
    the score of all the others classes.

## Examples

```python
from xplique.methods import Saliency

# load images, labels and model
# ...

method = Saliency(model)
explanations = method.explain(images, labels)
```

Using Saliency method on the layer before softmax (as recommended in the original paper).
```python
from xplique.methods import Saliency

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
method = Saliency(model, output_layer_index=-2)
explanations = method.explain(images, labels)
```


{{xplique.methods.saliency.Saliency}}

[^1]:[Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034)