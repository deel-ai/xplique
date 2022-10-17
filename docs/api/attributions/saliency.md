# Saliency Maps

<sub><img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20"></sub>[ View colab tutorial](https://colab.research.google.com/drive/19eB3uwAtCKZgkoWtMzrF0LTJ-htF_KE7?authuser=1) | <sub><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20"></sub>[ View source](https://github.com/deel-ai/xplique/blob/master/xplique/attributions/saliency.py)

Saliency is a visualization techniques based on the gradient of a class score relative to the
input.

!!! quote
    An interpretation of computing the image-specific class saliency using the class score derivative
    is that the magnitude of the derivative indicates which pixels need to be changed the least
    to affect the class score the most. One can expect that such pixels correspond to the object location
    in the image.

    -- <cite>[Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps (2013)](https://arxiv.org/abs/1312.6034)</cite>[^1]

More precisely, the explanation $\phi_x$ for an input $x$, for a given class $c$ is defined as

$$ \phi_x = \Big{\|} \frac{\partial{S_c(x)}}{\partial{x}} \Big{\|} $$

with $S_c$ the unormalized class score (layer before softmax).

## Example

```python
from xplique.attributions import Saliency

# load images, labels and model
# ...

method = Saliency(model)
explanations = method.explain(images, labels)
```

## Notebooks

- [**Attribution Methods**: Getting started](https://colab.research.google.com/drive
/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2)
- [**Saliency**: Going Further](https://colab.research.google.com/drive/19eB3uwAtCKZgkoWtMzrF0LTJ-htF_KE7?authuser=1)


{{xplique.attributions.saliency.Saliency}}

[^1]:[Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034)