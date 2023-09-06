# Occlusion sensitivity

<sub><img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20"></sub>[ View colab tutorial](https://colab.research.google.com/drive/15xmmlxQkNqNuXgHO51eKogXvLgs-sG4q) | <sub><img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20"></sub>[ View source](https://github.com/deel-ai/xplique/blob/master/xplique/attributions/occlusion.py)

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
from xplique.attributions import Occlusion

# load images, labels and model
# ...

method = Occlusion(model, patch_size=(10, 10),
                   patch_stride=(2, 2), occlusion_value=0.5)
explanations = method.explain(images, labels)
```

## Notebooks

- [**Attribution Methods**: Getting started](https://colab.research.google.com/drive
/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2)
- [**Occlusion**: Going Further](https://colab.research.google.com/drive/15xmmlxQkNqNuXgHO51eKogXvLgs-sG4q)

{{xplique.attributions.occlusion.Occlusion}}

[^1]: [Visualizing and Understanding Convolutional Networks (2014).](https://arxiv.org/abs/1311.2901)
[^2]: [Towards better understanding of gradient-based attribution methods for Deep Neural Networks](https://arxiv.org/abs/1711.06104)

!!!info
    `patch_size` and `patch_stride` will define patch to apply to the original input. Thus, a combination of patches will generate pertubed samples of the original input (masked by patches with `occlusion_value` value).
    Consequently, the number of pertubed instances of an input depend on those parameters. Too little value of those two arguments on large image might lead to an incredible amount of pertubed samples and increase compuation time. On another hand too huge values might not be accurate enough.
