# VarGrad

Similar to SmoothGrad, VarGrad a variance analog of SmoothGrad, and can be defined as follows:

$$
\phi_x = \underset{\xi ~\sim\ \mathcal{N}(0, \sigma^2)}{\mathcal{V}}
                    \Big{[}\frac { \partial{S_c(x + \xi)} } { \partial{x} }\Big{]}
$$

with $S_c$ the unormalized class score (layer before softmax). The $\sigma$ in the formula is controlled using the noise
parameter.

## Example

```python
from xplique.attributions import VarGrad

# load images, labels and model
# ...

method = VarGrad(model, nb_samples=50, noise=0.15)
explanations = method.explain(images, labels)
```

## Notebooks

- [**Attribution Methods**: Getting started](https://colab.research.google.com/drive
/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) </sub>
- [**VarGrad**: Going Further](https://colab.research.google.com/drive/1x_sNUM5xhAvzg1KmO5ZBlkxQpgxZyoux) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1x_sNUM5xhAvzg1KmO5ZBlkxQpgxZyoux) </sub>

{{xplique.attributions.VarGrad}}
