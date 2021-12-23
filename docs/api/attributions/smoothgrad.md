# SmoothGrad

SmoothGrad is a gradient-based explanation method, which, as the name suggests, averages the
gradient at several points corresponding to small perturbations around the point of interest.
The smoothing effect induced by the average help reducing the visual noise, and hence improve the
explanations.

!!! quote
    \[...] The gradient at any given point will be less meaningful than a local average of gradient
    values. This suggests a new way to create improved sensitivity maps: instead of basing a
    visualization directly on the gradient, we could base it on a smoothing of the gradients with a
    Gaussian kernel.

    -- <cite>[SmoothGrad: removing noise by adding noise (2017)](https://arxiv.org/abs/1706.03825)</cite>[^1]


More precisely, the explanation $\phi_x$ for an input $x$, for a given class $c$ is defined as

$$ \phi_x = \underset{\xi ~\sim~ \mathcal{N}(0, \sigma^2)}{\mathbb{E}}
            \Big{[}
             \frac { \partial{S_c(x + \xi)} } { \partial{x} }
             \Big{]} $$

with $S_c$ the unormalized class score (layer before softmax). The $\sigma$ in the formula is controlled using the noise
parameter, and the expectation is estimated using multiple samples.

## Example

```python
from xplique.attributions import SmoothGrad

# load images, labels and model
# ...

method = SmoothGrad(model, nb_samples=50, noise=0.15)
explanations = method.explain(images, labels)
```

## Notebooks

- [**Attribution Methods**: Getting started](https://colab.research.google.com/drive
/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) </sub>
- [**SmoothGrad**: Going Further](https://colab.research.google.com/drive/1kHPB07XExQ0CvTIk51Hm5qS2Vl66OUH2) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kHPB07XExQ0CvTIk51Hm5qS2Vl66OUH2) </sub>

{{xplique.attributions.smoothgrad.SmoothGrad}}

[^1]: [SmoothGrad: removing noise by adding noise (2017)](https://arxiv.org/abs/1706.03825)
