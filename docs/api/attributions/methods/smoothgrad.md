# SmoothGrad

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20">
</sub>[View colab tutorial](https://colab.research.google.com/drive/12-tlM_TdZ12oc5lNL2S2g-hcMJV8tZUD) |
<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source](https://github.com/deel-ai/xplique/blob/master/xplique/attributions/smoothgrad.py) |
📰 [Paper](https://arxiv.org/abs/1706.03825)


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


More precisely, the explanation $\phi$ for an input $x$ and a classifier $f$ is defined as

$$
\phi = \mathbb{E}_{\delta ~\sim~ \mathcal{N}(0, \sigma^2) }( \nabla_x f(x + \delta) ) 
\approx \frac{1}{N} \sum_{i=0}^N \nabla_x f(x + \delta_i)
$$

The $\sigma$ in the formula is controlled using the `noise`
parameter, and the expectation is estimated using $N$ samples controlled by the `nb_samples` parameter.

!!!tip
    It is recommended to have a noise level $\sigma$ at about 20% of the range of your inputs, i.e. $\sigma=0.2$ if your inputs are between $[0, 1]$ or $\sigma=0.4$ if your inputs are between $[-1, 1]$.

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
/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2)
- [**SmoothGrad**: Going Further](https://colab.research.google.com/drive/12-tlM_TdZ12oc5lNL2S2g-hcMJV8tZUD)

{{xplique.attributions.gradient_statistics.SmoothGrad}}

[^1]: [SmoothGrad: removing noise by adding noise (2017)](https://arxiv.org/abs/1706.03825)
