# RISE

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20">
</sub>[View colab tutorial](https://colab.research.google.com/drive/1icu2b1JGfpTRa-ic8tBSXnqqfuCGW2mO) | 
<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source](https://github.com/deel-ai/xplique/blob/master/xplique/attributions/rise.py) |
📰 [Paper](https://arxiv.org/abs/1806.07421)

The RISE method consist of probing the model with randomly masked versions of the input image and
obtaining the corresponding outputs to deduce critical areas.

!!! quote
    \[...] we estimate the importance of pixels by dimming them in random combinations,
    reducing their intensities down to zero. We model this by multiplying an image with a \[0,1\]
    valued mask.

     -- <cite>[RISE: Randomized Input Sampling for Explanation of Black-box Models (2018)](https://arxiv.org/abs/1806.07421)</cite>[^1]


with $f(x)$ the prediction of a classifier, for an input $x$ and $m  \sim \mathcal{M}$ a mask with value in $[0,1]$ created from a low dimension ($m$ is in ${0, 1}^{w \times h}$ with $w \ll W$ and $h \ll H$ then upsampled, see the paper for more details).

The RISE importance estimator is defined as:

$$
\phi_i = \mathbb{E}( f(x \odot m) | m_i = 1) 
\approx \frac{1}{\mathbb{E}(\mathcal{M}) N} \sum_{i=1}^N f(x \odot m_i) m_i
$$

The most important parameters here are (1) the `grid_size` that control $w$ and $h$ and (2)
`nb_samples` that control $N$.
The pourcentage of visible pixels $\mathbb{E}(\mathcal{M})$ is controlled using the `preservation_probability` parameter.

## Example

```python
from xplique.attributions import Rise

# load images, labels and model
# ...

method = Rise(model, nb_samples=4000, grid_size=7, preservation_probability=0.5)
explanations = method.explain(images, labels)
```

## Notebooks

- [**Attribution Methods**: Getting started](https://colab.research.google.com/drive
/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2)
- [**RISE**: Going Further](https://colab.research.google.com/drive/1icu2b1JGfpTRa-ic8tBSXnqqfuCGW2mO)

{{xplique.attributions.rise.Rise}}

[^1]: [RISE: Randomized Input Sampling for Explanation of Black-box Models (2018)](https://arxiv.org/abs/1806.07421)
