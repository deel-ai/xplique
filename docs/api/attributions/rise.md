# RISE

<sub><img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20"></sub>[ View colab tutorial](https://colab.research.google.com/drive/1icu2b1JGfpTRa-ic8tBSXnqqfuCGW2mO?authuser=1) | <sub><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20"></sub>[ View source](https://github.com/deel-ai/xplique/blob/master/xplique/attributions/rise.py)

The RISE method consist of probing the model with randomly masked versions of the input image and
obtaining the corresponding outputs to deduce critical areas.

!!! quote
    \[...] we estimate the importance of pixels by dimming them in random combinations,
    reducing their intensities down to zero. We model this by multiplying an image with a \[0,1\]
    valued mask.

     -- <cite>[RISE: Randomized Input Sampling for Explanation of Black-box Models (2018)](https://arxiv.org/abs/1806.07421)</cite>[^1]


with $S_c$ the class score **after softmax**, $x$ an input, and $m  \sim \mathcal{M}$ a mask (not
 binary) the RISE importance map $\phi$ is defined as :

$$ \phi_i = \frac{1}{\mathbb{E}(\mathcal{M}) N} \sum_{i=1}^N S_c(x \odot m_i) m_i $$

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
- [**RISE**: Going Further](https://colab.research.google.com/drive/1icu2b1JGfpTRa-ic8tBSXnqqfuCGW2mO?authuser=1)

{{xplique.attributions.rise.Rise}}

[^1]: [RISE: Randomized Input Sampling for Explanation of Black-box Models (2018)](https://arxiv.org/abs/1806.07421)
