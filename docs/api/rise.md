# RISE

The RISE method consist of probing the model with randomly masked versions of the input image and
obtaining the corresponding outputs to deduce critical areas.

!!! quote
    \[...] we estimate the importance of pixels by dimming them in random combinations,
    reducing their intensities down to zero. We model this by multiplying an image with a \[0,1\]
    valued mask.

     -- <cite>[RISE: Randomized Input Sampling for Explanation of Black-box Models (2018)](https://arxiv.org/abs/1806.07421)</cite>[^1]


with $S_c$ the class score **after softmax**, $x$ an input, and $m  \sim \mathcal{M}$ a mask (not
 binary) the RISE importance map $\phi$ is defined as :

$$ \phi_i = \frac{1}{\mathbb{E}(\mathcal{M}) N} \sum_{i=0}^N S_c(x \odot m_i) m_i $$

## Example

```python
from xplique.attributions import Rise

# load images, labels and model
# ...

method = Rise(model, nb_samples=4000, grid_size=7, preservation_probability=0.5)
explanations = method.explain(images, labels)
```

{{xplique.attributions.rise.Rise}}

[^1]: [RISE: Randomized Input Sampling for Explanation of Black-box Models (2018)](https://arxiv.org/abs/1806.07421)
