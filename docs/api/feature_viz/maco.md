# Modern Feature Visualization (MaCo)

Feature visualization has become increasingly popular, especially after the groundbreaking work by Olah et al. [^1], which established it as a vital tool for enhancing explainability. Despite its significance, the widespread adoption of feature visualization has been hindered by the reliance on various tricks to create interpretable images, making it challenging to scale the method effectively for deeper neural networks.

Addressing these limitations, a recent method called MaCo [^2] offers a straightforward solution. The core concept involves generating images by optimizing the phase spectrum while keeping the magnitude of the Fourier spectrum constant. This ensures that the generated images reside in the space of natural images in the Fourier domain, providing a more stable and interpretable approach.

!!! quote
    It is known that human recognition of objects in images is driven not by magnitude but by phase.
    Motivated by this, we propose to optimize the phase of the Fourier spectrum while fixing
    its magnitude to a typical value of a natural image (with few high frequencies). In particular, the
    magnitude is kept constant at the average magnitude computed over a set of natural images (such as ImageNet)

    <cite>[MaCo -- Unlocking Feature Visualization for Deeper Networks with MAgnitude Constrained Optimization (2023)](https://arxiv.org/pdf/2306.06805.pdf)</cite>[^2]

To put it more precisely, let $\phi^{(n)}$ be an explanation of a neuron $n$, and let $x* \in \mathcal{X}$ be the corresponding input defined as:

$$ \varphi* = \underset{\varphi}{arg\ max}\ f(\mathcal{F}^{-1}(r e^{i \varphi}))^{(n)} $$

where $x* = \mathcal{F}^{-1}(r e^{i \varphi*})$, $f(x)^{(n)}$ represents the neuron score for a given input, and $\mathcal{F}^{-1}$ denotes the 2-D inverse Fourier transform.

In the optimization process, MaCo also generates an alpha mask, which is used to identify the most important area of the generated image. For the purpose of correctly visualizing the image blended with the alpha mask, we provide utilities in the `xplique.plot` module.


## Notebooks

- [**MaCo**: Getting started](https://colab.research.google.com/drive/1l0kag1o-qMY4NCbWuAwnuzkzd9sf92ic) In this notebook, you'll be introduced to the fundamentals of MaCo while also experimenting with various hyperparameters.


## Examples

To optimize the logit 1 of your neural network (we recommend to remove the softmax activation of your network).

```python
from xplique.features_visualizations import Objective
from xplique.features_visualizations import maco
from xplique.plot import plot_maco
# load a model...

# targeting the logit 1 of the layer 'logits'
# we can also target a layer by its index, like -1 for the last layer
logits_obj = Objective.neuron(model, "logits", 1)
image, alpha = maco(logits_obj)
plot_maco(image, alpha)
```

Or if you want to visualize a specific CAV (or any direction, like multiple neurons) in your models:

```python
from xplique.features_visualizations import Objective
from xplique.features_visualizations import maco
from xplique.plot import plot_maco
# load a model...

# cav is a vector of the shape of an activation in the -2 layer
# e.g 2048 for Resnet50
logits_obj = Objective.direction(model, -2, cav)
image, alpha = maco(logits_obj)
plot_maco(image, alpha)
```

{{xplique.features_visualizations.maco.maco}}

[^1]:[Feature Visualization -- How neural networks build up their understanding of images (2017)](https://distill.pub/2017/feature-visualization)
[^2]:[MaCo -- Unlocking Feature Visualization for Deeper Networks with MAgnitude Constrained Optimization (2023)](https://arxiv.org/pdf/2306.06805.pdf)
