# Integrated Gradients

<sub><img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20"></sub>[ View colab tutorial](https://colab.research.google.com/drive/1UXJYVebDVIrkTOaOl-Zk6pHG3LWkPcLo?authuser=1) | <sub><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20"></sub>[ View source](https://github.com/deel-ai/xplique/blob/master/xplique/attributions/integrated_gradients.py)

Integrated Gradients is a visualization technique resulting of a theoretical search for an
explanatory method that satisfies two axioms, Sensitivity and Implementation Invariance
(Sundararajan et al[^1]).

!!! quote
    We consider the straightline path (in $R^n$) from the baseline $\bar{x}$ to the input $x$, and compute the
    gradients at all points along the path. Integrated gradients are obtained by cumulating these
    gradients.

    -- <cite>[Axiomatic Attribution for Deep Networks (2017)](https://arxiv.org/abs/1703.01365)</cite>[^1]

Rather than calculating only the gradient relative to the image, the method consists of averaging
the gradient values along the path from a baseline state to the current value. The baseline state
is often set to zero, representing the complete absence of features.

More precisely, with $\bar{x}$ the baseline state, $x$ the image, $c$ the class of interest and
$S_c$ the unormalized class score (layer before softmax). The Integrated Gradient is defined as

$$IG(x) = (x - \bar{x}) \cdot \int_0^1{ \frac { \partial{S_c(\tilde{x})} } { \partial{\tilde{x}} }
            \Big|_{ \tilde{x} = \bar{x} + \alpha(x - \bar{x}) } d\alpha }$$


In order to approximate from a finite number of steps, the implementation here use the
Trapezoidal rule[^3] and not a left-Riemann summation, which allows for more accurate results
and improved performance. (see the paper below for a comparison of the methods[^2]).

## Example

```python
from xplique.attributions import IntegratedGradients

# load images, labels and model
# ...

method = IntegratedGradients(model, steps=50, baseline_value=0.0)
explanations = method.explain(images, labels)
```

## Notebooks

- [**Attribution Methods**: Getting started](https://colab.research.google.com/drive
/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2)
- [**Integrated Gradients**: Going Further](https://colab.research.google.com/drive/1UXJYVebDVIrkTOaOl-Zk6pHG3LWkPcLo?authuser=1)

{{xplique.attributions.integrated_gradients.IntegratedGradients}}

[^1]: [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365)
[^2]: [Computing Linear Restrictions of Neural Networks](https://arxiv.org/abs/1908.06214)
[^3]: [Trapezoidal rule](https://en.wikipedia.org/wiki/Trapezoidal_rule)
