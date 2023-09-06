# Average Stability

Average Stability is a Stability metric measuring how similar are explanations of similar inputs.

!!! quote
    [...]  We want to ensure that, if inputs are near each other and their model outputs are similar, then their explanations should be close to each other.

    -- <cite>[Evaluating and Aggregating Feature-based Model Explanations (2020)](https://arxiv.org/abs/2005.00631)</cite>[^1]

Formally, given a predictor $f$, an explanation function $g$, a point $x$, a radius $r$ and a two distance metric: $\rho$ over the inputs and $D$ over the explanations, the AverageStability is defined as:

$$ S = \underset{z : \rho(x, z) \leq r}{\int} D(g(f, x), g(f, z))\ dz $$

!!!info
    The better the method, the smaller the score.

## Example

```python
from xplique.metrics import AverageStability
from xplique.attributions import Saliency

# load images, labels and model
# ...
explainer = Saliency(model)

metric = AverageStability(model, inputs, labels)
score = metric.evaluate(explainer)
```

{{xplique.metrics.stability.AverageStability}}

[^1]:[Evaluating and Aggregating Feature-based Model Explanations (2020)](https://arxiv.org/abs/2005.00631)

!!!warning
    AverageStability will compute several time explanations for all the inputs (pertubed more or less severly).
    Thus, it might be very long to compute (especially if the explainer is already time consumming).
