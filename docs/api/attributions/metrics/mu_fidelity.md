# MuFidelity

MuFidelity is a fidelity metric measuring the correlation between important variables defined by the explanation method and the decline in the model score when these variables are reset to a baseline state.

!!! quote
    [...]  when we set particular features $x_s$ to a baseline value $x_0$ the change in predictor’s
    output should be proportional to the sum of attribution scores.

    -- <cite>[Evaluating and Aggregating Feature-based Model Explanations (2020)](https://arxiv.org/abs/2005.00631)</cite>[^1]

Formally, given a predictor $f$, an explanation function $g$, a point $x \in \mathbb{R}^n$ and a subset size $k$ the MuFidelity metric is defined as:

$$ \mu F = \underset{S \subseteq \{1, ..., d\} \\ |S| = k}{Corr}( \sum_{i \in S} g(f, x)_i, f(x) - f(x_{[x_i = x_0 | i \in S]})) $$

!!!info
    The better the method, the higher the score.

## Example

```python
from xplique.metrics import MuFidelity
from xplique.attributions import Saliency

# load images, labels and model
# ...
explainer = Saliency(model)
explanations = explainer(inputs, lablels)

metric = MuFidelity(model, inputs, labels)
score = metric.evaluate(explainations)
```

{{xplique.metrics.MuFidelity}}

[^1]:[Evaluating and Aggregating Feature-based Model Explanations (2020)](https://arxiv.org/abs/2005.00631)
