# DeletionTS

The Time Series Deletion Fidelity metric measures the faithfulness of attributions on Time Series predictions[^1].
This metric computes the capacity of the model to make predictions while perturbing only the most important features.

Specific explanation metrics for time series are necessary because time series and images have different shapes (number of dimensions) and perturbations should be applied differently to them.
As the insertion and deletion metrics use input perturbation to be computed, creating new metrics for time series is natural[^2].

The better the method, the smaller the score.

## Example

```python
from xplique.metrics import DeletionTS
from xplique.attributions import Saliency

# load time series, targets and model
# ...
explainer = Saliency(model)
explanations = explainer(inputs, targets)

metric = DeletionTS(model, inputs, targets)
score = metric.evaluate(explanations)
```

{{xplique.metrics.DeletionTS}}

[^1]: [RISE: Randomized Input Sampling for Explanation of Black-box Models (2018)](https://arxiv.org/abs/1806.07421)
[^2]: [Towards a Rigorous Evaluation of XAI Methods on Time Series (2019)](https://arxiv.org/abs/1909.07082)
