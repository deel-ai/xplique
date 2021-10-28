# InsertionTS

The Time Series Insertion Fidelity metric measures the faithfulness of explanations on Time Series predictions[^2].
This metric computes the capacity of the model to make predictions while only the most important features are not perturbed[^1].

The better the method, the higher the score.

## Example

```python
from xplique.metrics import InsertionTS
from xplique.attributions import Saliency

# load time series, labels and model
# ...
explainer = Saliency(model)
explanations = explainer(inputs, labels)

metric = InsertionTS(model, inputs, labels)
score = metric.evaluate(explanations)
```

{{xplique.metrics.InsertionTS}}

[^1]: [RISE: Randomized Input Sampling for Explanation of Black-box Models (2018)](https://arxiv.org/abs/1806.07421)
[^2]: [Towards a Rigorous Evaluation of XAI Methods on Time Series (2019)](https://arxiv.org/abs/1909.07082)
