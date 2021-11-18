# InsertionTS

The Time Series Insertion Fidelity metric measures the faithfulness of explanations on Time Series predictions[^2].
This metric computes the capacity of the model to make predictions while only the most important features are not perturbed[^1].


## Score interpretation

The interpretation of the score depends on the score metric you are using to evaluate your model.
- For metrics where the score increases with the performance of the model (such as accuracy).
If explanations are accurate, the score will quickly rise to the score on non-perturbed input.
  Thus, in this case, a higher score represent a more accurate explanation.
  
- For metrics where the score decreases with the performance of the model (such as losses). 
If explanations are accurate, the score will quickly fall to the score on non-perturbed input.
  Thus, in this case, a lower score represent a more accurate explanation.


## Remarks

This metric only evaluate the order of importance between features.

The parameters metric, steps and max_percentage_perturbed may drastically change the score :

- For inputs with many features, increasing the number of steps will allow you to capture more efficiently the difference between attributions methods.

- The order of importance of features with low importance may not matter, hence, decreasing the max_percentage_perturbed,
may make the score more relevant.
  
Sometimes, attributions methods also returns negative attributions,
for those methods, do not take the absolute value before computing insertion and deletion metrics.
Otherwise, negative attributions may have higher absolute values, and the order of importance between features will change.
Therefore, take those previous remarks into account to get a relevant score.


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
