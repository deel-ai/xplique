# Deletion

The Deletion Fidelity metric measures how well a saliency-map–based explanation of an image classification result localizes
the important pixels.

!!! quote
    The deletion metric measures the drop in the probability of a class as important pixels (given
    by the saliency map) are gradually removed from the image. A sharp drop, and thus a small
    area under the probability curve, are indicative of a good explanation.

    -- <cite>[RISE: Randomized Input Sampling for Explanation of Black-box Models (2018)](https://arxiv.org/abs/1806.07421)</cite>[^1]


## Score interpretation

If explanations are accurate, the score will quickly fall from the score on non-perturbed input to the score of a random predictor.
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
from xplique.metrics import Deletion
from xplique.attributions import Saliency

# load images, targets and model
# ...
explainer = Saliency(model)
explanations = explainer(inputs, targets)

metric = Deletion(model, inputs, targets)
score = metric.evaluate(explanations)
```

{{xplique.metrics.Deletion}}

[^1]:[RISE: Randomized Input Sampling for Explanation of Black-box Models (2018)](https://arxiv.org/abs/1806.07421)
