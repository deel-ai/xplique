# Insertion

The Insertion Fidelity metric measures how well a saliency-map–based explanation can find elements that are minimal for the predictions.

!!! quote
    The insertion metric, on the other hand, captures the importance of the
    pixels in terms of their ability to synthesize an image and is measured by the rise in the
    probability of the class of interest as pixels are added according to the generated importance
    map.

    -- <cite>[RISE: Randomized Input Sampling for Explanation of Black-box Models (2018)](https://arxiv.org/abs/1806.07421)</cite>[^1]


## Score interpretation

If explanations are accurate, the score will quickly rise to the score on non-perturbed input.
  Thus, in this case, a higher score represent a more accurate explanation.


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
from xplique.metrics import Insertion
from xplique.attributions import Saliency

# load images, labels and model
# ...
explainer = Saliency(model)
explanations = explainer(inputs, labels)

metric = Insertion(model, inputs, labels)
score = metric.evaluate(explanations)
```

{{xplique.metrics.Insertion}}

[^1]:[RISE: Randomized Input Sampling for Explanation of Black-box Models (2018)](https://arxiv.org/abs/1806.07421)
