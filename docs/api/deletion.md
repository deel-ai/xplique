# Deletion

The Deletion Fidelity metric measures how well a saliency-map–based explanation of an image classification result localizes
the important pixels.

!!! quote
    The deletion metric measures the drop in the probability of a class as important pixels (given
    by the saliency map) are gradually removed from the image. A sharp drop, and thus a small
    area under the probability curve, are indicative of a good explanation.
    
    -- <cite>[RISE: Randomized Input Sampling for Explanation of Black-box Models (2018)](https://arxiv.org/abs/1806.07421)</cite>[^1]

The better the method, the smaller the score.

## Example

```python
from xplique.metrics import Deletion
from xplique.attributions import Saliency

# load images, labels and model
# ...
explainer = Saliency(model)

metric = Deletion(model, inputs, labels)
score = metric.evaluate(explainer)
```

{{xplique.metrics.fidelity.Deletion}}

[^1]:[RISE: Randomized Input Sampling for Explanation of Black-box Models (2018)](https://arxiv.org/abs/1806.07421)