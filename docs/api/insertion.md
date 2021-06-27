# Deletion

The Insertion Fidelity metric measures how well a saliency-map–based explanation can find elements that are minimal for the predictions.

!!! quote
    The insertion metric, on the other hand, captures the importance of the
    pixels in terms of their ability to synthesize an image and is measured by the rise in the
    probability of the class of interest as pixels are added according to the generated importance
    map. 
    
    -- <cite>[RISE: Randomized Input Sampling for Explanation of Black-box Models (2018)](https://arxiv.org/abs/1806.07421)</cite>[^1]

The better the method, the higher the score.

## Example

```python
from xplique.metrics import Insertion
from xplique.attributions import Saliency

# load images, labels and model
# ...
explainer = Saliency(model)

metric = Insertion(model, inputs, labels)
score = metric.evaluate(explainer)
```

{{xplique.metrics.fidelity.Insertion}}

[^1]:[RISE: Randomized Input Sampling for Explanation of Black-box Models (2018)](https://arxiv.org/abs/1806.07421)