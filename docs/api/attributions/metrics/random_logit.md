# Random Logit Metric

Random Logit Invariance metric tests whether explanations change when the target logit is randomized to a different class. This is a sanity check to verify that explainers are sensitive to the target label.

!!! quote
    We propose sanity checks for saliency methods. [...] We find that some widely deployed saliency methods are independent of both the data the model was trained on, and the model parameters.

    -- <cite>[Sanity Checks for Saliency Maps (2018)](https://arxiv.org/abs/1810.03292)</cite>[^1]

For each sample $(x, y)$:

1. Compute explanation for the true class $y$
2. Randomly draw an off-class $y' \neq y$
3. Compute explanation for $y'$
4. Measure SSIM (Structural Similarity Index) between both explanations

!!!info
    A **low SSIM** indicates that explanations are sensitive to the target label (desirable if we expect class-specific explanations).

## Score Interpretation

- **Lower scores are better**: A low SSIM means the explanations change significantly when the target class changes, indicating the explainer is properly sensitive to the target.
- Values range from -1 to 1, where 1 means identical explanations.
- High SSIM values suggest the explainer may not be faithfully explaining class-specific features.

## Example

```python
from xplique.metrics import RandomLogitMetric
from xplique.attributions import Saliency

# load images, labels and model
# ...
explainer = Saliency(model)

metric = RandomLogitMetric(model, inputs, labels)
score = metric.evaluate(explainer)
```

!!!warning
    This metric requires one-hot encoded labels with shape `(N, C)` where C is the number of classes.

{{xplique.metrics.randomization.RandomLogitMetric}}

[^1]: [Sanity Checks for Saliency Maps (2018)](https://arxiv.org/abs/1810.03292)
