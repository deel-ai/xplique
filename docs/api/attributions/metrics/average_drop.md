# Average Drop

Average Drop (AD) measures the relative decrease in the model's confidence when the input is masked according to the explanation. A good explanation should identify features that, when masked, cause the model's score to drop significantly.

!!! quote
    We define Average Drop % as the percentage of positive drops averaged over all images.

    -- <cite>[Grad-CAM++: Generalized Gradient-based Visual Explanations for Deep Convolutional Networks (2018)](https://arxiv.org/abs/1710.11063)</cite>[^1]

Formally, for each sample $i$:

$$ AD_i = \frac{\text{ReLU}(base_i - after_i)}{base_i + \epsilon} $$

Where:
- $base_i = g(f, x_i, y_i)$ is the model score on the original input
- $after_i = g(f, x_i \odot M_i, y_i)$ is the model score after masking with explanation-derived mask $M_i$

!!!info
    The better the explanation, the **lower** the Average Drop score. A low score indicates that the explanation correctly identifies important features.

## Score Interpretation

- **Lower scores are better**: A low Average Drop means that important features (as identified by the explanation) are correctly captured. When these features are preserved (via masking), the model's confidence doesn't drop much.
- The ReLU ensures we only measure drops, not increases in confidence.
- The normalization by $base_i$ makes the metric scale-invariant.

## Example

```python
from xplique.metrics import AverageDropMetric
from xplique.attributions import Saliency

# load images, labels and model
# ...
explainer = Saliency(model)
explanations = explainer(inputs, labels)

metric = AverageDropMetric(model, inputs, labels, activation="softmax")
score = metric.evaluate(explanations)
```

!!!tip
    This metric works best with probabilistic outputs. Set `activation="softmax"` or `"sigmoid"` if your model returns logits.

{{xplique.metrics.fidelity.AverageDropMetric}}

[^1]: [Grad-CAM++: Generalized Gradient-based Visual Explanations for Deep Convolutional Networks (2018)](https://arxiv.org/abs/1710.11063)
