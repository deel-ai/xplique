# Average Gain

Average Gain (AG) measures the relative increase in the model's confidence when the input is masked according to the explanation. This metric is complementary to Average Drop and evaluates whether explanations capture truly discriminative features.

!!! quote
    We define Opti-CAM, a saliency method that obtains the saliency map by directly optimizing for the Average Gain metric.

    -- <cite>[Opti-CAM: Optimizing saliency maps for interpretability (2024)](https://www.sciencedirect.com/science/article/pii/S1077314224001644)</cite>[^1]

Formally, for each sample $i$:

$$ AG_i = \frac{\text{ReLU}(after_i - base_i)}{1 - base_i + \epsilon} $$

Where:
- $base_i = g(f, x_i, y_i)$ is the model score on the original input
- $after_i = g(f, x_i \odot M_i, y_i)$ is the model score after masking with explanation-derived mask $M_i$

!!!info
    The better the explanation, the **higher** the Average Gain score. A high score indicates that isolated important features are sufficient to maintain or increase the model's confidence.

## Score Interpretation

- **Higher scores are better**: A high Average Gain means that the explanation successfully identifies features which, when isolated, are sufficient to maintain or increase the model's confidence.
- The ReLU ensures we only measure gains, not decreases in confidence.
- The normalization by $(1 - base_i)$ accounts for the remaining headroom to achieve a perfect score.

## Example

```python
from xplique.metrics import AverageGainMetric
from xplique.attributions import Saliency

# load images, labels and model
# ...
explainer = Saliency(model)
explanations = explainer(inputs, labels)

metric = AverageGainMetric(model, inputs, labels, activation="softmax")
score = metric.evaluate(explanations)
```

!!!tip
    This metric is intended for scores in [0, 1]. If your model outputs logits, use `activation="softmax"` or `"sigmoid"` at construction to operate on probabilities.

{{xplique.metrics.fidelity.AverageGainMetric}}

[^1]: [Opti-CAM: Optimizing saliency maps for interpretability (2024)](https://www.sciencedirect.com/science/article/pii/S1077314224001644)

