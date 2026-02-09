# Average Increase in Confidence

- [**Metrics**: Average Drop/Increase/Gain Fidelity](https://colab.research.google.com/drive/1nGP13qiQrsJMBx8TXgA69D-5ALoP3l9p) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nGP13qiQrsJMBx8TXgA69D-5ALoP3l9p) </sub>

Average Increase in Confidence (AIC) measures the fraction of samples for which the masked input yields a higher score than the original input. This binary indicator provides a simple measure of explanation quality.

!!! quote
    We use the Average Increase metric to evaluate whether the saliency map highlights features that, when isolated, increase the model's confidence.

    -- <cite>[Grad-CAM++: Generalized Gradient-based Visual Explanations for Deep Convolutional Networks (2018)](https://arxiv.org/abs/1710.11063)</cite>[^1]

Formally, for each sample $i$:

$$ AIC_i = \mathbb{1}[after_i > base_i] $$

Where:
- $base_i = g(f, x_i, y_i)$ is the model score on the original input
- $after_i = g(f, x_i \odot M_i, y_i)$ is the model score after masking with explanation-derived mask $M_i$

The dataset-level AIC is the mean of all $AIC_i$ values.

!!!info
    The better the explanation, the **higher** the Average Increase score. A high score indicates that explanations capture truly discriminative features.

## Score Interpretation

- **Higher scores are better**: A high Average Increase means that for more samples, the explanation identifies features which, when isolated, are sufficient or even more predictive than the full input.
- Returns binary indicators (0 or 1); the dataset-level metric is the proportion of samples showing an increase.
- This metric reveals whether explanations capture truly discriminative features.

## Example

```python
from xplique.metrics import AverageIncreaseMetric
from xplique.attributions import Saliency

# load images, labels and model
# ...
explainer = Saliency(model)
explanations = explainer(inputs, labels)

metric = AverageIncreaseMetric(model, inputs, labels, activation="softmax")
score = metric.evaluate(explanations)
```

!!!tip
    This metric works best with probabilistic outputs. Set `activation="softmax"` or `"sigmoid"` if your model returns logits.

{{xplique.metrics.fidelity.AverageIncreaseMetric}}

[^1]: [Grad-CAM++: Generalized Gradient-based Visual Explanations for Deep Convolutional Networks (2018)](https://arxiv.org/abs/1710.11063)

