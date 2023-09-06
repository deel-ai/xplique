# FORGrad

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20">
</sub> [View colab tutorial](https://colab.research.google.com/drive/1ibLzn7r9QQIEmZxApObowzx8n9ukinYB) |
<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub> [View source](https://github.com/deel-ai/xplique/blob/master/xplique/attributions/forgrad.py) |
📰 [Paper](https://arxiv.org/pdf/2307.09591.pdf)

ForGrad is an enhancement for any attribution method by effectively filtering
out high-frequency noise in gradient-based attribution maps, resulting in improved explainability scores and
promoting the adoption of more computationally efficient techniques for model interpretability.

!!! quote
    The application of an optimal low-pass filter to attribution maps improves gradient-based attribution methods significantly, resulting in higher explainability scores across multiple models and elevating gradient-based methods to a top ranking among state-of-the-art techniques, sparking renewed interest in simpler and more computationally efficient explainability approaches.

    -- <cite>[Gradient strikes back: How filtering out high frequencies improves explanations (2023)](https://arxiv.org/pdf/2307.09591.pdf)</cite>[^1]

In a more precise manner, to obtain an attribution map $\varphi_\sigma(x)$, we apply a filter $w_\sigma$ with a cutoff value $\sigma$ to remove high frequencies, as shown in the equation:

$$ \varphi_\sigma(x) = \mathcal{F}^{-1}((\mathcal{F} \cdot \varphi)(x) \odot w_\sigma) $$

The parameter $\sigma$ controls the amount of frequencies retained and ranges between $(0, W]$, where $W$ represents the dimension of the squared image. A value of $0$ eliminates all frequencies, while $W$ retains all frequencies. The paper presents a method to estimate the optimal cutoff, and for ImageNet images, the recommended default value for the optimal sigma is typically around 15.


## Example

```python
from xplique.attributions import Saliency
from xplique.common import forgrad

# load images, labels and model
# ...

method = Saliency(model)
explanations = method.explain(images, labels)
explanations_filtered = forgrad(explanations, sigma=15)
```

## Notebooks

- [**FORGRad**: Gradient strikes back with FORGrad](https://colab.research.google.com/drive/1ibLzn7r9QQIEmZxApObowzx8n9ukinYB)
- [**Attribution Methods**: Getting started](https://colab.research.google.com/drive
/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2)


{{xplique.commons.forgrad.forgrad}}


[^1]: [Gradient strikes back: How filtering out high frequencies improves explanations (2023)](https://arxiv.org/pdf/2307.09591.pdf)