# Square Grad

<sub><img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20"></sub>[ View colab tutorial](https://colab.research.google.com/drive/12-tlM_TdZ12oc5lNL2S2g-hcMJV8tZUD?authuser=1) | <sub><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20"></sub>[ View source](https://github.com/deel-ai/xplique/blob/master/xplique/attributions/square_grad.py) | 📰 [ See paper](https://arxiv.org/abs/1806.10758)

Similar to SmoothGrad, Square Grad average the square of the gradients.

$$
\phi_x = \underset{\xi ~\sim~ \mathcal{N}(0, \sigma^2)}{\mathbb{E}}
            \Big{[}\Big{(}
             \frac { \partial{S_c(x + \xi)} } { \partial{x} }
             \Big{)}^2\Big{]}
$$

with $S_c$ the unormalized class score (layer before softmax). The $\sigma$ in the formula is controlled using the noise
parameter.

## Example

```python
from xplique.attributions import SquareGrad

# load images, labels and model
# ...

method = SquareGrad(model, nb_samples=50, noise=0.15)
explanations = method.explain(images, labels)
```

## Notebooks

- [**Attribution Methods**: Getting started](https://colab.research.google.com/drive
/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2)
- [**SquareGrad**: Going Further](https://colab.research.google.com/drive/12-tlM_TdZ12oc5lNL2S2g-hcMJV8tZUD?authuser=1)

{{xplique.attributions.SquareGrad}}
