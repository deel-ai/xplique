# Square Grad

Similar to SmoothGrad, Square Grad average the square of the gradients.

$$ \phi_x = \underset{\xi ~\sim~ \mathcal{N}(0, \sigma^2)}{\mathbb{E}}
            \Big{[}\Big{(}
             \frac { \partial{S_c(x + \xi)} } { \partial{x} } 
             \Big{)}^2\Big{]} $$


with $S_c$ the unormalized class score (layer before softmax). The $\sigma$ in the formula is controlled using the noise
parameter.

## Example

```python
from xplique.methods import SquareGrad

# load images, labels and model
# ...

method = SquareGrad(model, nb_samples=50, noise=0.5)
explanations = method.explain(images, labels)
```

{{xplique.methods.SquareGrad}}