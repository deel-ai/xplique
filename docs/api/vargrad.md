# VarGrad

Similar to SmoothGrad, VarGrad a variance analog of SmoothGrad, and can be defined as follows:

$$ \phi_x = \underset{\xi ~\sim\ \mathcal{N}(0, \sigma^2)}{\mathcal{V}}
                    \Big{[}\frac { \partial{S_c(x + \xi)} } { \partial{x} }\Big{]}$$

with $S_c$ the unormalized class score (layer before softmax). The $\sigma$ in the formula is controlled using the noise
parameter.

## Example

```python
from xplique.attributions import VarGrad

# load images, labels and model
# ...

method = VarGrad(model, nb_samples=50, noise=0.5)
explanations = method.explain(images, labels)
```

{{xplique.methods.VarGrad}}