# SquareGrad

## Example

```python
from xplique.methods import SquareGrad

# load images, labels and model
# ...

method = SquareGrad(model, nb_samples=50, noise=0.5)
explanations = method.explain(images, labels)
```

{{xplique.methods.SquareGrad}}