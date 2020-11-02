# VarGrad

## Example

```python
from xplique.methods import VarGrad

# load images, labels and model
# ...

method = VarGrad(model, nb_samples=50, noise=0.5)
explanations = method.explain(images, labels)
```

{{xplique.methods.VarGrad}}