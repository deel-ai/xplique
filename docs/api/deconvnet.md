# Deconvnet

## Example

```python
from xplique.methods import DeconvNet

# load images, labels and model
# ...

method = DeconvNet(model)
explanations = method.explain(images, labels)
```

{{xplique.methods.DeconvNet}}