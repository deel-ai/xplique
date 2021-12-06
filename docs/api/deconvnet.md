# Deconvnet

## Example

```python
from xplique.attributions import DeconvNet

# load images, labels and model
# ...

method = DeconvNet(model)
explanations = method.explain(images, labels)
```

{{xplique.attributions.DeconvNet}}
