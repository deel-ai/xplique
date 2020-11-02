# Guided Backpropagation

## Example

```python
from xplique.methods import GuidedBackprop

# load images, labels and model
# ...

method = GuidedBackprop(model)
explanations = method.explain(images, labels)
```

{{xplique.methods.GuidedBackprop}}