# Guided Backpropagation

## Example

```python
from xplique.attributions import GuidedBackprop

# load images, labels and model
# ...

method = GuidedBackprop(model)
explanations = method.explain(images, labels)
```

{{xplique.attributions.GuidedBackprop}}