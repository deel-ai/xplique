# Guided Backpropagation

## Example

```python
from xplique.attributions import GuidedBackprop

# load images, labels and model
# ...

method = GuidedBackprop(model)
explanations = method.explain(images, labels)
```

## Notebooks

- [**Attribution Methods**: Getting started](https://colab.research.google.com/drive
/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) </sub>
- [**Guided Backprop**: Going Further](https://colab.research.google.com/drive/16cmbKC0b6SVl1HjhOKhLTNak3ytm1Ib1) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16cmbKC0b6SVl1HjhOKhLTNak3ytm1Ib1) </sub>

{{xplique.attributions.guided_backpropagation.GuidedBackprop}}
