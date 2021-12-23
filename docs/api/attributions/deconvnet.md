# Deconvnet

## Example

```python
from xplique.attributions import DeconvNet

# load images, labels and model
# ...

method = DeconvNet(model)
explanations = method.explain(images, labels)
```

## Notebooks

- [**Attribution Methods**: Getting started](https://colab.research.google.com/drive
/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) </sub>
- [**DeconvNet**: Going Further](https://colab.research.google.com/drive/1qBxwsMILPvQs3WLLcX_hRb3kzTSI4rkz) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19oLUjmvrBIMTmNKXcJNbB6pJvkfutLEb) </sub>

{{xplique.attributions.deconvnet.DeconvNet}}
